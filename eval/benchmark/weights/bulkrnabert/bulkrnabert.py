import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import PretrainedConfig, PreTrainedModel


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        key_size: int,
        add_bias_kv: bool = False,
        value_size: Optional[int] = None,
        model_size: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        if not model_size:
            model_size = key_size
        if not value_size:
            value_size = key_size
        self.model_size = model_size
        self.key_size = key_size
        self.value_size = value_size
        self.add_bias_kv = add_bias_kv
        self.name = name
        self.num_heads = num_heads

        self.w_k = nn.Linear(self.model_size, self.num_heads * self.key_size)
        self.w_q = nn.Linear(self.model_size, self.num_heads * self.key_size)
        self.w_v = nn.Linear(self.model_size, self.num_heads * self.value_size)
        self.output = nn.Linear(self.num_heads * self.value_size, self.model_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_weight_bias: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dictionary containing attention weights
            and outputs.
        """
        key_heads = self.w_k(key).reshape(
            (*key.shape[:-1], self.num_heads, self.key_size)
        )
        query_heads = self.w_q(query).reshape(
            (*query.shape[:-1], self.num_heads, self.key_size)
        )
        value_heads = self.w_v(value).reshape(
            (*value.shape[:-1], self.num_heads, self.value_size)
        )
        attention_weights = torch.einsum(
            "...thd, ...Thd -> ...htT", query_heads, key_heads
        )
        sqrt_key_size = np.sqrt(self.key_size)
        attention_weights = attention_weights / sqrt_key_size
        if attention_mask is not None:
            attention_weights = torch.where(attention_mask, attention_weights, -1e30)
        if attention_weight_bias:
            attention_weights = F.softmax(
                attention_weights + attention_weight_bias, dim=-1
            )
        else:
            attention_weights = F.softmax(attention_weights, dim=-1)
        value_out = torch.einsum(
            "...htT, ...Thd->...thd", attention_weights, value_heads
        )
        value_out = value_out.reshape((*value_out.shape[:-2], -1))
        embeddings = self.output(value_out)

        return {"attention_weights": attention_weights, "embeddings": embeddings}


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        ffn_embed_dim: int,
        key_size: Optional[int] = None,
        add_bias_kv: bool = False,
        add_bias_fnn: bool = True,
        ffn_activation_name: str = "gelu-no-approx",
        use_glu_in_ffn: bool = False,
        layer_norm_eps: float = 1e-5,  # this is the default haiku value
        pre_layer_norm: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__()
        if key_size is None:
            if embed_dim % num_heads != 0:
                raise ValueError(
                    f"The embedding dimension should be divisible by the number of "
                    f"heads, however provided embedding dimension is {embed_dim} and "
                    f"the number of heads is {num_heads}."
                )
            else:
                key_size = embed_dim // num_heads

        # Get ffn activation function
        self._pre_layer_norm = pre_layer_norm
        self._use_glu_in_fnn = use_glu_in_ffn
        # Define layers
        if use_glu_in_ffn:
            # user should multiply ffn_embed_dim by 2/3 when using GLU
            # to keep total number of parameters equal
            # see https://arxiv.org/pdf/2002.05202.pdf. for more details
            # we multiply by 2 here as the output will be split in 2 for GLU
            self.fc1 = nn.Linear(embed_dim, int(2 * ffn_embed_dim), bias=add_bias_fnn)
        else:
            self.fc1 = nn.Linear(embed_dim, ffn_embed_dim, bias=add_bias_fnn)

        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim, bias=add_bias_fnn)

        self.layer_norm_self_attention = nn.LayerNorm(
            embed_dim,
        )
        self.layer_norm_mlp = nn.LayerNorm(embed_dim)
        if ffn_activation_name == "swish":
            self._ffn_activation_fn = nn.SiLU()
        elif ffn_activation_name == "gelu-no-approx":
            self._ffn_activation_fn = lambda x: F.gelu(x, approximate="none")
        else:
            self._ffn_activation_fn = getattr(torch.nn, ffn_activation_name)

        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_size=key_size,
            add_bias_kv=add_bias_kv,
            model_size=embed_dim,
            name="self_attention",
        )

    def mlp(self, embed: torch.Tensor) -> torch.Tensor:

        if self._pre_layer_norm:
            x = self.layer_norm_mlp(embed)
        else:
            x = embed

        if self._use_glu_in_fnn:
            x = self.fc1(x)
            x1, x2 = torch.split(x, split_size_or_sections=x.shape[-1] // 2, dim=-1)
            x = self._ffn_activation_fn(x1) * x2
        else:
            x = self._ffn_activation_fn(self.fc1(x))
        x = self.fc2(x)

        if not self._pre_layer_norm:
            x = self.layer_norm_mlp(x + embed)
        return x

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_weight_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        res = x
        if self._pre_layer_norm:
            x = self.layer_norm_self_attention(x)

        output = self.mha(
            x,
            x,
            x,
            attention_mask=attention_mask,
            attention_weight_bias=attention_weight_bias,
        )

        if not self._pre_layer_norm:
            output["embeddings"] = self.layer_norm_self_attention(
                output["embeddings"] + res
            )

            x = output["embeddings"]
        else:
            x = output["embeddings"]
            x = res + x

        # MLP
        if not self._pre_layer_norm:
            x = self.mlp(x)
        else:
            x = x + self.mlp(x)

        output["embeddings"] = x
        return output


class BulkRNABertConfig(PretrainedConfig):
    model_type = "BulkRNABert"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n_genes = kwargs.get("n_genes", 19_062)
        self.n_expressions_bins = kwargs.get("n_expressions_bins", 64)
        self.embed_dim = kwargs.get("embed_dim", 256)
        self.init_gene_embed_dim = kwargs.get("init_gene_embed_dim", 200)
        self.use_gene_embedding = kwargs.get("use_gene_embedding", True)
        self.project_gene_embedding = kwargs.get("project_gene_embedding", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.key_size = kwargs.get("key_size", None)
        self.ffn_embed_dim = kwargs.get("ffn_embed_dim", 512)
        self.num_layers = kwargs.get("num_layers", 4)

        # return
        self.embeddings_layers_to_save: tuple[int, ...] = kwargs.get(
            "embeddings_layers_to_save", ()
        )
        self.attention_maps_to_save: list[tuple[int, int]] = kwargs.get(
            "attention_maps_to_save", []
        )

        self.__post_init__()

    def __post_init__(self):
        # Validate attention key size
        key_size = self.key_size
        if key_size is None:
            embed_dim = self.embed_dim
            num_attention_heads = self.num_attention_heads
            if not embed_dim % num_attention_heads == 0:
                raise ValueError(
                    f"When no key size is provided, the embedding dimension should be "
                    f"divisible by the number of heads, however provided embedding "
                    f"dimension is {embed_dim} and the number of heads is "
                    f"{num_attention_heads}."
                )
            self.key_size = embed_dim // num_attention_heads

        # Validate gene embedding projection
        use_gene_embedding = self.use_gene_embedding
        if use_gene_embedding:
            init_gene_embed_dim = self.init_gene_embed_dim
            embed_dim = self.embed_dim
            if init_gene_embed_dim != embed_dim:
                project_gene_embedding = self.project_gene_embedding
                if not project_gene_embedding:
                    logging.warning(
                        f"Init gene embedding dimension ({init_gene_embed_dim})"
                        f"different than embedding dimension ({embed_dim})."
                        f"Setting `project_gene_embedding` to True"
                    )
                    self.project_gene_embedding = True


class BulkRNABert(PreTrainedModel):
    config_class = BulkRNABertConfig

    def __init__(self, config: BulkRNABertConfig):
        super().__init__(config=config)

        self.expression_embedding_layer = nn.Embedding(
            config.n_expressions_bins, config.embed_dim
        )
        self.gene_embedding_layer = nn.Embedding(
            config.n_genes,
            config.init_gene_embed_dim,
        )
        self.fc_gene_embedding = nn.Linear(config.init_gene_embed_dim, config.embed_dim)

        attention_maps_to_save = config.attention_maps_to_save
        self._attention_layers_to_save = list({t[0] for t in attention_maps_to_save})

        self._attention_maps_per_layer_to_save = {
            layer: [t[1] for t in attention_maps_to_save if t[0] == layer]
            for layer in self._attention_layers_to_save
        }
        max_layer = max(self._attention_layers_to_save + [0])
        if max_layer > config.num_layers:
            raise ValueError(
                f"You are requiring attention maps for layer {max_layer}, "
                f"while the model has {config.num_layers} layers only."
            )
        self.transformer_layers = nn.ModuleList(
            [
                SelfAttentionBlock(
                    num_heads=config.num_attention_heads,
                    embed_dim=config.embed_dim,
                    key_size=config.key_size,
                    ffn_embed_dim=config.ffn_embed_dim,
                    name=f"attention_layer_{layer_idx}",
                )
                for layer_idx in range(config.num_layers)
            ]
        )

        self.lm_head = nn.Linear(config.embed_dim, config.n_expressions_bins)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        outs = {}
        x = self.expression_embedding_layer(input_ids)

        if self.config.use_gene_embedding:
            gene_indices = torch.arange(self.config.n_genes, device=x.device)
            gene_embedding = self.gene_embedding_layer(gene_indices)
            if self.config.project_gene_embedding:
                gene_embedding = self.fc_gene_embedding(gene_embedding)
            x = x + gene_embedding

        if attention_mask is None:
            batch_size, seq_length = input_ids.shape
            attention_mask = torch.ones(  # noqa
                (batch_size, 1, seq_length, seq_length),
                device=input_ids.device,
                dtype=bool,
            )

        for layer_idx, transformer in enumerate(self.transformer_layers):
            output = transformer(x, attention_mask=attention_mask)
            x = output["embeddings"]
            if (layer_idx + 1) in self.config.embeddings_layers_to_save:
                outs[f"embeddings_{(layer_idx + 1)}"] = output["embeddings"]
            if (layer_idx + 1) in self._attention_layers_to_save:
                for map_number in self._attention_maps_per_layer_to_save[layer_idx + 1]:
                    dkey = f"attention_map_layer_{layer_idx + 1}_number_{map_number}"
                    outs[dkey] = output["attention_weights"][:, map_number + 1]

        outs["logits"] = self.lm_head(x)
        return outs
