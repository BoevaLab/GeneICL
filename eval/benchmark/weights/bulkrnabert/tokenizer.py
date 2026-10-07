import json
import os
from typing import List, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizer


class BinnedOmicTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        n_expressions_bins: int = 64,
        min_omic_value: float = 0.0,
        max_omic_value: float = 1.0,
        use_max_normalization: bool = True,
        normalization_factor: float = 1.0,
        prepend_cls_token: bool = False,
        fixed_sequence_length: Optional[int] = None,
        unpadded_length: Optional[int] = None,
        **kwargs,
    ):
        bin_tokens = [str(i) for i in range(n_expressions_bins)]
        special_tokens = ["<pad>", "<mask>", "<cls>"]

        vocab = {tok: i for i, tok in enumerate(bin_tokens)}
        offset = len(vocab)
        for i, tok in enumerate(special_tokens):
            vocab[tok] = offset + i

        ids_to_tokens = {i: tok for tok, i in vocab.items()}

        self.vocab = vocab
        self.ids_to_tokens = ids_to_tokens

        self.n_expressions_bins = n_expressions_bins
        self.min_omic_value = min_omic_value
        self.max_omic_value = max_omic_value
        self.use_max_normalization = use_max_normalization
        self.normalization_factor = normalization_factor
        self.prepend_cls_token = prepend_cls_token
        self.fixed_sequence_length = fixed_sequence_length
        self.unpadded_length = unpadded_length

        self.bin_edges = np.linspace(min_omic_value, max_omic_value, n_expressions_bins)

        self.pad_token = "<pad>"
        self.mask_token = "<mask>"
        self.cls_token = "<cls>"

        super().__init__(**kwargs)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def get_vocab(self) -> dict:
        return self.vocab

    def _tokenize(self, text, **kwargs):
        raise NotImplementedError("Use `encode` or `batch_encode_plus` methods.")

    def decode(self, token_ids, **kwargs):
        return [self._convert_id_to_token(i) for i in token_ids]

    def encode(
        self,
        gene_expr: Union[np.ndarray, List[float]],
        pad_to_fixed_length: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Union[List[int], torch.Tensor]:
        gene_expr = np.array(gene_expr)

        if self.use_max_normalization:
            gene_expr = gene_expr / self.normalization_factor

        token_ids = np.digitize(gene_expr, self.bin_edges).astype(int)
        token_ids = np.clip(token_ids, 0, self.n_expressions_bins - 1)
        token_ids[gene_expr == 0.0] = 0

        if self.prepend_cls_token:
            token_ids = np.concatenate([[self.cls_token_id], token_ids])

        if pad_to_fixed_length:
            current_max_length = self.fixed_sequence_length or max_length
            if current_max_length is None:
                raise ValueError("fixed_sequence_length or max_length must be set.")
            pad_len = current_max_length - len(token_ids)
            if pad_len > 0:
                token_ids = np.concatenate([token_ids, [self.pad_token_id] * pad_len])
            else:
                token_ids = token_ids[:current_max_length]

        if return_tensors == "pt":
            return torch.tensor(token_ids).unsqueeze(0)
        return token_ids.tolist()  # type: ignore

    def batch_encode_plus(
        self,
        batch_gene_expr: Union[np.ndarray, List[np.ndarray]],
        pad_to_fixed_length: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ):
        if isinstance(batch_gene_expr, list):
            batch_gene_expr = np.array(batch_gene_expr)

        encoded = [
            self.encode(
                gene_expr,
                pad_to_fixed_length=pad_to_fixed_length,
                max_length=max_length,
                return_tensors=None,
                **kwargs,
            )
            for gene_expr in batch_gene_expr
        ]

        encoded = np.array(encoded, dtype=np.int64)

        if return_tensors == "pt":
            return {"input_ids": torch.tensor(encoded)}
        return {"input_ids": encoded}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ):
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )
        with open(vocab_file, "w") as f:
            json.dump(self.vocab, f)
        return (vocab_file,)
