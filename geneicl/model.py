import inspect, math, torch, torch.nn as nn

MAX_K = 10        # classification head width; tasks with K<MAX_K classes leave remaining cols unused
PCA_WHITEN = 0.5  # whitening exponent on PC scores (0=rotation only, 1=full whitening)


def rank_gauss(y: torch.Tensor) -> torch.Tensor:
    """(B, n) continuous labels -> per-row Gaussian rank scores. Monotone; removes marginal skew/outliers."""
    ranks = y.float().argsort(-1).argsort(-1).float()
    uniform = (ranks + 0.5) / y.shape[-1]
    return (torch.erfinv(2 * uniform - 1) * math.sqrt(2)).to(y.dtype)


def get_mlp(n_in, n_hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.GELU(), nn.Linear(n_hidden, n_out))


class TableAttnBase(nn.Module):
    # Tables are (B, rows, cols, D). row_attn folds B*rows into the attention batch, so tokens attend within
    # each row (across features). col_attn transposes first, so tokens attend within each column (across samples).
    def row_attn(self, q, kv=None, **kwargs):
        n_batch, n_rows, n_cols, embed_dim = q.shape
        q, kv = (None if t is None else t.flatten(0, 1) for t in [q, kv])
        return self(q, kv, **kwargs).reshape(n_batch, n_rows, -1, embed_dim)

    def col_attn(self, q, kv=None, **kwargs):
        return self.row_attn(q.transpose(1, 2), None if kv is None else kv.transpose(1, 2), **kwargs).transpose(1, 2)


class TransformerBlock(nn.MultiheadAttention, TableAttnBase):
    # Subclasses nn.MultiheadAttention only to reuse its parameter layout (in_proj_weight, out_proj), which the
    # checkpoints' state-dict keys depend on. Attention itself is SDPA. _in_projection_packed is a private torch API.
    def __init__(self, embed_dim, num_heads, ssmax=False, use_mlp=True):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)
        self.ssmax_layer = QASSMax(num_heads=num_heads, head_dim=embed_dim // num_heads) if ssmax else None
        self.mlp = get_mlp(embed_dim, embed_dim * 2, embed_dim) if use_mlp else None
        self.ln_attn = nn.RMSNorm(embed_dim)
        self.ln_mlp = nn.RMSNorm(embed_dim) if use_mlp else None

    def forward(self, q, kv=None, q_max_idx=None, kv_max_idx=None):
        x, q = q, self.ln_attn(q)
        kv = q if kv is None else self.ln_attn(kv)
        if kv_max_idx is not None: kv = kv[..., :kv_max_idx, :]
        if q_max_idx is not None: x, q = x[..., :q_max_idx, :], q[..., :q_max_idx, :]
        x = x + self._attn(q, kv)
        del q, kv
        if self.mlp is None:
            return x
        return x + self.mlp(self.ln_mlp(x))

    def _attn(self, q, k):
        q, k, v = nn.functional._in_projection_packed(q, k, k, self.in_proj_weight, self.in_proj_bias)
        q, k, v = (t.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2) for t in [q, k, v])
        q = q if self.ssmax_layer is None else self.ssmax_layer(q=q, n=k.size(-2))
        out = nn.functional.scaled_dot_product_attention(q, k, v)
        del q, k, v
        return self.out_proj(out.transpose(-3, -2).flatten(-2, -1))

class PostNormICLBlock(TransformerBlock):
    """Post-norm variant of TransformerBlock (which is pre-norm), used for the recurrent ICL stage. Same
    attention / QASSMax / MLP / RMSNorm modules, norms applied after each residual:

        x = RMSNorm(x + Attn(x, KV))
        x = RMSNorm(x + MLP(x))

    KV is restricted to the first kv_max_idx positions (thinking + support) exactly as the pre-norm
    block, so query positions are queries-only and never enter K/V."""
    def forward(self, q, kv=None, q_max_idx=None, kv_max_idx=None):
        kv_full = q if kv is None else kv
        if kv_max_idx is not None:
            kv_full = kv_full[..., :kv_max_idx, :]
        if q_max_idx is not None:
            q = q[..., :q_max_idx, :]
        x = self.ln_attn(q + self._attn(q, kv_full))     # post-norm attention residual (raw q, raw KV)
        x = self.ln_mlp(x + self.mlp(x))                 # post-norm MLP residual
        return x


class QASSMax(nn.Module):  # query-aware scalable softmax (TabICLv2)
    # Scales queries by a learned function of log(#keys), so attention does not flatten out as the support grows.
    # query_mlp starts at zero, i.e. training starts from a purely length-dependent temperature.
    def __init__(self, num_heads, head_dim, n_hidden=64):
        super().__init__()
        self.base_mlp = get_mlp(1, n_hidden, num_heads * head_dim)
        self.query_mlp = get_mlp(head_dim, n_hidden, head_dim)
        nn.init.zeros_(self.query_mlp[-1].weight)
        nn.init.zeros_(self.query_mlp[-1].bias)

    def forward(self, q, n):
        logn = q.new_tensor(math.log(max(1, n))).view(1, 1)
        return self.base_mlp(logn).view(1, self.num_heads, 1, self.head_dim) * (1 + torch.tanh(self.query_mlp(q))) * q

    @property
    def num_heads(self): return self.base_mlp[-1].out_features // self.head_dim
    @property
    def head_dim(self): return self.query_mlp[0].in_features


class InducedTransformerBlock(TableAttnBase): # Bottleneck attention in the column axis over a few rows instead of full attention
    def __init__(self, embed_dim, num_heads, n_inducing, ssmax=False):
        super().__init__()
        self.tfm1 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ssmax=ssmax, use_mlp=False)
        self.tfm2 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, use_mlp=False)
        self.inducing_vectors = nn.Parameter(0.02 * torch.randn(1, n_inducing, embed_dim))

    def forward(self, q, kv=None, q_max_idx=None, kv_max_idx=None):
        kv = self.tfm1(self.inducing_vectors.expand(q.shape[0], -1, -1), q if kv is None else kv, kv_max_idx=kv_max_idx)
        return self.tfm2(q, kv, q_max_idx=q_max_idx)


class BottleneckRowBlock(nn.Module): # Same bottleneck attention but also over the row axis
    """CLS-bottleneck row attention: CLS<-[CLS+genes], then genes<-CLS.
    For full self-attention over all tokens (full_attn ablation) use a plain TransformerBlock instead."""
    def __init__(self, embed_dim, num_heads, n_cls, cls_ffn=True):
        super().__init__()
        self.n_cls = n_cls
        # cls_ffn=False drops the FFN after the CLS<-[CLS+genes] attention (the gene->CLS step)
        self.cls_attn = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ssmax=False, use_mlp=cls_ffn)
        self.gene_attn = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ssmax=False)

    def row_attn(self, emb):
        cls = self.cls_attn.row_attn(emb[:, :, :self.n_cls], emb)
        return torch.cat([cls, self.gene_attn.row_attn(emb[:, :, self.n_cls:], cls)], dim=2)


class GeneICL(nn.Module):
    """ICL transformer for fixed-size gene expression: regression (scalar head) and classification (MAX_K logits).

    is_cls=False -> regression preds; is_cls=True -> class logits; is_cls=(B,) tensor (training) -> per-item labels.
    enc_mode: "bottleneck" (GeneICL) or "full" (full row attention, ablation).
    pca_var: the model reads per-dataset PCs covering this fraction of the support set's variance.
    """
    def __init__(self, n_genes=20021, embed_dim=128, n_blocks=3, nhead=8, n_think_rows=8,
                 n_cls_cols=4, n_cls_rows=4, loop_iters=8, enc_mode="bottleneck",
                 max_k=MAX_K, pca_var=0.9, enc_cls_ffn=True):
        super().__init__()
        assert enc_mode in ("bottleneck", "full"), enc_mode
        assert n_think_rows > 0, "the recurrent ICL stage needs n_think_rows > 0 (the thinking workspace)"
        assert 0 < pca_var <= 1, pca_var
        self.n_cls_cols = n_cls_cols
        self.pca_var = pca_var
        self.max_k = max_k
        icl_dim = embed_dim * n_cls_cols

        self.x_embed = nn.Linear(2, embed_dim)
        # Dual label pathways: classification uses an Embedding (one freely-placed vector per class);
        # regression uses a Linear on the rank-gaussified scalar.
        self.y_embed_in_cls  = nn.Embedding(max_k, embed_dim)
        self.y_embed_icl_cls = nn.Embedding(max_k, icl_dim)
        self.y_embed_in_num  = nn.Linear(1, embed_dim)
        self.y_embed_icl_num = nn.Linear(1, icl_dim)

        def _row_block():
            if enc_mode == "full":
                return TransformerBlock(embed_dim, nhead)
            return BottleneckRowBlock(embed_dim, nhead, n_cls=n_cls_cols, cls_ffn=enc_cls_ffn)

        self.col_blocks = nn.ModuleList([
            InducedTransformerBlock(embed_dim, nhead, n_inducing=n_cls_rows, ssmax=True) for _ in range(n_blocks)])
        self.row_blocks = nn.ModuleList([_row_block() for _ in range(n_blocks)])

        # loop_layers and trm_continue_head (below) are never run. They are part of the released checkpoints, and
        # constructing them draws from the RNG, so removing them would change the seeded initialization.
        self.loop_layers = nn.ModuleList([TransformerBlock(icl_dim, nhead, ssmax=True)])
        self.loop_iters = loop_iters     # recurrence depth of the ICL stage

        self.cls_tokens = nn.Parameter(0.02 * torch.randn(1, 1, n_cls_cols, embed_dim))
        self.thinking_rows = nn.Parameter(0.02 * torch.randn(1, n_think_rows, icl_dim))
        self.row_ln = nn.RMSNorm(embed_dim)
        self.out_ln = nn.RMSNorm(icl_dim)
        self.out_mlp     = get_mlp(icl_dim, icl_dim * 2, 1)             # regression: scalar
        self.out_mlp_cls = get_mlp(icl_dim, icl_dim * 2, max_k)        # classification: MAX_K logits

        # Recurrent ICL stage: one post-norm block applied at every recurrence step, with the encoded problem kept
        # as fixed evidence E and a separate hidden state H.
        self._n_think = n_think_rows
        self.eqr_block = PostNormICLBlock(icl_dim, nhead, ssmax=True, use_mlp=True)
        self.trm_continue_head = nn.Linear(icl_dim, 1)   # unused, see loop_layers above
        nn.init.zeros_(self.trm_continue_head.weight); nn.init.constant_(self.trm_continue_head.bias, 5.0)

    # ── label embedding routing ───────────────────────────────────────────────────────────────────
    def _y_emb(self, y, is_cls, cls_layer, num_layer):
        """Route (B, T) labels to the correct embedding layer based on is_cls (True/False/(B,) tensor)."""
        if is_cls is True:
            return cls_layer(y.long())
        numeric = num_layer(rank_gauss(y)[..., None])
        if is_cls is False:
            return numeric
        # mixed batch: evaluate both, torch.where picks per element
        # regression items are also looked up as classes (then discarded by torch.where), the clamp keeps their
        # cast values inside the embedding table
        classes = cls_layer(y.long().clamp(0, cls_layer.num_embeddings - 1))
        return torch.where(is_cls[:, None, None], classes, numeric)

    def _y_in(self, y, is_cls):
        return self._y_emb(y, is_cls, self.y_embed_in_cls, self.y_embed_in_num)

    def _y_icl(self, y, is_cls):
        return self._y_emb(y, is_cls, self.y_embed_icl_cls, self.y_embed_icl_num)

    def _head(self, emb, n_skip, is_cls):
        feat = self.out_ln(emb[:, n_skip:])
        return self.out_mlp_cls(feat) if is_cls else self.out_mlp(feat)

    def reg_point(self, out):
        """Scalar point prediction from the regression head output (..., 1)."""
        return out[..., 0]

    def reg_elem_loss(self, pred, target):
        """Per-item squared error, shape of target."""
        return (pred[..., 0] - target) ** 2

    # ── recurrent ICL stage (segmented / TRM) ─────────────────────────────────────────────────
    #   encode_evidence -> E (fixed) ;  trm_init_state -> H0 ;  trm_recur -> N post-norm steps ;
    #   trm_predict -> query logits/preds. In segmented training E is re-encoded after every optimizer step while
    #   H is carried over (detached).
    def _evidence(self, rows_emb, n_think):
        """Fixed evidence E = [0_think, rows_emb]. rows_emb (B, n_rows, icl_dim) already carries the ICL
        y embedding on the support rows; thinking positions hold zero evidence."""
        B, _, D = rows_emb.shape
        return torch.cat([rows_emb.new_zeros(B, n_think, D), rows_emb], dim=1)   # (B, n_think+n_rows, D)

    def encode_evidence(self, x, y, is_cls, pca_fit_n=None):
        """Run the GeneICL encoder to the ICL evidence rows, build E=[0_think, rows_emb]. Returns (E, meta)."""
        rows_emb = self.forward(x, y, is_cls=is_cls, return_enc=True, pca_fit_n=pca_fit_n)  # (B, n_rows, icl_dim)
        n_train, n_think = y.shape[1], self._n_think
        E = self._evidence(rows_emb, n_think)                        # [0_think, rows_emb]
        return E, dict(n_train=n_train, kv_max=n_think + n_train, n_skip=n_think + n_train, n_think=n_think)

    def trm_init_state(self, E):
        """H0 = [ learned thinking_rows , 0 support , 0 query ]."""
        H = torch.zeros_like(E)
        H[:, :self._n_think] = self.thinking_rows.to(E.dtype).expand(E.shape[0], -1, -1)
        return H

    def trm_recur(self, H, E, kv_max, n_steps):
        """n_steps of post-norm recurrence with repeated evidence injection: H = eqr_block(H + E). Query rows
        never enter K/V (kv_max = thinking + support)."""
        for _ in range(int(n_steps)):
            H = self.eqr_block(H + E, kv_max_idx=kv_max)
        return H

    def trm_predict(self, H, meta, is_cls):
        """Query-row logits (is_cls=True) / scalar preds (is_cls=False) from state H (query never in K/V)."""
        return self._head(H, meta["n_skip"], is_cls)

    def _recurrent_icl(self, rows_emb, n_train, is_cls):
        """forward()'s ICL stage: loop_iters recurrence steps from H0, decode the query rows of the final H."""
        E = self._evidence(rows_emb, self._n_think)
        kv_max = self._n_think + n_train                                         # K/V = thinking + support
        return self._head(self.trm_recur(self.trm_init_state(E), E, kv_max, self.loop_iters), kv_max, is_cls)

    def forward(self, x: torch.Tensor, y: torch.Tensor, is_cls=False,
                return_enc: bool = False, pca_fit_n: int = None) -> torch.Tensor:
        """is_cls=False -> regression preds (B, n_query, 1); True -> class logits (B, n_query, MAX_K).
        return_enc=True returns the encoded ICL evidence rows (used by encode_evidence; is_cls may be a (B,) tensor)."""
        n_batch, n_rows, n_cols = x.shape
        n_train = y.shape[1]
        n_fit = n_train if pca_fit_n is None else int(pca_fit_n)   # PCA/standardization FIT rows (default = support)

        # --- embed: mean-impute NaNs with train stats, standardize, then per-dataset PCA
        mask = ~torch.isfinite(x)
        x = x.masked_fill(mask, float('nan'))
        mean = torch.nan_to_num(torch.nanmean(x[:, :n_fit], dim=1, keepdim=True))
        x = torch.where(mask, mean.expand_as(x), x)
        std = x[:, :n_fit].std(dim=1, unbiased=False, keepdim=True) + 1e-8
        x = ((x - mean) / std).clamp(-10.0, 10.0)

        # Per-dataset PCA on already-standardized x. Gram trick: decompose (T×T) not (G×G).
        # Fit on n_fit rows only; project all rows. fp32 required: eigh rejects bf16.
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            tr_mean = x32[:, :n_fit].mean(1, keepdim=True)
            centered = x32[:, :n_fit] - tr_mean
            gram = centered @ centered.transpose(-1, -2)          # (B, n_fit, n_fit)
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            eigenvalues = eigenvalues.flip(-1).clamp_min(0); eigenvectors = eigenvectors.flip(-1)
            cumvar = eigenvalues.cumsum(-1) / eigenvalues.sum(-1, keepdim=True).clamp_min(1e-12)
            # one PC count for the whole batch (the max over tasks): tensors stay rectangular, but the task with
            # the flattest spectrum sets the width for all
            n_pcs = int(((cumvar < self.pca_var).sum(-1) + 1).max().clamp(1, n_fit).item())
            sing_vals = eigenvalues[..., :n_pcs].clamp_min(1e-12).sqrt()
            loadings = centered.transpose(-1, -2) @ eigenvectors[..., :n_pcs] / sing_vals[:, None, :]  # (B, G, k)
            score_std = (sing_vals / math.sqrt(n_fit)).clamp_min(1e-8)
            # geometric interpolation between no whitening (every PC keeps PC1's scale) and full whitening
            # (unit variance): with PCA_WHITEN=0.5 each PC is divided by sqrt(own_std * pc1_std)
            scale = score_std ** PCA_WHITEN * score_std[:, :1] ** (1.0 - PCA_WHITEN)
            pc_scores = ((x32 - tr_mean) @ (loadings / scale[:, None, :])).clamp(-10.0, 10.0)
        pc_scores = pc_scores.to(x.dtype)
        pca_inp = torch.stack([pc_scores, torch.zeros_like(pc_scores)], dim=-1)  # mask=0: PCs are dense
        emb = self.x_embed(pca_inp)

        # support labels enter twice: here into every feature token (so the encoder can relate features to y),
        # and again below into the row token that the recurrent ICL stage reads
        emb[:, :n_train] += self._y_in(y, is_cls)[:, :, None, :]  # broadcast over gene/PC axis

        # --- alternating col / row encoder blocks (CLS-bottleneck attention)
        cls = self.cls_tokens.expand(n_batch, n_rows, -1, -1)
        emb = torch.cat([cls, emb], dim=2)                          # emb^0 (B, R, C, D)
        for col_block, row_block in zip(self.col_blocks, self.row_blocks):
            # only support rows are written into the inducing summary, so query rows can read it but never
            # influence each other or the support (no leakage across query rows)
            genes = col_block.col_attn(emb[:, :, self.n_cls_cols:], kv_max_idx=n_train)
            emb = torch.cat([emb[:, :, :self.n_cls_cols], genes], dim=2)
            emb = row_block.row_attn(emb)

        # --- fold CLS tokens into one ICL token per row
        emb = self.row_ln(emb[:, :, :self.n_cls_cols]).flatten(-2, -1)  # (B, R, icl_dim)

        # --- ICL stage: attach the ICL label embedding to the support rows
        emb[:, :n_train] += self._y_icl(y, is_cls)                  # (B, n_train, icl_dim)
        if return_enc:                                             # ICL evidence rows (used by encode_evidence)
            return emb
        return self._recurrent_icl(emb, n_train, is_cls)


def load_ckpt(path, device="cpu"):
    """Rebuild the model from a checkpoint's saved `arch` (keys the constructor does not take, e.g. training
    metadata, are ignored) and load its weights strictly. Resolves the class via the module global, so a
    subclass patched in as geneicl.model.GeneICL is used instead."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cls = globals()["GeneICL"]
    accepted = inspect.signature(cls.__init__).parameters
    model = cls(**{k: v for k, v in ckpt["arch"].items() if k in accepted})
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()
