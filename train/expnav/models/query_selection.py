import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalGuidedGlobalCond(nn.Module):
    """
    Variant A:
      - goal -> goal_vec
      - learned N queries + goal conditioning -> Q_goal
      - cross-attn(Q_goal -> image tokens) -> selected tokens
      - global_cond = MLP([mean(selected), goal_vec])

    Expected inputs:
      img_enc:  [B, L, C, H, W]   (e.g., B,L,4,28,28)
      goal_enc: [B, C, H, W]      (e.g., B,4,28,28)
    Output:
      global_cond: [B, out_dim]
    """
    def __init__(
        self,
        num_queries: int = 16,
        model_dim: int = 256,
        num_heads: int = 8,
        out_dim: int = 256,
        goal_mlp_hidden: int = 256,
        cond_mlp_hidden: int = 512,
        dropout: float = 0.0,
        use_ln: bool = True,
    ):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        self.num_queries = num_queries
        self.model_dim = model_dim
        self.out_dim = out_dim

        # # Project tokens from C -> D
        # self.img_proj = nn.Linear(in_channels, model_dim)
        # self.goal_proj = nn.Linear(in_channels, model_dim)

        # Learned query slots (N, D)
        self.queries = nn.Parameter(torch.randn(num_queries, model_dim) * 0.02)

        # Goal -> query bias (B, D), then broadcast to (B, N, D)
        self.goal_to_q = nn.Sequential(
            nn.Linear(model_dim, goal_mlp_hidden),
            nn.SiLU(),
            nn.Linear(goal_mlp_hidden, model_dim),
        )

        # Cross-attention: Q attends to image tokens (K,V)
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.use_ln = use_ln
        if use_ln:
            self.ln_q = nn.LayerNorm(model_dim)
            self.ln_kv = nn.LayerNorm(model_dim)
            self.ln_sel = nn.LayerNorm(model_dim)
            self.ln_goal = nn.LayerNorm(model_dim)

        # Final cond head: [sel_vec; goal_vec] -> out_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(2 * model_dim, cond_mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(cond_mlp_hidden, out_dim),
        )

    @staticmethod
    def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, ..., C, H, W] or [B, C, H, W]
        returns tokens: [B, M, C]
        """
        if x.dim() == 5:
            # [B, L, C, H, W] -> [B, M, C] where M=L*H*W
            B, L, C, H, W = x.shape
            x = x.permute(0, 1, 3, 4, 2).contiguous()  # [B, L, H, W, C]
            return x.view(B, L * H * W, C)
        elif x.dim() == 4:
            # [B, C, H, W] -> [B, M, C] where M=H*W
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
            return x.view(B, H * W, C)
        else:
            raise ValueError(f"Expected x dim 4 or 5, got {x.dim()}")

    def forward(self, img_enc: torch.Tensor, goal_enc: torch.Tensor) -> torch.Tensor:
        """
        img_enc:  [B, L, C, H, W]
        goal_enc: [B, C, H, W]
        """
        # --- tokens ---
        img_tokens = self._flatten_tokens(img_enc)    # [B, M, C]
        goal_tokens = self._flatten_tokens(goal_enc)  # [B, G, C]

        # # --- project to model_dim ---
        # img_tokens = self.img_proj(img_tokens)        # [B, M, D]
        # goal_tokens = self.goal_proj(goal_tokens)     # [B, G, D]

        if self.use_ln:
            img_tokens = self.ln_kv(img_tokens)

        # --- goal summary vec ---
        goal_vec = goal_tokens.mean(dim=1)            # [B, D]
        if self.use_ln:
            goal_vec = self.ln_goal(goal_vec)

        # --- build goal-conditioned queries ---
        B = img_tokens.shape[0]
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)   # [B, N, D]
        Q = Q + self.goal_to_q(goal_vec).unsqueeze(1)     # [B, N, D]
        if self.use_ln:
            Q = self.ln_q(Q)

        # --- cross-attn: queries attend into image tokens ---
        # selected: [B, N, D]
        selected, _attn = self.attn(query=Q, key=img_tokens, value=img_tokens, need_weights=False)

        if self.use_ln:
            selected = self.ln_sel(selected)

        # --- make global conditioning vector ---
        sel_vec = selected.mean(dim=1)                # [B, D]
        global_cond = torch.cat([sel_vec, goal_vec], dim=-1)  # [B, 2D]
        global_cond = self.cond_mlp(global_cond)      # [B, out_dim]
        return global_cond


class GoalGuidedHardQuerySelector(nn.Module):
    """
    GroundingDINO-style language-guided (goal-guided) query selection.

    Input:
        img_enc  : [B, L, C, H, W]
        goal_enc : [B, C, H, W]

    Output:
        global_cond : [B, out_dim]
    """

    def __init__(
        self,
        model_dim: int = 256,
        num_query: int = 16,
        out_dim: int = 256,
    ):
        super().__init__()

        self.num_query = num_query

        # final global conditioning head
        self.cond_mlp = nn.Sequential(
            nn.Linear(model_dim * 2, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    @staticmethod
    def _flatten_img_tokens(x):
        # [B, L, C, H, W] -> [B, L*H*W, C]
        B, L, C, H, W = x.shape
        return x.permute(0, 1, 3, 4, 2).reshape(B, L * H * W, C)

    @staticmethod
    def _flatten_goal_tokens(x):
        # [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, H * W, C)

    def forward(self, img_enc, goal_enc):
        """
        img_enc:  [B, L, C, H, W]
        goal_enc: [B, C, H, W]
        """

        # --- flatten to tokens ---
        img_tokens = self._flatten_img_tokens(img_enc)     # [B, M, C]
        goal_tokens = self._flatten_goal_tokens(goal_enc) # [B, G, C]

        # --- GroundingDINO similarity ---
        # logits: [B, M, G]
        logits = torch.einsum("bmd,bgd->bmg", img_tokens, goal_tokens)

        # max over goal tokens → image relevance
        # [B, M]
        logits_per_img = logits.max(dim=-1).values

        # top-k image tokens
        # [B, K]
        topk_idx = torch.topk(
            logits_per_img,
            k=self.num_query,
            dim=1
        ).indices

        # gather selected image tokens
        # [B, K, D]
        idx = topk_idx.unsqueeze(-1).expand(-1, -1, img_tokens.size(-1))
        selected_tokens = torch.gather(img_tokens, dim=1, index=idx)

        # --- build global conditioning vector ---
        selected_vec = selected_tokens.mean(dim=1)  # [B, D]
        goal_vec = goal_tokens.mean(dim=1)           # [B, D]

        global_cond = torch.cat([selected_vec, goal_vec], dim=-1)
        global_cond = self.cond_mlp(global_cond)     # [B, out_dim]

        return global_cond