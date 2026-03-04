"""
xlstm_model.py
==============
Triển khai mô hình xLSTM theo paper:
  "xLSTM: Extended Long Short-Term Memory" (Beck et al., 2024)
  arXiv:2405.04517v2

Kiến trúc:
  - sLSTM: Scalar memory + Exponential gating + Memory mixing
  - mLSTM: Matrix memory + Covariance update rule + Fully parallelizable
  - xLSTM Block: Residual blocks (pre/post up-projection)
  - xLSTM: Stacked xLSTM blocks → Classifier head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# 1.  sLSTM – Scalar memory, Exponential gates, Memory mixing
# ══════════════════════════════════════════════════════════════════════════════
class sLSTMCell(nn.Module):
    """
    sLSTM cell (Eq. 8-17 trong paper).
    Exponential input/forget gate + Normalizer state.
    Hỗ trợ multi-head memory mixing.
    """

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size phải chia hết cho num_heads"

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = hidden_size // num_heads

        # Input projections: z (cell input), i (input gate), f (forget gate), o (output gate)
        self.W_z = nn.Linear(input_size,  hidden_size)
        self.W_i = nn.Linear(input_size,  hidden_size)
        self.W_f = nn.Linear(input_size,  hidden_size)
        self.W_o = nn.Linear(input_size,  hidden_size)

        # Recurrent weights (block-diagonal = memory mixing within each head)
        self.R_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_o = nn.Linear(hidden_size, hidden_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "W_" in name and "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "R_" in name and "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        # Khởi tạo forget gate bias = [3,6] (theo paper)
        nn.init.uniform_(self.W_f.bias, 3.0, 6.0)

    def forward(
        self,
        x: torch.Tensor,                              # (B, input_size)
        state: Optional[Tuple[torch.Tensor, ...]]
    ) -> Tuple[torch.Tensor, Tuple]:
        B = x.size(0)

        if state is None:
            c = torch.zeros(B, self.hidden_size, device=x.device)
            n = torch.ones(B,  self.hidden_size, device=x.device)
            h = torch.zeros(B, self.hidden_size, device=x.device)
            m = torch.zeros(B, self.hidden_size, device=x.device)
        else:
            c, n, h, m = state

        # Gate pre-activations
        z_tilde = torch.tanh(self.W_z(x) + self.R_z(h))           # cell input
        i_tilde = self.W_i(x) + self.R_i(h)                        # input gate (exp)
        f_tilde = self.W_f(x) + self.R_f(h)                        # forget gate (exp)
        o_tilde = torch.sigmoid(self.W_o(x) + self.R_o(h))         # output gate

        # ── Stabilizer state (Eq.15) ─────────────────────────────────────────
        m_new = torch.max(f_tilde + m, i_tilde)                     # m_t

        # Stabilized gates
        i_prime = torch.exp(i_tilde - m_new)                        # i'_t
        f_prime = torch.exp(f_tilde + m - m_new)                    # f'_t

        # ── Cell & normalizer state update (Eq.8, 9) ─────────────────────────
        c_new = f_prime * c + i_prime * z_tilde
        n_new = f_prime * n + i_prime

        # ── Hidden state (Eq.10) ─────────────────────────────────────────────
        # h_tilde = c / max(|n|, 1)
        denom = torch.clamp(torch.abs(n_new), min=1.0)
        h_tilde = c_new / denom
        h_new   = o_tilde * h_tilde

        return h_new, (c_new, n_new, h_new, m_new)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  mLSTM – Matrix memory, Covariance update, Fully parallelizable
# ══════════════════════════════════════════════════════════════════════════════
class mLSTMCell(nn.Module):
    """
    mLSTM cell (Eq. 19-27 trong paper).
    Matrix memory C ∈ R^(head_dim × head_dim).
    Hỗ trợ multi-head và parallel training.
    """

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = hidden_size // num_heads

        d = self.head_dim

        # Query, Key, Value projections (mỗi head)
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)

        # Gate projections (không có recurrent weights – no memory mixing)
        self.w_i = nn.Linear(input_size, num_heads)   # input gate
        self.w_f = nn.Linear(input_size, num_heads)   # forget gate
        self.W_o = nn.Linear(input_size, hidden_size) # output gate

        # Optional causal convolution (kernel size 4, theo Figure 11)
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=4,
            padding=3,
            groups=hidden_size   # depthwise
        )
        # Learnable skip
        self.skip_alpha = nn.Parameter(torch.ones(hidden_size) * 0.1)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() == 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.uniform_(self.w_f.bias, 3.0, 6.0)

    def forward(
        self,
        x: torch.Tensor,                              # (B, T, input_size)
        state: Optional[Tuple[torch.Tensor, ...]]
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Chế độ recurrent: x shape (B, input_size),
        Chế độ parallel:  x shape (B, T, input_size)
        Ở đây dùng parallel để tốc độ huấn luyện.
        """
        if x.dim() == 2:
            # Single step: (B, F) → (B, 1, F)
            x = x.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False

        B, T, _ = x.shape
        H = self.num_heads
        D = self.head_dim

        # ── Convolution tiền xử lý (causal) ──────────────────────────────────
        x_conv = self.conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_in   = x + self.skip_alpha.unsqueeze(0).unsqueeze(0) * x_conv

        # ── Q, K, V projections ───────────────────────────────────────────────
        Q = self.W_q(x_in).view(B, T, H, D)    # (B, T, H, D)
        K = self.W_k(x_in).view(B, T, H, D) / math.sqrt(D)
        V = self.W_v(x_in).view(B, T, H, D)

        # ── Gates (exponential) ───────────────────────────────────────────────
        i_tilde = self.w_i(x_in)               # (B, T, H)
        f_tilde = self.w_f(x_in)               # (B, T, H)

        # Stabilization: D matrix (T×T per head)
        # log D = cumsum(f) + i  (lower triangular causal mask)
        f_log   = F.logsigmoid(f_tilde)         # (B, T, H)
        f_cumsum = torch.cumsum(f_log, dim=1)   # (B, T, H)

        # Broadcast: D_ij = f_cumsum_i - f_cumsum_j + i_j  (i >= j)
        # Shape: (B, T, T, H)
        f_cum_i = f_cumsum.unsqueeze(2)         # (B, T, 1, H)
        f_cum_j = f_cumsum.unsqueeze(1)         # (B, 1, T, H)
        i_j     = i_tilde.unsqueeze(1)          # (B, 1, T, H)

        D_log   = f_cum_i - f_cum_j + i_j       # (B, T, T, H)

        # Causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        D_log = D_log.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(-1), float('-inf'))

        # Stabilize
        m_row = D_log.max(dim=2, keepdim=True).values
        D_prime = torch.exp(D_log - m_row)      # (B, T, T, H)

        # ── Attention-like computation ────────────────────────────────────────
        # Q: (B,T,H,D)  K: (B,T,H,D)  →  QK: (B,T,T,H)  i=query index, j=key index
        QK = torch.einsum('bihd,bjhd->bijh', Q, K)    # (B, T, T, H)

        # Modulate by stabilized gate matrix D_prime: (B, T, T, H)
        C_tilde = QK * D_prime                          # (B, T, T, H)

        # Normalizer: sum over key dimension j → (B, T, H)
        n_raw  = C_tilde.sum(dim=2)
        n_norm = torch.clamp(torch.abs(n_raw), min=1.0)

        # Normalize: broadcast n_norm over j dimension
        C = C_tilde / n_norm.unsqueeze(2)              # (B, T, T, H)

        # Retrieve: H_tilde = C @ V  →  (B, T, H, D)
        H_tilde = torch.einsum('bijh,bjhd->bihd', C, V)
        H_tilde = H_tilde.contiguous().view(B, T, self.hidden_size)  # (B,T,hidden)

        # Output gate
        o = torch.sigmoid(self.W_o(x_in))             # (B,T,hidden_size)
        out = o * H_tilde                              # (B,T,hidden_size)

        if squeeze:
            out = out.squeeze(1)

        return out, None  # mLSTM không có explicit state (parallelized)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  xLSTM Blocks
# ══════════════════════════════════════════════════════════════════════════════
class sLSTMBlock(nn.Module):
    """
    sLSTM Block với Post Up-Projection (như Transformer, Figure 10).
    PreLayerNorm residual → sLSTM → GroupNorm → Gated MLP
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float, proj_factor: float = 4/3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        expanded         = int(hidden_size * proj_factor * 2)

        self.norm1  = nn.LayerNorm(hidden_size)
        self.slstm  = sLSTMCell(hidden_size, hidden_size, num_heads)
        self.gnorm  = nn.GroupNorm(num_heads, hidden_size)

        # Gated MLP (post up-projection)
        self.norm2  = nn.LayerNorm(hidden_size)
        self.proj_up   = nn.Linear(hidden_size, expanded)
        self.proj_gate = nn.Linear(hidden_size, expanded // 2)
        self.proj_down = nn.Linear(expanded // 2, hidden_size)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H)"""
        B, T, _ = x.shape

        # sLSTM over time steps
        residual = x
        x_n = self.norm1(x)

        state = None
        outputs = []
        for t in range(T):
            h_t, state = self.slstm(x_n[:, t, :], state)
            outputs.append(h_t)

        h = torch.stack(outputs, dim=1)        # (B, T, hidden)

        # GroupNorm (head-wise LayerNorm)
        h = self.gnorm(h.transpose(1, 2)).transpose(1, 2)
        h = h + residual

        # Gated MLP
        res2 = h
        h_n  = self.norm2(h)
        up   = self.proj_up(h_n)
        gate = torch.sigmoid(self.proj_gate(h_n))
        mid  = up[..., : up.shape[-1] // 2] * gate
        out  = self.proj_down(self.dropout(mid))
        return out + res2


class mLSTMBlock(nn.Module):
    """
    mLSTM Block với Pre Up-Projection (như SSM, Figure 11).
    PreLayerNorm residual → Up-proj → mLSTM → GroupNorm → Gate → Down-proj
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float, proj_factor: float = 2):
        super().__init__()
        self.hidden_size = hidden_size
        expanded         = int(hidden_size * proj_factor)

        self.norm   = nn.LayerNorm(hidden_size)

        # Pre up-projection
        self.up1    = nn.Linear(hidden_size, expanded)   # for mLSTM input
        self.up2    = nn.Linear(hidden_size, expanded)   # for output gate

        self.mlstm  = mLSTMCell(expanded, expanded, num_heads)
        self.gnorm  = nn.GroupNorm(num_heads, expanded)
        self.act    = nn.SiLU()

        # Down-projection
        self.down   = nn.Linear(expanded, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, hidden_size)"""
        residual = x
        x_n = self.norm(x)                             # (B,T,H)

        # Up-project
        x_up  = self.act(self.up1(x_n))               # (B,T,expanded)
        x_gate = torch.sigmoid(self.up2(x_n))          # external output gate

        # mLSTM (parallel)
        h, _ = self.mlstm(x_up, None)                 # (B,T,expanded)

        # GroupNorm
        h = self.gnorm(h.transpose(1, 2)).transpose(1, 2)  # (B,T,expanded)

        # Learnable skip + gate
        h = h * x_gate

        # Down-project + residual
        out = self.down(self.dropout(h))               # (B,T,hidden_size)
        return out + residual


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Full xLSTM Model
# ══════════════════════════════════════════════════════════════════════════════
class xLSTM(nn.Module):
    """
    Mô hình xLSTM hoàn chỉnh cho bài toán phân loại xu hướng.

    Kiến trúc:
      Input Projection
      → Stacked xLSTM blocks (mLSTM + sLSTM xen kẽ theo tỉ lệ mlstm:slstm)
      → Classifier head (GlobalPool → FC → LayerNorm → Dropout → FC → Softmax)

    Cách dùng:
      model = xLSTM.from_config(config)
    """

    def __init__(
        self,
        input_size:         int,
        hidden_size:        int   = 128,
        num_heads:          int   = 4,
        num_mlstm_layers:   int   = 3,
        num_slstm_layers:   int   = 1,
        dropout:            float = 0.2,
        fc_hidden:          int   = 64,
        num_classes:        int   = 3,
        proj_factor:        float = 4/3,
    ):
        super().__init__()
        self.input_size       = input_size
        self.hidden_size      = hidden_size
        num_layers            = num_mlstm_layers + num_slstm_layers

        # ── Input projection ─────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        # ── Build blocks ──────────────────────────────────────────────────────
        # Sắp xếp xen kẽ: phần lớn là mLSTM, ít sLSTM (như xLSTM[7:1])
        # Đặt sLSTM ở các vị trí cách đều nhau
        blocks = []
        slstm_positions = set()
        if num_slstm_layers > 0:
            step = num_layers // num_slstm_layers
            for i in range(num_slstm_layers):
                slstm_positions.add(i * step + step // 2)

        for i in range(num_layers):
            if i in slstm_positions:
                blocks.append(sLSTMBlock(hidden_size, num_heads, dropout, proj_factor))
            else:
                blocks.append(mLSTMBlock(hidden_size, num_heads, dropout))

        self.blocks = nn.ModuleList(blocks)

        # ── Classifier head ───────────────────────────────────────────────────
        self.final_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.LayerNorm(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

        self._log_architecture(num_mlstm_layers, num_slstm_layers, slstm_positions, num_layers)
        self._count_params()

    def _log_architecture(self, n_m, n_s, s_pos, total):
        print(f"[xLSTM] Kiến trúc: {total} blocks  "
              f"({n_m} mLSTM + {n_s} sLSTM)  "
              f"— sLSTM tại vị trí: {sorted(s_pos)}")

    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[xLSTM] Tổng tham số: {total:,}  |  Trainable: {trainable:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_size)
        returns: logits (B, num_classes)
        """
        # Input projection
        h = self.input_proj(x)                  # (B, T, hidden_size)

        # Stacked blocks
        for block in self.blocks:
            h = block(h)                         # (B, T, hidden_size)

        # Global pooling (lấy trung bình theo time)
        h = self.final_norm(h)
        h = h.mean(dim=1)                        # (B, hidden_size)

        # Classify
        logits = self.classifier(h)              # (B, num_classes)
        return logits

    @classmethod
    def from_config(cls, cfg: dict) -> "xLSTM":
        """Tạo model từ MODEL_CONFIG."""
        return cls(
            input_size        = cfg["input_size"],
            hidden_size       = cfg["hidden_size"],
            num_heads         = cfg["num_heads"],
            num_mlstm_layers  = cfg["num_mlstm_layers"],
            num_slstm_layers  = cfg["num_slstm_layers"],
            dropout           = cfg["dropout"],
            fc_hidden         = cfg["fc_hidden"],
            num_classes       = cfg["num_classes"],
            proj_factor       = cfg["proj_factor"],
        )
