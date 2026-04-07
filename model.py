"""
Dual-stream regression model for predicting continuous ATAC-seq signal.

Architecture:
    DNA (1000×5)  ──→ [Conv1D × N] → [GlobalAvgPool] → [FC] ──┐
                                                               ├→ [Fusion FC] → [ReLU] → Linear(1)
    RNA (expr_dim) ──→ [FC] → [BN] → [ReLU] ────────────────┘

Output: single unbounded value (no sigmoid).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── V3 Components ─────────────────────────────────────────────────────────

class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block for 1D convolutions."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualBlock1D(nn.Module):
    """Residual block with SE attention and optional downsampling."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 downsample=False, use_se=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock1D(out_channels) if use_se else nn.Identity()

        self.shortcut = nn.Identity()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return self.relu(out)


# ── V3 Model ──────────────────────────────────────────────────────────────

class DualStreamRegressor(nn.Module):
    """V3: Residual-SE architecture for ATAC signal regression."""

    def __init__(self, seq_input_dim=5, seq_len=1000,
                 num_filters=64, kernel_size=12, num_conv_layers=6,
                 expression_dim=1, hidden_dim=128,
                 dropout_rate=0.4, l2_reg=0.0001):
        super().__init__()
        self.l2_reg = l2_reg

        # ── Sequence branch (ResNet) ──────────────────────────────
        # Initial projection
        self.initial_conv = nn.Sequential(
            nn.Conv1d(seq_input_dim, num_filters, kernel_size, 
                      padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(inplace=True)
        )

        res_blocks = []
        in_ch = num_filters
        for i in range(num_conv_layers):
            # Downsample every 2 layers to manage spatial dimension
            downsample = (i % 2 == 0 and i > 0)
            stride = 2 if downsample else 1
            res_blocks.append(
                ResidualBlock1D(in_ch, num_filters, kernel_size=3,
                                stride=stride, downsample=downsample)
            )
            in_ch = num_filters

        self.res_stack = nn.Sequential(*res_blocks)
        self.seq_pool = nn.AdaptiveAvgPool1d(1)
        self.seq_fc = nn.Linear(num_filters, hidden_dim)

        # ── Expression branch (MLP - V5 Enhanced) ────────────────
        self.expr_branch = nn.Sequential(
            nn.Linear(expression_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # ── Fusion → regression output ────────────────────────────
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_bn = nn.BatchNorm1d(hidden_dim)
        self.fusion_dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_dim, 1)          # linear output

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, seq, expr):
        """
        Args:
            seq:  (batch, seq_len, input_dim)
            expr: (batch, expression_dim)
        Returns:
            (batch,)  — predicted ATAC signal (standardised)
        """
        # Sequence branch
        x = seq.transpose(1, 2)                       # → (B, C, L)
        x = self.initial_conv(x)
        x = self.res_stack(x)
        x = self.seq_pool(x).view(x.size(0), -1)      # → (B, C)
        seq_feat = F.relu(self.seq_fc(x))

        # Expression branch
        expr_feat = self.expr_branch(expr)

        # Fusion
        combined = torch.cat([seq_feat, expr_feat], dim=1)
        h = F.relu(self.fusion_fc(combined))
        h = self.fusion_bn(h)
        h = self.fusion_dropout(h)
        return self.out(h).squeeze(-1)

    def get_l2_loss(self):
        l2 = torch.tensor(0.0, device=next(self.parameters()).device)
        for p in self.parameters():
            if not isinstance(p, nn.Parameter): continue
            l2 = l2 + torch.norm(p)
        return self.l2_reg * l2
