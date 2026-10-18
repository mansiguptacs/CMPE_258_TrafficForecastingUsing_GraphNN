"""STGCN (Yu, Yin & Zhu, IJCAI 2018) for multi-step speed forecasting.

Internal tensor layout is [batch, channels, time, nodes]. Each ST-Conv block is
gated temporal conv -> Chebyshev graph conv -> gated temporal conv -> layer norm,
so every block mixes information across time and across neighbouring sensors.
Unlike the paper's one-step-ahead output, the output layer predicts all
`output_len` steps at once.
"""
import torch
import torch.nn as nn


class TemporalGLU(nn.Module):
    """Gated temporal convolution over `kt` steps with a residual path: (P + x) * sigmoid(Q)."""

    def __init__(self, c_in: int, c_out: int, kt: int):
        super().__init__()
        self.kt = kt
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(kt, 1))
        self.align = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        p, q = self.conv(x).chunk(2, dim=1)
        residual = self.align(x[:, :, self.kt - 1:, :])
        return (p + residual) * torch.sigmoid(q)


class ChebGraphConv(nn.Module):
    """Spectral graph convolution with a K-order Chebyshev polynomial of the scaled Laplacian."""

    def __init__(self, c_in: int, c_out: int, ks: int, lap: torch.Tensor):
        super().__init__()
        self.ks = ks
        self.register_buffer("lap", lap)
        self.theta = nn.Conv2d(ks * c_in, c_out, kernel_size=1)

    def forward(self, x):
        terms = [x]
        if self.ks > 1:
            terms.append(torch.einsum("nm,bctm->bctn", self.lap, x))
        for _ in range(2, self.ks):
            terms.append(2 * torch.einsum("nm,bctm->bctn", self.lap, terms[-1]) - terms[-2])
        return self.theta(torch.cat(terms, dim=1))


class STConvBlock(nn.Module):
    def __init__(self, c_in: int, c_spatial: int, c_out: int, num_nodes: int,
                 kt: int, ks: int, lap: torch.Tensor, dropout: float):
        super().__init__()
        self.temporal1 = TemporalGLU(c_in, c_spatial, kt)
        self.spatial = ChebGraphConv(c_spatial, c_spatial, ks, lap)
        self.temporal2 = TemporalGLU(c_spatial, c_out, kt)
        self.norm = nn.LayerNorm([num_nodes, c_out])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.temporal1(x)
        x = torch.relu(self.spatial(x))
        x = self.temporal2(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x)


class STGCN(nn.Module):
    def __init__(self, num_nodes: int, lap: torch.Tensor, input_len: int = 12, output_len: int = 12,
                 blocks=((1, 32, 64), (64, 32, 128)), kt: int = 3, ks: int = 3, dropout: float = 0.3):
        super().__init__()
        self.blocks = nn.ModuleList(
            STConvBlock(c_in, c_sp, c_out, num_nodes, kt, ks, lap, dropout) for c_in, c_sp, c_out in blocks
        )
        t_left = input_len - 2 * len(blocks) * (kt - 1)
        if t_left < 1:
            raise ValueError("input_len too short for the number of blocks and kernel size")
        c_last = blocks[-1][-1]
        self.out_temporal = TemporalGLU(c_last, c_last, t_left)
        self.out_norm = nn.LayerNorm([num_nodes, c_last])
        self.out_fc = nn.Sequential(
            nn.Conv2d(c_last, c_last, kernel_size=1), nn.ReLU(), nn.Conv2d(c_last, output_len, kernel_size=1)
        )

    def forward(self, x):
        """x: [batch, input_len, nodes, 1] normalized speeds -> [batch, output_len, nodes] normalized."""
        x = x.permute(0, 3, 1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.out_temporal(x)
        x = self.out_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.out_fc(x).squeeze(2)
