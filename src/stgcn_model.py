"""
模块B: ST-GCN 跌倒检测模型 (纯 PyTorch)
输入: (B, T, J, C) → T=16, J=17, C=3(x,y,conf)
输出: (B, 2) → [safe, risk]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGraphConv(nn.Module):
    """空间图卷积：在骨骼关节图上进行消息传递（3分区策略）"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim * 3, out_dim, 1)

    def forward(self, x, A):
        B, C, T, J = x.shape
        D = A.sum(-1).sqrt().clamp(min=1e-6)
        A_norm = A / (D.unsqueeze(-1) * D.unsqueeze(-2) + 1e-6)
        x = x.permute(0, 2, 3, 1).reshape(B * T, J, C)
        parts = [torch.bmm(A_norm[k].unsqueeze(0).expand(B * T, -1, -1), x) for k in range(3)]
        x = torch.cat(parts, dim=-1).reshape(B, T, J, -1).permute(0, 3, 1, 2)
        return self.conv(x)


class STGCNBlock(nn.Module):
    """ST-GCN 块：空间GCN + 时间TCN + 残差连接"""
    def __init__(self, in_dim, out_dim, temporal_kernel=9, dropout=0.2):
        super().__init__()
        self.gcn = SpatialGraphConv(in_dim, out_dim)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, (temporal_kernel, 1), padding=(temporal_kernel // 2, 0)),
            nn.BatchNorm2d(out_dim), nn.ReLU(), nn.Dropout(dropout))
        self.skip = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, A):
        return F.relu(self.tcn(self.gcn(x, A)) + self.skip(x))


class STGCN(nn.Module):
    """
    ST-GCN 跌倒检测器 (~100万参数)
    等价于 MMAction2 的 ST-GCN++，纯 PyTorch 实现
    """
    def __init__(self, num_joints=17, in_channels=3, num_classes=2,
                 hidden=128, num_blocks=6, dropout=0.2):
        super().__init__()
        self.A = nn.Parameter(torch.randn(3, num_joints, num_joints) * 0.02)
        # 确保对角线有值（自身连接）
        with torch.no_grad():
            for k in range(3):
                self.A[k].diagonal().fill_(1.0)
        self.input_proj = nn.Conv2d(in_channels, hidden, 1)
        self.blocks = nn.ModuleList([STGCNBlock(hidden, hidden, dropout=dropout) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.input_proj(x)
        for b in self.blocks:
            x = b(x, self.A)
        return self.head(self.pool(x).flatten(1))

    def predict_risk(self, x):
        return torch.softmax(self.forward(x), dim=-1)[:, 1]


if __name__ == "__main__":
    m = STGCN()
    print(f"参数量: {sum(p.numel() for p in m.parameters()):,}")
    x = torch.randn(2, 16, 17, 3)
    print(f"输入: {x.shape} → 输出: {m(x).shape} | Risk: {m.predict_risk(x).tolist()}")
