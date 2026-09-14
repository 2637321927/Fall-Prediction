"""
TCN 跌倒检测模型 (时间卷积网络)
输入: (B, T, J, C) → T=16, J=17, C=3
输出: (B, 2) → [safe, risk]

比 ST-GCN 简单：把骨架展平后沿时间轴卷
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """因果卷积：只看过去，不看未来（适合实时预警）"""
    def __init__(self, in_ch, out_ch, kernel, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    """TCN 基础块：膨胀因果卷积 ×2 + 残差"""
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        self.c1 = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.c2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        r = self.skip(x)
        x = self.drop(F.relu(self.bn1(self.c1(x))))
        x = self.drop(F.relu(self.bn2(self.c2(x))))
        return F.relu(x[:, :, :r.shape[-1]] + r)


class TCN(nn.Module):
    """
    TCN 跌倒检测器 (~38万参数)
    膨胀系数逐层翻倍: 1, 2, 4, 8 → 感受野覆盖 31 帧
    """
    def __init__(self, num_joints=17, in_channels=3, num_classes=2,
                 hidden=128, num_layers=4, dropout=0.2):
        super().__init__()
        input_dim = num_joints * in_channels  # 51

        layers = []
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden
            layers.append(TCNBlock(in_ch, hidden, dilation=2 ** i, dropout=dropout))
        self.tcn = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes))

    def forward(self, x):
        B, T = x.shape[:2]
        x = x.reshape(B, T, -1).permute(0, 2, 1)  # (B, 51, T)
        return self.head(self.tcn(x))

    def predict_risk(self, x):
        return torch.softmax(self.forward(x), dim=-1)[:, 1]


if __name__ == "__main__":
    m = TCN()
    print(f"参数量: {sum(p.numel() for p in m.parameters()):,}")
    x = torch.randn(2, 16, 17, 3)
    print(f"输入: {x.shape} → {m(x).shape} | Risk: {m.predict_risk(x).tolist()}")
