import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels, dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv3D(in_channels, out_channels, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_z = x2.size(2) - x1.size(2)
        diff_y = x2.size(3) - x1.size(3)
        diff_x = x2.size(4) - x1.size(4)

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2,
                        diff_z // 2, diff_z - diff_z // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self,
                 n_channels: int = 1,
                 n_classes: int = 1,
                 features: List[int] = [32, 64, 128, 256],
                 dropout: float = 0.1,
                 bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv3D(n_channels, features[0], dropout)
        self.down1 = Down3D(features[0], features[1], dropout)
        self.down2 = Down3D(features[1], features[2], dropout)
        self.down3 = Down3D(features[2], features[3], dropout)

        factor = 2 if bilinear else 1
        self.down4 = Down3D(features[3], features[3] * 2 // factor, dropout)

        # ⚠️ Fix in_channels for concatenation
        self.up1 = Up3D(features[3] * 2 // factor + features[3], features[3], dropout, bilinear)
        self.up2 = Up3D(features[3] + features[2], features[2], dropout, bilinear)
        self.up3 = Up3D(features[2] + features[1], features[1], dropout, bilinear)
        self.up4 = Up3D(features[1] + features[0], features[0], dropout, bilinear)

        self.outc = OutConv3D(features[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(n_channels=1, n_classes=1).to(device)
    x = torch.randn(1, 1, 64, 64, 32).to(device)

    with torch.no_grad():
        output = model(x)

    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Total number of parameters: {count_parameters(model):,}")
