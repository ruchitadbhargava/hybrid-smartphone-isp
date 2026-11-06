"""Compact U-Net used for the learned ISP components.

Input : packed RGGB Bayer planes (4) + per-plane noise std map (4), H/2 x W/2
Output: 3-channel image at full resolution H x W (pixel shuffle)

The noise map makes the network adapt to the noise level of each frame
(as in FFDNet / Brooks et al.). The global branch pools the bottleneck over
the whole frame so the network can use scene-level cues (illuminant,
brightness) that a purely local receptive field cannot see.
"""
import torch
import torch.nn as nn


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, width=24, cin=8):
        super().__init__()
        c = width
        self.e1, self.e2 = block(cin, c), block(c, 2 * c)
        self.e3, self.e4 = block(2 * c, 4 * c), block(4 * c, 8 * c)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(8 * c, 8 * c, 1), nn.ReLU(inplace=True))
        self.d3, self.d2, self.d1 = block(12 * c, 4 * c), block(6 * c, 2 * c), block(3 * c, c)
        self.head = nn.Conv2d(c, 12, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        e4 = e4 + self.glob(e4)
        d3 = self.d3(torch.cat([self.up(e4), e3], 1))
        d2 = self.d2(torch.cat([self.up(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up(d2), e1], 1))
        return self.shuffle(self.head(d1))


if __name__ == '__main__':
    for w in (8, 12, 16, 24, 32):
        net = UNet(w)
        out = net(torch.zeros(1, 8, 64, 64))
        assert out.shape == (1, 3, 128, 128)
        print(f'width {w}: {sum(p.numel() for p in net.parameters()) / 1e6:.3f} M params')
