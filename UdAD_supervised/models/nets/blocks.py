import torch
import torch.nn as nn


class Conv3D(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, dropout_rate=0.3):
        super(Conv3D, self).__init__()
        stride = 2 if downsample else 1
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        output = self.block(x)
        return output


# class DeConv3D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DeConv3D, self).__init__()
#         self.block = nn.Sequential(
#             nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         output = self.block(x)
#         return output


class Encoder(nn.Module):
    def __init__(self, in_channels, cnum=32):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            Conv3D(in_channels, cnum, downsample=False),
            Conv3D(cnum, cnum, downsample=True)
        )
        self.layer2 = nn.Sequential(
            Conv3D(cnum, cnum * 2, downsample=False),
            Conv3D(cnum * 2, cnum * 2, downsample=True)
        )
        self.layer3 = nn.Sequential(
            Conv3D(cnum * 2, cnum * 4, downsample=False),
            Conv3D(cnum * 4, cnum * 4, downsample=False)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# class Decoder(nn.Module):
#     def __init__(self, cnum, out_channels):
#         super(Decoder, self).__init__()
#         self.decoder0 = nn.Sequential(
#             DeConv3D(cnum, cnum // 2),
#             Conv3D(cnum // 2, cnum // 2, downsample=False),
#         )
#         self.decoder1 = nn.Sequential(
#             DeConv3D(cnum, cnum // 2),
#             Conv3D(cnum // 2, cnum // 4, downsample=False),
#         )
#         self.decoder2 = nn.Sequential(
#             DeConv3D(cnum // 2, cnum // 4),
#             Conv3D(cnum // 4, cnum // 8, downsample=False),
#         )
#         self.decoder3 = nn.Sequential(
#             DeConv3D(cnum // 4, cnum // 8),
#             Conv3D(cnum // 8, cnum // 16, downsample=False),
#         )
#         self.decoder4 = nn.Sequential(
#             DeConv3D(cnum // 8, cnum // 16),
#             Conv3D(cnum // 16, cnum // 32, downsample=False),
#         )
#
#         self.fc = nn.Linear(cnum // 32, out_channels)
#         self.out_norm = nn.BatchNorm3d(out_channels)
#
#     def forward(self, x, feats):
#         decode0 = self.decoder0(x)
#         decode1 = self.decoder1(torch.cat([decode0, feats[3]], dim=1))
#         decode2 = self.decoder2(torch.cat([decode1, feats[2]], dim=1))
#         decode3 = self.decoder3(torch.cat([decode2, feats[1]], dim=1))
#         decode4 = self.decoder4(torch.cat([decode3, feats[0]], dim=1))
#         out = self.out_norm(self.fc(decode4.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3))
#
#         return out
