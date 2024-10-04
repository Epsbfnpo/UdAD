import torch
import torch.nn as nn
import torch.nn.functional as F


def crop_or_pad(tensor, target_size):
    current_size = tensor.size()
    _, _, D, H, W = current_size
    D_target, H_target, W_target = target_size
    D_diff = D_target - D
    H_diff = H_target - H
    W_diff = W_target - W

    if D_diff < 0:
        D_start = (-D_diff) // 2
        tensor = tensor[:, :, D_start:D_start+D_target, :, :]
    elif D_diff > 0:
        pad_front = D_diff // 2
        pad_back = D_diff - pad_front
        tensor = F.pad(tensor, (0, 0, 0, 0, pad_front, pad_back))

    if H_diff < 0:
        H_start = (-H_diff) // 2
        tensor = tensor[:, :, :, H_start:H_start+H_target, :]
    elif H_diff > 0:
        pad_top = H_diff // 2
        pad_bottom = H_diff - pad_top
        tensor = F.pad(tensor, (0, 0, pad_top, pad_bottom, 0, 0))

    if W_diff < 0:
        W_start = (-W_diff) // 2
        tensor = tensor[:, :, :, :, W_start:W_start+W_target]
    elif W_diff > 0:
        pad_left = W_diff // 2
        pad_right = W_diff - pad_left
        tensor = F.pad(tensor, (pad_left, pad_right, 0, 0, 0, 0))

    return tensor


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


class DeConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, output_padding=1):
        super(DeConv3D, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=output_padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        output = self.block(x)
        return output


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
            Conv3D(cnum * 4, cnum * 4, downsample=True)
        )

    def forward(self, x):
        x1 = self.layer1(x)
        # print(f"Encoder Layer1 output size: {x1.size()}")
        x2 = self.layer2(x1)
        # print(f"Encoder Layer2 output size: {x2.size()}")
        x3 = self.layer3(x2)
        # print(f"Encoder Layer3 output size: {x3.size()}")
        return x3, [x1, x2]


class Decoder(nn.Module):
    def __init__(self, cnum):
        super(Decoder, self).__init__()
        self.deconv1 = DeConv3D(cnum * 4, cnum * 2, output_padding=(1, 0, 0))
        self.conv1 = nn.Sequential(
            Conv3D(cnum * 4, cnum * 2, downsample=False)
        )
        self.deconv2 = DeConv3D(cnum * 2, cnum, output_padding=(1, 1, 0))
        self.conv2 = nn.Sequential(
            Conv3D(cnum * 2, cnum, downsample=False)
        )

    def forward(self, x, enc_feats):
        x = self.deconv1(x)
        # print(f"Decoder Deconv1 output size: {x.size()}")
        # print(f"Encoder feature size at skip connection 1: {enc_feats[1].size()}")

        x = crop_or_pad(x, enc_feats[1].size()[2:])
        x = torch.cat([x, enc_feats[1]], dim=1)
        x = self.conv1(x)

        x = self.deconv2(x)
        # print(f"Decoder Deconv2 output size: {x.size()}")
        # print(f"Encoder feature size at skip connection 0: {enc_feats[0].size()}")

        x = crop_or_pad(x, enc_feats[0].size()[2:])
        x = torch.cat([x, enc_feats[0]], dim=1)
        x = self.conv2(x)
        return x



