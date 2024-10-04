# from models.nets.blocks import *
#
#
# class DisNet(nn.Module):
#     def __init__(self, in_channels, cnum, out_channels):
#         super(DisNet, self).__init__()
#         self.out_channels = out_channels
#         self.encoder = Encoder(in_channels, cnum)
#         self.relu = nn.ReLU(inplace=True)
#         self.sig = nn.Sigmoid()
#         self.fc1 = torch.nn.utils.spectral_norm(nn.Linear(cnum * 32, 4))
#
#     def forward(self, x):
#         feats = self.encoder(x)
#         z = feats[-1]
#         z = self.relu(self.fc1(z.permute(0, 2, 3, 4, 1)))
#         z = z.view(z.shape[0], -1)
#         if not hasattr(self, 'fc2'):
#             input_dim = z.shape[1]
#             self.fc2 = torch.nn.utils.spectral_norm(nn.Linear(input_dim, self.out_channels)).to(z.device)
#         z = self.sig(self.fc2(z))
#
#         return z
