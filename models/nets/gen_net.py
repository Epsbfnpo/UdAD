from models.nets.blocks import *


class GenNet(nn.Module):
    def __init__(self, in_channels, cnum, num_classes, input_size):
        super(GenNet, self).__init__()
        self.encoder = Encoder(in_channels, cnum)
        self.decoder = Decoder(cnum)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.num_classes = num_classes

        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *input_size)
            x, enc_feats = self.encoder(dummy_input)
            x = self.decoder(x, enc_feats)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            self.fc_input_dim = x.size(1)

        self.fc = nn.Linear(self.fc_input_dim, self.num_classes)

    def forward(self, x):
        x, enc_feats = self.encoder(x)
        x = self.decoder(x, enc_feats)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits
