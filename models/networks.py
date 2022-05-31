from models.unet_parts import *

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UNetEncoder(nn.Module):
    def __init__(self, n_channels=3):
        super(UNetEncoder, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5, (x1, x2, x3, x4)

class UNetDecoder(nn.Module):
    def __init__(self, n_channels=3):
        super(UNetDecoder, self).__init__()
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, enc_outs):
        x = self.sigmoid(x)
        x = self.up1(x, enc_outs[3])
        x = self.up2(x, enc_outs[2])
        x = self.up3(x, enc_outs[1])
        x = self.up4(x, enc_outs[0])
        x = self.outc(x)
        return nn.Tanh()(x)

