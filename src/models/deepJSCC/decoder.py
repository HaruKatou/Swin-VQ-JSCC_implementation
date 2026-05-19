from torch import nn

class TransConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activate=nn.PReLU(), padding=0, output_padding=0):
        super(TransConvWithPReLU, self).__init__()
        self.transconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.activate = activate
        
        if activate == nn.PReLU():
            nn.init.kaiming_normal_(self.transconv.weight, mode='fan_out',
                                nonlinearity='leaky_relu')
        else:
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x):
        x = self.transconv(x)
        x = self.activate(x)
        return x

class DJSCC_Decoder(nn.Module):
    def __init__(self, config, C):
        super(DJSCC_Decoder, self).__init__()

        self.C = C

        self.tconv1 = TransConvWithPReLU(in_channels=self.C, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv2 = TransConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv3 = TransConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv4 = TransConvWithPReLU(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tconv5 = TransConvWithPReLU(in_channels=16, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1,activate=nn.Sigmoid())

    def forward(self, x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        return x
