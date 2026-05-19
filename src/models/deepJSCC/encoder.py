from torch import nn

class ConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvWithPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
    
    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x

class DJSCC_Encoder(nn.Module):
    def __init__(self, config, C):
        super(DJSCC_Encoder, self).__init__()

        self.C = C

        self.conv1 = ConvWithPReLU(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = ConvWithPReLU(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv3 = ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv4 = ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv5 = ConvWithPReLU(in_channels=32, out_channels=self.C, kernel_size=5, padding=2)

    def forward(self, input_image):
        x = self.conv1(input_image)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        return x
    