import timm
import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.mul_(torch.sigmoid(x))
        else:
            return x * torch.sigmoid(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class SwishDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SwishDoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            Swish(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            Swish(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, encoder_name='tf_efficientnetv2_l', num_classes=1):
        super(UNet, self).__init__()
        self.encoder = timm.create_model(encoder_name, features_only=True, pretrained=True)
        encoder_channels = self.encoder.feature_info.channels()
        self.pool = nn.MaxPool2d(2, 2)
        self.upconv4 = nn.ConvTranspose2d(encoder_channels[-1], 512, kernel_size=2, stride=2)
        self.double_conv4 = DoubleConv(encoder_channels[-2] + 512, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.double_conv3 = DoubleConv(encoder_channels[-3] + 256, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.double_conv2 = DoubleConv(encoder_channels[-4] + 128, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.double_conv1 = DoubleConv(encoder_channels[-5] + 64, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid layer for final output

    def forward(self, x):
        enc_features = self.encoder(x)
        x = self.upconv4(enc_features[-1])
        x = torch.cat([x, enc_features[-2]], dim=1)
        x = self.double_conv4(x)
        x = self.upconv3(x)
        x = torch.cat([x, enc_features[-3]], dim=1)
        x = self.double_conv3(x)
        x = self.upconv2(x)
        x = torch.cat([x, enc_features[-4]], dim=1)
        x = self.double_conv2(x)
        x = self.upconv1(x)
        x = torch.cat([x, enc_features[-5]], dim=1)
        x = self.double_conv1(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)  # Apply sigmoid to ensure output is between 0 and 1
        return x
    

class SwishUNet(nn.Module):
    def __init__(self, encoder_name='tf_efficientnetv2_l', num_classes=1):
        super(SwishUNet, self).__init__()
        self.encoder = timm.create_model(encoder_name, features_only=True, pretrained=True)
        encoder_channels = self.encoder.feature_info.channels()
        self.pool = nn.MaxPool2d(2, 2)
        self.upconv4 = nn.ConvTranspose2d(encoder_channels[-1], 512, kernel_size=2, stride=2)
        self.double_conv4 = SwishDoubleConv(encoder_channels[-2] + 512, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.double_conv3 = SwishDoubleConv(encoder_channels[-3] + 256, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.double_conv2 = SwishDoubleConv(encoder_channels[-4] + 128, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.double_conv1 = SwishDoubleConv(encoder_channels[-5] + 64, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid layer for final output

    def forward(self, x):
        enc_features = self.encoder(x)
        x = self.upconv4(enc_features[-1])
        x = torch.cat([x, enc_features[-2]], dim=1)
        x = self.double_conv4(x)
        x = self.upconv3(x)
        x = torch.cat([x, enc_features[-3]], dim=1)
        x = self.double_conv3(x)
        x = self.upconv2(x)
        x = torch.cat([x, enc_features[-4]], dim=1)
        x = self.double_conv2(x)
        x = self.upconv1(x)
        x = torch.cat([x, enc_features[-5]], dim=1)
        x = self.double_conv1(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)  # Apply sigmoid to ensure output is between 0 and 1
        return x
