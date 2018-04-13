from torchvision import models
import torch
import torch.nn as nn



class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet = models.resnet18(True)
        self.upsample1 = UpsamplingBlock(512, 256)
        self.upsample2 = UpsamplingBlock(256, 128)
        self.upsample3 = UpsamplingBlock(128, 64)
        self.upsample4 = UpsamplingBlock(64, 64)
        self.upsample5 = UpsamplingBlock(64, 64, skip=False)

    def forward(self, x):
        outputs = {}
        for name, module in list(self.resnet.named_children())[:-2]:
            if (x!=x).data.any():
                print("nan in layer "+name)
            x = module(x)
            outputs[name] = x
        features = outputs['layer4']  # Resnet output before final avgpool and fc layer
        features = self.upsample1(features, outputs['layer3'])
        if (features!=features).data.any():
            print("nan in upsampling layer 1")
        features = self.upsample2(features, outputs['layer2'])
        if (features!=features).data.any():
            print("nan in upsampling layer 2")
        features = self.upsample3(features, outputs['layer1'])
        if (features!=features).data.any():
            print("nan in upsampling layer 3")
        features = self.upsample4(features, outputs['relu'])
        if (features!=features).data.any():
            print("nan in upsampling layer 4")
        features = self.upsample5(features)
        if (features!=features).data.any():
            print("nan in upsampling layer 5")

        return features


class UpsamplingBlock(nn.Module):
    def __init__(self, channels_in, channels_out, skip=True):
        super(UpsamplingBlock, self).__init__()
        self.upsamplingLayer = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=2)
        if skip:
            self.conv = nn.Conv2d(2*channels_out, channels_out, kernel_size=1, stride=1)  # Possibly more conv layers
        else:
            self.conv = nn.Conv2d(channels_out, channels_out, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()
        self.batchNorm = nn.BatchNorm2d(channels_out)

    def forward(self, x, skip_input=None):
        x = self.upsamplingLayer(x)
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)
        x = self.conv(x)
        x = self.relu1(x)
        x = self.batchNorm(x)
        return x
