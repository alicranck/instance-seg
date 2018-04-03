from torchvision import models
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet = models.resnet18(True)
        self.upsample1 = UpsamplingBlock(512, 256)
        self.upsample2 = UpsamplingBlock(256, 128)
        self.upsample3 = UpsamplingBlock(128, 64)
        self.upsample4 = UpsamplingBlock(64, 64)
        self.upsample5 = UpsamplingBlock(64, 64)

    def forward(self, x):
        outputs = {}
        if (x != x).data.any():
            print("nan in input layer")
            #print(x)
        for name, module in list(self.resnet.named_children())[:-2]:
            x = module(x)
            if (x!=x).data.any():
                print("nan in layer "+name)
                #print(x)
            outputs[name] = x
            #print(name)

        features = outputs['layer4'] # Resnet output before final avgpool and fc layer
        features = self.upsample1(features, outputs['layer3'])
        if (features!=features).data.any():
            print("nan in upsampling layer 1")
            #print(features)
        features = self.upsample2(features, outputs['layer2'])
        if (features!=features).data.any():
            print("nan in upsampling layer 2")
            #print(features)
        features = self.upsample3(features, outputs['layer1'])
        if (features!=features).data.any():
            print("nan in upsampling layer 3")
            #print(features)
        features = self.upsample4(features, outputs['relu'])
        if (features!=features).data.any():
            print("nan in upsampling layer 4")
            #print(features)
        features = self.upsample5(features, 0)
        if (features!=features).data.any():
            print("nan in upsampling layer 5")
            #print(features)

        return features


class UpsamplingBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(UpsamplingBlock, self).__init__()
        self.upsamplingLayer = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        #self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride= 1) # Possibly more conv layers
        self.batchNorm = nn.BatchNorm2d(channels_out)

    def forward(self, x, skip_input):
        x = self.upsamplingLayer(x)
        if (x!=x).data.any():
            print('nan in upsampling layer')
        x = self.relu(x)
        if (x != x).data.any():
            print('nan in relu layer')
        #x = self.conv(x)
        #if (x!=x).data.any():
            #print('nan in upsampling block:conv layer')
        x = x + skip_input
        x = self.batchNorm(x)
        if (x!=x).data.any():
            print('nan in batch norm layer')

        return x
