import torch
import functools
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from basic_layers import ResidualBlock
from attention_module import AttentionModule_stage1, AttentionModule_stage2, AttentionModule_stage3, AttentionModule_stage0
from attention_module import AttentionModule_stage1_cifar, AttentionModule_stage2_cifar, AttentionModule_stage3_cifar


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
    )

class ResNet_Attention_UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)
        
        self.residual_block4 = ResidualBlock(512, 512)
        self.attention_module4 = AttentionModule_stage3(512, 512)
        self.residual_block3 = ResidualBlock(256, 256)
        self.attention_module3 = AttentionModule_stage2(256, 256)
        self.residual_block2 = ResidualBlock(128, 128)
        self.attention_module2 = AttentionModule_stage1(128, 128)
        self.residual_block1 = ResidualBlock(64, 64)
        self.attention_module1 = AttentionModule_stage0(64, 64)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(640, 256, 3, 1)
        self.conv_up1 = convrelu(192, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, 5, 1)
        
    # Model with 2 layers
    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)

        layer2 = self.residual_block2(layer2)
        layer2 = self.attention_module2(layer2)
        x = self.upsample(layer2)
        layer1 = self.residual_block1(layer1)
        layer1 = self.attention_module1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)
        return out

    # # Model with 4 layers
    # def forward(self, input):
    #     x_original = self.conv_original_size0(input)
    #     x_original = self.conv_original_size1(x_original)

    #     layer0 = self.layer0(input)
    #     layer1 = self.layer1(layer0)
    #     layer2 = self.layer2(layer1)

    #     layer3 = self.layer3(layer2)
    #     layer4 = self.layer4(layer3)
        
    #     layer4 = self.residual_block4(layer4)
    #     layer4 = self.attention_module4(layer4)
    #     x = self.upsample(layer4)
    #     layer3 = self.residual_block3(layer3)
    #     layer3 = self.attention_module3(layer3)
    #     x = torch.cat([x, layer3], dim=1)
    #     x = self.conv_up3(x)

    #     x = self.upsample(x)
    #     layer2 = self.residual_block2(layer2)
    #     layer2 = self.attention_module2(layer2)
    #     x = torch.cat([x, layer2], dim=1)
    #     x = self.conv_up2(x)

    #     x = self.upsample(x)
    #     layer1 = self.residual_block1(layer1)
    #     layer1 = self.attention_module1(layer1)
    #     x = torch.cat([x, layer1], dim=1)
    #     x = self.conv_up1(x)

    #     x = self.upsample(x)
    #     layer0 = self.layer0_1x1(layer0)
    #     x = torch.cat([x, layer0], dim=1)
    #     x = self.conv_up0(x)

    #     x = self.upsample(x)
    #     x = torch.cat([x, x_original], dim=1)
    #     x = self.conv_original_size2(x)

    #     out = self.conv_last(x)
    #     return out
    
    
    
class EfficientNet_Attention_UNet(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.base_model = model.from_pretrained('efficientnet-b7')

        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)
        
        self.residual_block4 = ResidualBlock(512, 512)
        self.attention_module4 = AttentionModule_stage3(512, 512)
        self.residual_block3 = ResidualBlock(256, 256)
        self.attention_module3 = AttentionModule_stage2(256, 256)
        self.residual_block2 = ResidualBlock(128, 128)
        self.attention_module2 = AttentionModule_stage1(128, 128)
        self.residual_block1 = ResidualBlock(64, 64)
        self.attention_module1 = AttentionModule_stage0(64, 64)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(640, 256, 3, 1)
        self.conv_up1 = convrelu(320, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_size0 = convrelu(32, 64, 3, 1)
        self.conv_size1 = convrelu(48, 64, 3, 1)
        self.conv_size2 = convrelu(80, 128, 3, 1)
        self.conv_size3 = convrelu(224, 256, 3, 1)
        self.conv_size4 = convrelu(640, 512, 3, 1)

        self.conv_last = nn.Conv2d(64, 5, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        endpoints = self.base_model.extract_endpoints(input)
        layer0 = endpoints['reduction_1'] # torch.Size([1, 32, 112, 112])
        layer0 = self.conv_size0(layer0)
        layer1 = endpoints['reduction_2'] # torch.Size([1, 48, 56, 56])
        layer1 = self.conv_size1(layer1)
        layer2 = endpoints['reduction_3'] # torch.Size([1, 80, 28, 28])
        layer2 = self.conv_size2(layer2)
        layer3 = endpoints['reduction_4'] # torch.Size([1, 224, 14, 14])
        layer3 = self.conv_size3(layer3)
        layer4 = endpoints['reduction_5'] # torch.Size([1, 640, 7, 7])
        layer4 = self.conv_size4(layer4)
        
        layer4 = self.residual_block4(layer4)
        layer4 = self.attention_module4(layer4)
        x = self.upsample(layer4)
        layer3 = self.residual_block3(layer3)
        layer3 = self.attention_module3(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.residual_block2(layer2)
        layer2 = self.attention_module2(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.residual_block1(layer1)
        layer1 = self.attention_module1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)
        return out
    
    
def build_model(device, model_name, base_model = None):
    if model_name == "resnet":
        return ResNet_Attention_UNet()
    elif model_name == "efficientnet":
        return EfficientNet_Attention_UNet(base_model)