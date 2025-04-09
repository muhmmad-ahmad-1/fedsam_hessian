import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, n_cls):
        super(ResNet, self).__init__()
        resnet18 = models.resnet18()
        resnet18.fc = nn.Linear(512, n_cls)

        # Change BN to GN 
        resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

        resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

        resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
        resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

        resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
        resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

        resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
        resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

        assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'
        
        self.model = resnet18
    
    def forward(self, x):
        x = self.model(x)
        return x

