import torch
import torch.nn as nn
import torchvision.models as models

class ViT_B_32(nn.Module):
    def __init__(self, num_classes=10, num_groups=8):  # num_groups for GroupNorm
        super(ViT_B_32, self).__init__()

        # Load the pretrained ViT-B/32 model
        self.vit = models.vision_transformer.vit_b_32(pretrained=True)

        # Modify the classifier (last Linear layer)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

        # Replace all LayerNorm layers with GroupNorm
        self._replace_layernorm_with_groupnorm(num_groups)

    def _replace_layernorm_with_groupnorm(self, num_groups):
        """Recursively replace all LayerNorm layers with GroupNorm"""
        for name, module in self.vit.named_modules():
            if isinstance(module, nn.LayerNorm):
                parent_module, attr = self._get_parent_module(name)
                setattr(parent_module, attr, nn.GroupNorm(num_groups, module.normalized_shape[0]))

    def _get_parent_module(self, module_name):
        """Helper to get parent module and attribute name"""
        components = module_name.split(".")
        parent = self.vit
        for comp in components[:-1]:
            parent = getattr(parent, comp)
        return parent, components[-1]

    def forward(self, x):
        return self.vit(x) 