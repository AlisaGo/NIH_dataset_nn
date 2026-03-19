"""
models.py

Pre-trained CNN architectures adapted for multi-label classification on the NIH Chest X-Ray dataset.
"""

import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import resnet50, ResNet50_Weights


class MultiLabelResNet(nn.Module):
    """
    A ResNet-50 based multi-label classifier.
    """
    def __init__(self, num_labels):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        base_model = resnet50(weights=weights)

        in_feats = base_model.fc.in_features
        base_model.fc = nn.Linear(in_feats, num_labels)

        self.model = base_model

    def forward(self, x):
        return self.model(x)

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_head(self):
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def set_trainable(self, freeze_mode="head_only"):
        self.freeze_all()
        self.unfreeze_head()

        if freeze_mode == "head_only":
            pass

        elif freeze_mode == "last_block":
            for param in self.model.layer4.parameters():
                param.requires_grad = True

        elif freeze_mode == "last_two_blocks":
            for param in self.model.layer3.parameters():
                param.requires_grad = True
            for param in self.model.layer4.parameters():
                param.requires_grad = True

        elif freeze_mode == "full":
            for param in self.model.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

    def get_trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def get_classifier_parameters(self):
        return self.model.fc.parameters()


class MultiLabelMobileNet(nn.Module):
    """
    A MobileNetV2-based multi-label classifier.
    """
    def __init__(self, num_labels):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT
        base_model = mobilenet_v2(weights=weights)

        in_feats = base_model.classifier[1].in_features
        base_model.classifier[1] = nn.Linear(in_feats, num_labels)

        self.model = base_model

    def forward(self, x):
        return self.model(x)

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_head(self):
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def set_trainable(self, freeze_mode="head_only"):
        self.freeze_all()
        self.unfreeze_head()

        if freeze_mode == "head_only":
            pass

        elif freeze_mode == "last_block":
            for param in self.model.features[-1].parameters():
                param.requires_grad = True

        elif freeze_mode == "last_two_blocks":
            for block in self.model.features[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

        elif freeze_mode == "full":
            for param in self.model.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

    def get_trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def get_classifier_parameters(self):
        return self.model.classifier.parameters()