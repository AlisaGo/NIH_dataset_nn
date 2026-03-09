"""
models.py

Pre-trained CNN architectures adapted for multi-label classification on the NIH Chest X-Ray dataset.

Models:
  MultiLabelResNet:
    Uses a ResNet-50 backbone pre-trained on ImageNet, with the final fully-connected layer replaced
    to output `num_labels` logits for multi-label classification.

  MultiLabelMobileNet:
    Uses a MobileNetV2 backbone pre-trained on ImageNet, with the classifier head’s final linear layer
    replaced to produce `num_labels` outputs for multi-label tasks.
"""

import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import resnet50, ResNet50_Weights

#############################################
# 3) MODEL DEFINITION
#############################################
class MultiLabelResNet(nn.Module):
    """
    A ResNet-based multi-label classifier.
    The final layer is adapted to have `num_labels` outputs.
    """
    def __init__(self,num_labels):
        super().__init__()
        # Start from a pretrained resnet
        weights = ResNet50_Weights.DEFAULT
        base_model = resnet50(weights=weights)
        # Replace the last fc to classify NUM_LABELS
        in_feats = base_model.fc.in_features
        base_model.fc = nn.Linear(in_feats, num_labels)
        self.model = base_model

    def forward(self, x):
        """
        x shape: [batch_size, 3, H, W]
        returns: [batch_size, NUM_LABELS]
        """
        return self.model(x)


class MultiLabelMobileNet(nn.Module):
    """
    A MobileNetV2-based multi-label classifier.
    The final layer is adapted to have `num_labels` outputs.
    """

    def __init__(self, num_labels):
        super().__init__()
        # Load pretrained MobileNetV2

        weights = MobileNet_V2_Weights.DEFAULT
        base_model = mobilenet_v2(weights=weights)

        # Get input feature size of the classifier
        in_feats = base_model.classifier[1].in_features

        # Replace the classifier with a new one for multi-label output
        base_model.classifier[1] = nn.Linear(in_feats, num_labels)

        self.model = base_model

    def forward(self, x):
        return self.model(x)
