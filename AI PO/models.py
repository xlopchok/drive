import torch
import torch.nn as nn
from torchvision import models

class EfficientNetRegression(nn.Module):
    def __init__(self):
        super(EfficientNetRegression, self).__init__()
        self.efficient_net = models.efficientnet_b1(pretrained=True)

        # Modify the first convolutional layer to accept 6 channels
        original_conv = self.efficient_net.features[0][0]
        self.efficient_net.features[0][0] = nn.Conv2d(
            in_channels=9,  
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        # Modify the classifier to output a single value for regression
        self.efficient_net.classifier[1] = nn.Linear(
            self.efficient_net.classifier[1].in_features, 1
        )

    def forward(self, x):
        return self.efficient_net(x)