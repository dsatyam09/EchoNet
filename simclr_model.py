import torch
import torch.nn as nn
import torchvision.models as models

class SimCLR(nn.Module):
    def __init__(self, model_name='resnet50', projection_dim=128):
        super(SimCLR, self).__init__()
        
        # Select backbone architecture
        if model_name == 'resnet50':
            self.encoder = models.resnet50(pretrained=True)
            self.encoder.fc = nn.Identity()  # Remove classification layer
        elif model_name == 'vgg16':
            self.encoder = models.vgg16(pretrained=True)
            self.encoder.classifier = nn.Identity()  # Remove classification layer
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Define projection head
        self.projection_head = nn.Sequential(
            nn.Linear(self._get_encoder_output_dim(model_name), 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def _get_encoder_output_dim(self, model_name):
        if model_name == 'resnet50':
            return 2048
        elif model_name == 'vgg16':
            return 4096
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return projections

    def encode(self, x):
        return self.encoder(x)

    def project(self, x):
        return self.projection_head(x)
