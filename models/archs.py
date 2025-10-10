import torch
import torchvision
from torch import nn
from torch.nn import functional as F


def get_arch(arch_name: str, pretrained: bool = True, desc_dim: int = 2048):
    arch_name = arch_name.lower()
    if arch_name == "resnet50gem" and pretrained:
        return ResNet50Gem(desc_dim)
    elif arch_name == "resnet18gem" and pretrained:
        return ResNet18Gem(desc_dim)
    elif arch_name == "eigenplaces":
        return torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=desc_dim,
        )
    else:
        raise ValueError(f"Arch {arch_name} not supported")


class GeMPool(nn.Module):
    """GeM pooling with optional normalization."""

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # GeM pooling: (avg(x^p))^(1/p)
        pooled = F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

        flattened = pooled.flatten(1)
        return flattened


class ResNet50Gem(nn.Module):
    def __init__(self, descriptor_dim: int = 512):
        super().__init__()
        self.descriptor_dim = descriptor_dim

        # Load pretrained ResNet50
        resnet = torchvision.models.resnet50(weights="IMAGENET1K_V1")

        # Create feature extractor (everything except avgpool and fc)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # Freeze early layers
        self._freeze_layers([resnet.layer1, resnet.layer2])

        # Custom pooling and projection
        self.gem = GeMPool()  # We'll normalize at the end
        self.fc = nn.Linear(2048, descriptor_dim)

    def _freeze_layers(self, layers):
        """Helper method to freeze multiple layers."""
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.features(x)
        pooled = self.gem(features)
        descriptor = self.fc(pooled)
        return F.normalize(descriptor, p=2, dim=1)



class ResNet18Gem(nn.Module): 
    def __init__(self, descriptor_dim: int=512): 
        resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gem = GeMPool()
        self.fc = nn.Linear(512, descriptor_dim)

    def forward(self, x): 
        x = self.conv1(x)
        x = self.model.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gem(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)



class ResNet18Gem(nn.Module): 
    def __init__(self, descriptor_dim: int=512): 
        super().__init__()
        resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gem = GeMPool()
        self.fc = nn.Linear(512, descriptor_dim)

    def forward(self, x): 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gem(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

