import torch
import torch.nn as nn
import torchvision.models as models

def get_efficientnet_b3_model(num_classes=2, pretrained=True):
    # Load EfficientNet B3 with option to use pretrained weights
    model = models.efficientnet_b3(pretrained=pretrained)
    
    # Freeze all layers except classifier
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Get input features for classifier
    num_features = model.classifier[1].in_features
    
    # Replace classifier with custom layers
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1536),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1536, num_classes)
    )

    return model
