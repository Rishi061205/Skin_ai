#we have used efficientNEt-B3 which is better than Vit(Vision transformer) for this task as it is a small dataset and efficientNet-B3 is better for small datasets and also it is faster than Vit.


import torch
import torch.nn as nn
import timm                              # for pretrained EfficientNet backbones

import config


def build_model(num_classes, pretrained=True):
    """
    Builds an EfficientNet-B3 with a custom classification head for skin diseases.

    Architecture:
        EfficientNet-B3 backbone (pretrained on ImageNet)
            └── Global Average Pooling  [already inside EfficientNet]
            └── Dropout(0.3)            [regularization — reduces overfitting]
            └── Linear(1536 → 512)      [intermediate layer]
            └── ReLU + Dropout(0.2)
            └── Linear(512 → 14)         [one score per disease class]
    """

    # Load pretrained backbone. timm handles downloading automatically.
    backbone = timm.create_model(
        "efficientnet_b3",
        pretrained   = pretrained,
        num_classes  = 0,               # 0 = remove the original classifier head
        global_pool  = "avg",           # global average pooling after conv layers
    )

    # Find out how many features the backbone outputs
    num_features = backbone.num_features  # EfficientNet-B3 → 1536

    # Our custom classification head
    classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )

    # Combine backbone + head into one model
    model = SkinDiseaseModel(backbone, classifier)
    return model


class SkinDiseaseModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone   = backbone
        self.classifier = classifier

    def forward(self, x):
        features = self.backbone(x)         # extract visual features
        logits   = self.classifier(features) # map features → class scores
        return logits


def get_optimizer(model):
    """
    Two-group optimizer: backbone gets a much smaller learning rate than
    the classifier head. This is called "differential learning rates."
    
    Why? The backbone already knows useful features from ImageNet.
    We want to gently nudge it, not overwrite everything it learned.
    The classifier head is brand new, so it needs a bigger learning rate.
    """
    backbone_params   = {"params": model.backbone.parameters(),
                         "lr": config.BACKBONE_LR}
    classifier_params = {"params": model.classifier.parameters(),
                         "lr": config.LEARNING_RATE}

    optimizer = torch.optim.AdamW(
        [backbone_params, classifier_params],
        weight_decay = config.WEIGHT_DECAY,
    )
    return optimizer


def get_scheduler(optimizer, steps_per_epoch):
    """
    OneCycleLR: ramps the learning rate up then down over training.
    Works much better than a fixed learning rate for fine-tuning.
    """
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr         = [config.BACKBONE_LR, config.LEARNING_RATE],
        steps_per_epoch = steps_per_epoch,
        epochs          = config.NUM_EPOCHS,
        pct_start       = 0.3,
    )
