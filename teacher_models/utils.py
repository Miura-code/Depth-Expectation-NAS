import torch
import torch.nn as nn


def replace_classifier_to_numClasses(model, num_classes):
    classifier = model.get_classifier()
    if not isinstance(classifier, nn.Sequential):
        if isinstance(classifier, nn.Linear):
            in_features = classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    else:
        for i, (name, module) in enumerate(classifier.named_modules()):
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                model.classifier[i-1] = nn.Linear(in_features, num_classes)

def freeze_model(model, unfreeze: bool = False):
    """
        freeze the model parameter excepting last classification layer
        Args:
            model: ニューラルネットモデル
            unfreeze: 指定することでモデルのすべての層の学習を開始する
    """
    if unfreeze:
        for name, params in model.named_parameters():
            params.require_grad = True
    else:
        classifier_layer = model.get_classifier()
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in classifier_layer.named_parameters():
            param.requires_grad = True