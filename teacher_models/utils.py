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
                model.classifier[i] = nn.Linear(in_features, num_classes)