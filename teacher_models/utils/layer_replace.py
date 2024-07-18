import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation


def replace_classifier_to_numClasses(model, num_classes):
    classifier = model.get_classifier()
    if not isinstance(classifier, nn.Sequential):
        if isinstance(classifier, nn.Linear):
            in_features = classifier.in_features
            if hasattr(model, "classifier"):
                model.classifier = nn.Linear(in_features, num_classes)
            else:
                model.fc = nn.Linear(in_features, num_classes)
    else:
        for i, (name, module) in enumerate(classifier.named_modules()):
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                model.classifier[i-1] = nn.Linear(in_features, num_classes)

def replace_stem_for_cifar(model_name, model, printer=print):
    if "densenet" in model_name:
        original_stem = [model.features.conv0, model.features.norm0, model.features.relu0, model.features.pool0]
        out_channels = model.features.conv0.out_channels
        model.features.conv0 = nn.Conv2d(3, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        model.features.norm0 = nn.Identity()
        model.features.relu0 = nn.Identity()
        model.features.pool0 = nn.Identity()
        replaced_stem = [model.features.conv0]
        
    elif "efficientnet" in model_name:
        # out_channels = model.features[0][0].out_channels
        # if isinstance(model.features[0][1], nn.BatchNorm2d):
        #     norm_layer = nn.BatchNorm2d
        # else:
        #     norm_layer = None
        # original_stem = [model.features[0]]
        # model.features[0] = Conv2dNormActivation(
        #         3, out_channels, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=nn.SiLU
        #     )
        # replaced_stem = [model.features[0]]

        printer("{} can not be replaced for cifar dataset. So you should use advanced data transformer.".format(model_name))
    elif "resnet" in model_name:
        original_stem = [model.conv1, model.bn1, model.relu, model.maxpool]
        out_channels = model.conv1.out_channels
        model.conv1 = nn.Conv2d(3, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        model.bn1 = nn.Identity()
        model.relu = nn.Identity()
        model.maxpool = nn.Identity()
        replaced_stem = [model.conv1]
        

    printer("model stem layer is replaced from {} to {}".format(original_stem, replaced_stem))