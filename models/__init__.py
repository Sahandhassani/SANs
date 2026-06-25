from .alexnet import get_alexnet
from .custom_cnn import get_custom_cnn
from .pretrained_models import (
    get_resnet18,
    get_resnet50,
    get_densenet121,
    get_efficientnet_b0,
    get_vit_b16,
)


def build_model(model_name, num_classes=2, pretrained=True):
    model_name = model_name.lower()

    if model_name == "alexnet":
        return get_alexnet(num_classes=num_classes, pretrained=pretrained)

    if model_name == "customcnn":
        return get_custom_cnn(num_classes=num_classes)

    if model_name == "resnet18":
        return get_resnet18(num_classes=num_classes, pretrained=pretrained)

    if model_name == "resnet50":
        return get_resnet50(num_classes=num_classes, pretrained=pretrained)

    if model_name == "densenet121":
        return get_densenet121(num_classes=num_classes, pretrained=pretrained)

    if model_name == "efficientnet_b0":
        return get_efficientnet_b0(num_classes=num_classes, pretrained=pretrained)

    if model_name == "vit_b16":
        return get_vit_b16(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"Unknown model name: {model_name}")
