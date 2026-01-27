import segmentation_models_pytorch as smp

def unet(num_classes):
    return smp.Unet("resnet34", encoder_weights="imagenet", classes=num_classes)

def deeplab(num_classes):
    return smp.DeepLabV3Plus("resnet50", encoder_weights="imagenet", classes=num_classes)
