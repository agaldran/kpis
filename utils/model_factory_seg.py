import sys
import segmentation_models_pytorch as smp
def get_model(model_name, in_c=3, n_classes=1):


    ## FPNET ##
    # RESNET
    if model_name == 'fpn_resnet18':
        model = smp.FPN(encoder_name='resnet18', encoder_weights='imagenet', in_channels=in_c, classes=1)
    elif model_name == 'fpn_resnet34':
        model = smp.FPN(encoder_name='resnet34', encoder_weights='imagenet', in_channels=in_c, classes=1)
    elif model_name == 'fpn_resnet50':
        model = smp.FPN(encoder_name='resnet50', encoder_weights='imagenet', in_channels=in_c, classes=1)
    elif model_name == 'fpn_resnet101':
        model = smp.FPN(encoder_name='fpnet_resnet101', encoder_weights='imagenet', in_channels=in_c, classes=1)
    elif model_name == 'fpn_resnet152':
        model = smp.FPN(encoder_name='fpnet_resnet152', encoder_weights='imagenet', in_channels=in_c, classes=1)
    # MOBILENET
    elif model_name == 'fpn_mobilenet':
        model = smp.FPN(encoder_name='mobilenet_v2', encoder_weights='imagenet', in_channels=in_c, classes=1)
    # RESNEXT50
    elif model_name == 'fpn_resnext50_imagenet':
        model = smp.FPN(encoder_name='resnext50_32x4d', encoder_weights='imagenet', in_channels=in_c, classes=1)
    elif model_name == 'fpn_resnext50_ssl':
        model = smp.FPN(encoder_name='resnext50_32x4d', encoder_weights='ssl', in_channels=in_c, classes=1)
    elif model_name == 'fpn_resnext50_swsl':
        model = smp.FPN(encoder_name='resnext50_32x4d', encoder_weights='swsl', in_channels=in_c, classes=1)
    # RESNEXT101
    # 32X4
    elif model_name == 'fpn_resnext101_32x4d_ssl':
        model = smp.FPN(encoder_name='resnext101_32x4d', encoder_weights='ssl', in_channels=in_c, classes=1)
    elif model_name == 'fpn_resnext101_32x4d_swsl':
        model = smp.FPN(encoder_name='resnext101_32x4d', encoder_weights='swsl', in_channels=in_c, classes=1)
    # 32X8
    elif model_name == 'fpn_resnext101_32x8d_imagenet':
        model = smp.FPN(encoder_name='resnext101_32x8d', encoder_weights='imagenet', in_channels=in_c, classes=1)
    elif model_name == 'fpn_resnext101_32x8d_instagram':
        model = smp.FPN(encoder_name='resnext101_32x8d', encoder_weights='instagram', in_channels=in_c, classes=1)
    elif model_name == 'fpn_resnext101_32x8d_ssl':
        model = smp.FPN(encoder_name='resnext101_32x8d', encoder_weights='ssl', in_channels=in_c, classes=1)
    elif model_name == 'fpn_resnext101_32x8d_swsl':
        model = smp.FPN(encoder_name='resnext101_32x8d', encoder_weights='swsl', in_channels=in_c, classes=1)
    # DPN
    elif model_name == 'fpn_dpn68':
        model = smp.FPN(encoder_name='dpn68', encoder_weights='imagenet', in_channels=in_c, classes=1)
    elif model_name == 'fpn_dpn68b':
        model = smp.FPN(encoder_name='dpn68b', encoder_weights='imagenet+5k', in_channels=in_c, classes=1)
    # MIT
    elif model_name == 'fpn_mitb_0':
        model = smp.FPN(encoder_name='mit_b0', encoder_weights='imagenet', in_channels=in_c, classes=1)
    elif model_name == 'fpn_mitb_1':
        model = smp.FPN(encoder_name='mit_b1', encoder_weights='imagenet', in_channels=in_c, classes=1)
    elif model_name == 'fpn_mitb_2':
        model = smp.FPN(encoder_name='mit_b2', encoder_weights='imagenet', in_channels=in_c, classes=1)
    elif model_name == 'fpn_mitb_3':
        model = smp.FPN(encoder_name='mit_b3', encoder_weights='imagenet', in_channels=in_c, classes=1)
    elif model_name == 'fpn_mitb_4':
        model = smp.FPN(encoder_name='mit_b4', encoder_weights='imagenet', in_channels=in_c, classes=1)
    elif model_name == 'fpn_mitb_5':
        model = smp.FPN(encoder_name='mit_b5', encoder_weights='imagenet', in_channels=in_c, classes=1)
    # Mobileone
    elif model_name == 'fpn_mobileone_s0':
        model = smp.FPN(encoder_name='mobileone_s0', encoder_weights='imagenet', in_channels=in_c, classes=1)
    elif model_name == 'fpn_mobileone_s1':
        model = smp.FPN(encoder_name='mobileone_s1', encoder_weights='imagenet', in_channels=in_c, classes=1)
    elif model_name == 'fpn_mobileone_s2':
        model = smp.FPN(encoder_name='mobileone_s2', encoder_weights='imagenet', in_channels=in_c, classes=1)
    elif model_name == 'fpn_mobileone_s3':
        model = smp.FPN(encoder_name='mobileone_s3', encoder_weights='imagenet', in_channels=in_c, classes=1)
    elif model_name == 'fpn_mobileone_s4':
        model = smp.FPN(encoder_name='mobileone_s4', encoder_weights='imagenet', in_channels=in_c, classes=1)


    # just trying some stuff that wont work
    elif model_name == 'pspn_resnet50':
        model = smp.PSPNet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=in_c, classes=1)


    else:
        sys.exit('not a valid model_name, check models.get_model.py')

    setattr(model, 'n_classes', n_classes)


    return model


