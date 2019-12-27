import os
import torch
from torch import nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from backbone_utils import resnet_fpn_backbone

def fasterrcnn_resnet_fpn(backbone_name, backbone_path, pretrained=False, progress=True,
                          num_classes=91, **kwargs):
    """
    Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets dictionary,
    containing:
        - boxes (``Tensor[N, 4]``): the ground-truth boxes in ``[x0, y0, x1, y1]`` format, with values
          between ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``Tensor[N, 4]``): the predicted boxes in ``[x0, y0, x1, y1]`` format, with values between
          ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction

    Example::

        >>> model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # if pretrained:
    # no need to download the backbone if pretrained is set
    pretrained_backbone = pretrained
    in_channels = 0
    if backbone_name == 'resnet34':
        in_channels = 64
    elif backbone_name == 'resnet50':
        in_channels = 256
    backbone = resnet_fpn_backbone(backbone_name, backbone_path, in_channels)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    # if pretrained:
    #     state_dict = torchvision.load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def vgg_model(backbone_name, num_classes, backbone_path, pre_trained=False):
    model = torchvision.models.__dict__[backbone_name](pretrained=pre_trained)
    if os.path.exists(backbone_path):
        print('loading backbone models')
        state_dict = torch.load(backbone_path)
        model.load_state_dict({k:v for k,v in state_dict.items() if k in model.state_dict()})
        # model.load_state_dict(state_dict)
    backbone = nn.Sequential(*list(model.features._modules.values())[:-1])
    
    # Fix the layers before conv3:
    for layer in range(10):
        for p in backbone[layer].parameters(): p.requires_grad = False
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 512
    
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)
    
    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    
    return model
