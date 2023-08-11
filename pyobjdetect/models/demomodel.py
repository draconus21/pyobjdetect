import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


from pyobjdetect.transforms import base as transforms
from pyobjdetect.dataset.pennfundan import PennFudanDatatset


def finetune_example(num_classes=2):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # update number of classes to the one we want
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, num_classes=num_classes)

    return model


def new_backbone_example(num_classes=2):
    # load a pre-trained model for classification and return only the features
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features

    # FasterRCNN needs to know the number of output channles in a backbone.
    # For movilenet_v2, it's 1280, so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5x3 anchors per spatial location,
    # with 5 different sizes and 3 different aspect ratios.
    # We have a Tupe[Tuple[int]] because each features map could
    # potentially have different sizes and aspect ratios.
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will use to perform
    # the region of interest croptting, as well as the size of the crop
    # after rescaling.
    # If youre backboke returens a Tensor, featmap_names is expected to be [0].
    # More generally, the backbone should return an OrderedDict[Tensor], and in
    # featmap_names you can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)

    # put the pieces to gether inside a FasterRCNN model
    model = FasterRCNN(
        backbone=backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler
    )

    return model


# instalnce segmentation model example
def get_model_instance_segmentation(num_classes=2):
    # load an isntance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get numer of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv4_mask.in_channels
    hidden_layer = 256

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def test_forward():
    import os
    import torch
    import logging
    import numpy as np
    from pyobjdetect.utils import logutils, viz
    from pyobjdetect.pytorch_reference_detection import utils as pyt_utils

    logutils.setupLogging("DEBUG")
    batch_size = 2
    num_workers = 4

    root = os.path.join(os.environ["ODT_DATA_DIR"], "PennFudanPed")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    dataset = PennFudanDatatset(root, transforms=transforms.get_example_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=pyt_utils.collate_fn
    )

    # For training
    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    logging.info(f"{len(images)} images and {len(targets)} targets")

    target = targets[0]
    matToShow = [target["masks"][i].numpy() for i in range(target["masks"].shape[0])]
    viz.quickmatshow(matToShow, title=f"example")

    output = model(images, targets)  # Returns losses and detections

    logging.info(f"output: {output}")
    viz.show()

    # For inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)

    logging.info(f"{len(predictions)} predictions: {predictions}")


if __name__ == "__main__":
    # finetune_example()
    test_forward()
