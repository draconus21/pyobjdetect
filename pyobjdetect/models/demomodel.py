import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def finetune_example(num_classes=2):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the classifier with a new one, that has  num_classes which is user defined
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


if __name__ == "__main__":
    finetune_example()
