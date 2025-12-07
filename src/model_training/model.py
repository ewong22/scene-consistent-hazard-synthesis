import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance_segmentation(num_classes):
    """
    Returns a Mask R-CNN model with a ResNet-50-FPN backbone.
    
    Args:
        num_classes (int): Number of classes (including background).
                           For your project: 2 (0=Background, 1=Hazard).
    """
    # 1. Load a model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # 2. Replace the box predictor (Bounding Box Head)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one for our custom num_classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 3. Replace the mask predictor (Segmentation Head)
    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the mask predictor with a new one for our custom num_classes
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

if __name__ == "__main__":
    # Test the model definition
    # We use 2 classes: 0 = Background, 1 = Hazard (Debris)
    model = get_model_instance_segmentation(num_classes=2)
    print("Mask R-CNN model initialized successfully.")
    print(model)