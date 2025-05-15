import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import numpy as np
from backend.utils.preprocess import preprocess_image
from backend.config import CONFIG

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'


class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None):
        super(ISICModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        output = self.sigmoid(self.linear(pooled_features))
        return output

def load_efficientnet_model(weights_path):
    """
    Load the EfficientNet-B0 model with pre-trained weights
    """
    # Initialize model using CONFIG dictionary
    model = ISICModel(CONFIG['model_name'], pretrained=False)
    
    # Load weights
    model.load_state_dict(torch.load(weights_path, map_location=CONFIG['device']))
    
    # Set model to evaluation mode
    model.to(CONFIG['device'])
    model.eval()
    
    return model

@torch.inference_mode()
def predict_with_efficientnet(model, image_tensor):
    """
    Run inference with the EfficientNet-B0 model
    
    Args:
        model: Loaded EfficientNet model
        image_tensor: Preprocessed image tensor [1, 3, 224, 224]
        
    Returns:
        prediction: Binary prediction (0 or 1)
        confidence: Prediction confidence (0-1)
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Move tensor to the device specified in CONFIG
    image_tensor = image_tensor.to(CONFIG['device'])
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        prob = output.item()
    
    # Make binary prediction
    label = int(prob > 0.5)
    
    return label, prob 