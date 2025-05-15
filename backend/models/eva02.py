import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import numpy as np
from backend.utils.preprocess import preprocess_image
from backend.config import CONFIG_EVA02 as CONFIG

class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None):
        super(ISICModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)

        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        return self.sigmoid(self.model(images))

def load_eva02_model(weights_path):
    """
    Load the EVA-02 model with pre-trained weights
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
def predict_with_eva02(model, image_tensor):
    """
    Run inference with the EVA-02 model
    
    Args:
        model: Loaded EVA-02 model
        image_tensor: Preprocessed image tensor [1, 3, 336, 336]
        
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