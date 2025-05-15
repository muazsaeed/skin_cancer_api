import torch
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import CONFIG from config module
from backend.config import CONFIG, CONFIG_EVA02

# EfficientNet preprocessing - exactly as in test_model_special.py
efficientnet_transforms = A.Compose([
    A.Resize(CONFIG['img_size'], CONFIG['img_size']),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
    ),
    ToTensorV2()
])

# EVA-02 preprocessing
eva02_transforms = A.Compose([
    A.Resize(CONFIG_EVA02['img_size'], CONFIG_EVA02['img_size']),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
    ),
    ToTensorV2()
])

def preprocess_image(image_bytes, model_type):
    """
    Preprocess image bytes for the specified model type,
    using the exact same logic as in test_model_special.py
    
    Args:
        image_bytes: Image bytes from request
        model_type: String indicating which model to preprocess for ('efficient_b' or 'eva02')
        
    Returns:
        A preprocessed torch tensor ready for model inference
    """
    # Read image and convert to RGB
    if isinstance(image_bytes, bytes):
        # Convert bytes to PIL Image and then to numpy array
        img = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
    else:
        # If already a file-like object
        img = np.array(Image.open(image_bytes).convert("RGB"))
    
    # Apply appropriate transforms based on model type
    if model_type == 'efficient_b':
        device = CONFIG['device']
        transformed = efficientnet_transforms(image=img)
        img_tensor = transformed["image"].unsqueeze(0).to(device, dtype=torch.float)
    elif model_type == 'eva02':
        device = CONFIG_EVA02['device']
        transformed = eva02_transforms(image=img)
        img_tensor = transformed["image"].unsqueeze(0).to(device, dtype=torch.float)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'efficient_b' or 'eva02'")
    
    return img_tensor 