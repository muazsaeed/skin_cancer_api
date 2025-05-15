import os
import torch
from backend.models.efficientnet import load_efficientnet_model, predict_with_efficientnet, ISICModel as EfficientNetModel
from backend.models.eva02 import load_eva02_model, predict_with_eva02, ISICModel as EVA02Model
from backend.config import CONFIG, CONFIG_EVA02

# Model cache to avoid reloading
_model_cache = {}

def get_model(model_type, models_dir=None):
    """
    Get a model instance based on the model type, using caching to avoid reloading
    
    Args:
        model_type: String ('efficient_b' or 'eva02')
        models_dir: Directory containing model weight files
        
    Returns:
        Loaded model ready for inference
    """
    # If model_dir is not provided, use a default relative to the backend directory
    if models_dir is None:
        # Get the backend directory (assuming this file is in backend/models/__init__.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.abspath(os.path.join(current_dir, ".."))
        models_dir = os.path.join(backend_dir, "weights")
    
    # Print model directory for debugging
    print(f"Looking for models in: {models_dir}")
    
    # If model is already in cache, return it
    if model_type in _model_cache:
        return _model_cache[model_type]
    
    # Get the model based on the model type
    if model_type == "efficient_b":
        checkpoint_path = os.path.join(models_dir, "AUROC0.5180_Loss0.6361_epoch47.bin")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"EfficientNet weights not found at: {checkpoint_path}")
        
        # Create and load model using the exact approach from test_model_special.py
        model = EfficientNetModel(CONFIG['model_name'], pretrained=False)
        model.load_state_dict(torch.load(checkpoint_path, map_location=CONFIG['device']))
        model.to(CONFIG['device'])
        model.eval()
    elif model_type == "eva02":
        checkpoint_path = os.path.join(models_dir, "AUROC0.5185_Loss0.3027_epoch39.bin")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"EVA-02 weights not found at: {checkpoint_path}")
        
        # Create and load model using the approach from the EVA02 code
        model = EVA02Model(CONFIG_EVA02['model_name'], pretrained=False)
        model.load_state_dict(torch.load(checkpoint_path, map_location=CONFIG_EVA02['device']))
        model.to(CONFIG_EVA02['device'])
        model.eval()
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'efficient_b' or 'eva02'")
    
    # Cache the model
    _model_cache[model_type] = model
    
    return model

def predict(model_type, image_tensor):
    """
    Run prediction with the specified model type
    
    Args:
        model_type: String ('efficient_b' or 'eva02')
        image_tensor: Preprocessed image tensor
        
    Returns:
        Tuple of (prediction, confidence)
    """
    # Get the model
    model = get_model(model_type)
    
    # Run prediction based on model type - using the exact approach from test_model_special.py
    if model_type == "efficient_b":
        with torch.no_grad():
            output = model(image_tensor)
            prob = output.item()
            label = int(prob > 0.5)
        return label, prob
    elif model_type == "eva02":
        with torch.no_grad():
            output = model(image_tensor)
            prob = output.item()
            label = int(prob > 0.5)
        return label, prob
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'efficient_b' or 'eva02'") 