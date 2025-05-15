import torch

# Configuration for EfficientNet model
CONFIG = {
    "seed": 42,
    "img_size": 224,
    "model_name": "tf_efficientnet_b0_ns",
    "valid_batch_size": 32,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

# Configuration for EVA-02 model
CONFIG_EVA02 = {
    "seed": 42,
    "img_size": 336,
    "model_name": "eva02_small_patch14_336.mim_in22k_ft_in1k",
    "valid_batch_size": 32,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
} 