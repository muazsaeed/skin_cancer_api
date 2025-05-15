# Skin Cancer Detection API

![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)
![Python version](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.2-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-orange.svg)

A robust REST API for skin cancer detection using deep learning models. This project provides an easy-to-deploy backend service that uses state-of-the-art models (EfficientNet-B0 and EVA-02) for predicting malignancy in skin lesions.

## 📋 Features

- **Dual-Model Architecture**:
  - EfficientNet-B0 (224x224 input size)
  - EVA-02 (336x336 input size)
- **RESTful API Interface** with FastAPI
- **Docker Support** for containerized deployment
- **Automatic API Documentation** with Swagger UI
- **Production-Ready** with error handling, logging, and health checks

## 🔧 Installation

### Option 1: Local Installation

1. Clone this repository:
```bash
git clone https://github.com/muazsaeed/skin_cancer_api.git
cd skin_cancer_api
```

2. Install dependencies:
```bash
pip install -r backend/requirements.txt
```

3. Run the API:
```bash
uvicorn backend.main:app --reload
```

The model weights are already included in the `backend/weights` directory:
- `AUROC0.5180_Loss0.6361_epoch47.bin` (EfficientNet-B0 weights)
- `AUROC0.5185_Loss0.3027_epoch39.bin` (EVA-02 weights)

### Option 2: Docker Deployment

```bash
# Build the Docker image
docker build -t skin-cancer-api -f backend/Dockerfile .

# Run the container
docker run -p 8000:8000 skin-cancer-api
```

## 📚 API Endpoints

Once the API is running, you can access the interactive documentation at:

http://localhost:8000/docs

### Main Endpoints:

- `GET /health` - Health check endpoint
- `GET /model-check` - Check if model files are available
- `POST /predict` - Make a skin cancer prediction

## 📝 Example Usage

### Python

```python
import requests

url = "http://localhost:8000/predict"
params = {"model_type": "eva02"}  # Use 'efficient_b' for EfficientNet

# Open an image file
with open("skin-image.jpg", "rb") as img_file:
    files = {"image": img_file}
    response = requests.post(url, params=params, files=files)

result = response.json()
print(f"Prediction: {'Malignant' if result['prediction'] == 1 else 'Benign'}")
print(f"Confidence: {result['confidence']:.2f}")
```

## 📦 Project Structure

```
/
├── backend/
│   ├── models/              # Model implementations
│   │   ├── efficientnet.py  # EfficientNet model
│   │   ├── eva02.py         # EVA-02 model
│   │   └── __init__.py      # Model loading interface
│   ├── utils/               # Utility functions
│   │   └── preprocess.py    # Image preprocessing
│   ├── weights/             # Model weight files
│   │   ├── AUROC0.5180_Loss0.6361_epoch47.bin  # EfficientNet weights
│   │   └── AUROC0.5185_Loss0.3027_epoch39.bin  # EVA-02 weights
│   ├── app.py               # Application entry point
│   ├── config.py            # Configuration settings
│   ├── main.py              # Main FastAPI application
│   ├── requirements.txt     # Dependencies
│   └── Dockerfile           # Docker configuration
└── README.md                # This file
```

## 🚀 Deployment

This API can be easily deployed to cloud platforms like Render, Heroku, or Google Cloud Run.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 