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
git clone https://github.com/yourusername/skin-cancer-detection-api.git
cd skin-cancer-detection-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create directories for model weights:
```bash
mkdir -p lib/model
```

4. Add model weight files to the `lib/model` directory:
   - `AUROC0.5180_Loss0.6361_epoch47.bin` (EfficientNet-B0 weights)
   - `AUROC0.5185_Loss0.3027_epoch39.bin` (EVA-02 weights)

   **Note**: These model files are not included in this repository due to size constraints.

5. Run the API:
```bash
uvicorn backend.main:app --reload
```

### Option 2: Docker Deployment

```bash
# Build the Docker image
docker build -t skin-cancer-api .

# Create a volume to store model files
docker volume create model-weights

# Copy model files to the volume (adjust the paths as needed)
docker run --rm -v model-weights:/app/lib/model -v /path/to/your/models:/models alpine cp /models/* /app/lib/model/

# Run the container with the volume mounted
docker run -p 8000:8000 -v model-weights:/app/lib/model skin-cancer-api
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
backend/
├── models/              # Model implementations
│   ├── efficientnet.py  # EfficientNet model
│   ├── eva02.py         # EVA-02 model
│   └── __init__.py      # Model loading interface
├── utils/               # Utility functions
│   └── preprocess.py    # Image preprocessing
├── app.py               # Application entry point
├── config.py            # Configuration settings
├── main.py              # Main FastAPI application
├── requirements.txt     # Dependencies
└── Dockerfile           # Docker configuration
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 