# Skin Cancer Detection API

A FastAPI backend for skin cancer detection using pre-trained deep learning models.

## Features

- **Multiple Models**: Supports two different models for skin cancer detection:
  - EfficientNet-B0 (224x224 input size)
  - EVA-02 (336x336 input size)
- **REST API**: Simple POST endpoint for making predictions
- **Health Check**: GET endpoint for monitoring service health
- **CORS Support**: Cross-origin request support for frontend integration

## Setup

### Prerequisites

- Python 3.10+
- PyTorch
- FastAPI
- Uvicorn

### Installation

1. Clone the repository:
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
   - `AUROC0.5180_Loss0.6361_epoch47.bin` (EfficientNet-B0)
   - `AUROC0.5185_Loss0.3027_epoch39.bin` (EVA-02)

   **Note**: These model files are not included in this repository due to size constraints.

### Running the API

```bash
cd backend
uvicorn main:app --reload
```

The API will be available at http://localhost:8000.

## API Documentation

Once the API is running, you can access the auto-generated Swagger documentation at:

http://localhost:8000/docs

### Endpoints

#### GET /health

Returns the health status of the API.

**Response:**
```json
{
  "status": "ok"
}
```

#### GET /model-check

Checks if the model files are available.

**Response:**
```json
{
  "status": "ok",
  "efficient_model": true,
  "eva02_model": true,
  "model_dir": "/path/to/lib/model"
}
```

#### POST /predict

Accepts a dermoscopic image and returns a prediction of skin cancer.

**Parameters:**
- `image`: Multipart file upload (JPEG/PNG)
- `model_type`: Query parameter with value either `efficient_b` or `eva02`

**Response:**
```json
{
  "model": "eva02",
  "prediction": 1,
  "confidence": 0.84
}
```

Where:
- `model`: The model used for prediction
- `prediction`: Binary prediction (0 = benign, 1 = malignant)
- `confidence`: Confidence score (0-1)

## Docker Deployment

You can also run this API as a Docker container:

```bash
# Build the Docker image
docker build -t skin-cancer-api .

# Create a volume to store model files
docker volume create model-weights

# Copy model files to the volume - adjust the paths as needed
docker run --rm -v model-weights:/app/lib/model -v /path/to/your/models:/models alpine cp /models/* /app/lib/model/

# Run the container with the volume mounted
docker run -p 8000:8000 -v model-weights:/app/lib/model skin-cancer-api
```

## Example Usage

Using cURL:

```bash
curl -X POST "http://localhost:8000/predict?model_type=eva02" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/your/skin-image.jpg"
```

Using Python requests:

```python
import requests

url = "http://localhost:8000/predict"
params = {"model_type": "efficient_b"}
files = {"image": open("skin-image.jpg", "rb")}

response = requests.post(url, params=params, files=files)
print(response.json())
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 