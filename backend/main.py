import time
import logging
import os
import sys

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "..")))

from fastapi import FastAPI, UploadFile, HTTPException, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch

# Import model modules (use absolute imports for better compatibility)
from backend.models import predict
from backend.utils.preprocess import preprocess_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Skin Cancer Detection API",
    description="API for skin cancer detection using EfficientNet-B0 and EVA-02 models",
    version="1.0.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define response model
class PredictionResponse(BaseModel):
    model: str
    prediction: int
    confidence: float

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint that returns a status of 'ok'
    """
    return {"status": "ok"}

@app.get("/model-check")
async def model_check():
    """
    Check if the model files are available
    """
    models_dir = os.path.abspath(os.path.join(current_dir, "../lib/model"))
    efficient_path = os.path.join(models_dir, "AUROC0.5180_Loss0.6361_epoch47.bin")
    eva02_path = os.path.join(models_dir, "AUROC0.5185_Loss0.3027_epoch39.bin")
    
    if not os.path.exists(models_dir):
        os.makedirs(os.path.dirname(models_dir), exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        return JSONResponse(
            status_code=503,
            content={
                "status": "missing",
                "message": f"Model directory not found. Please create directory: {models_dir}"
            }
        )
    
    return {
        "status": "ok" if (os.path.exists(efficient_path) and os.path.exists(eva02_path)) else "missing",
        "efficient_model": os.path.exists(efficient_path),
        "eva02_model": os.path.exists(eva02_path),
        "model_dir": models_dir
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    image: UploadFile = File(...),
    model_type: str = Query(..., description="Model type: 'efficient_b' or 'eva02'")
):
    """
    Process an image for skin cancer detection.
    
    Args:
        image: Image file uploaded by the user
        model_type: Type of model to use for prediction
        
    Returns:
        PredictionResponse with model type, prediction (0 or 1), and confidence score
    """
    start_time = time.time()
    logger.info(f"Received prediction request with model_type={model_type}")
    
    try:
        # Validate model type
        if model_type not in ["efficient_b", "eva02"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid model_type. Must be 'efficient_b' or 'eva02'."
            )
        
        # Check if models exist
        models_dir = os.path.abspath(os.path.join(current_dir, "../lib/model"))
        model_file = "AUROC0.5180_Loss0.6361_epoch47.bin" if model_type == "efficient_b" else "AUROC0.5185_Loss0.3027_epoch39.bin"
        model_path = os.path.join(models_dir, model_file)
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise HTTPException(
                status_code=503,
                detail=f"Model file not found. Please place '{model_file}' in the lib/model directory."
            )
        
        # Read image file content
        image_bytes = await image.read()
        
        # Preprocess image
        try:
            image_tensor = preprocess_image(image_bytes, model_type)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error preprocessing image: {str(e)}"
            )
        
        # Run prediction
        try:
            prediction_value, confidence_score = predict(model_type, image_tensor)
        except Exception as e:
            logger.error(f"Error running prediction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error running prediction: {str(e)}"
            )
        
        # Measure end time
        end_time = time.time()
        
        # Log response time
        process_time_ms = (end_time - start_time) * 1000
        logger.info(f"Response time: {process_time_ms:.2f}ms")
        
        # Return prediction response
        return PredictionResponse(
            model=model_type,
            prediction=prediction_value,
            confidence=float(confidence_score)  # Convert to float for JSON serialization
        )
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 