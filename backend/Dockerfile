FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for model weights
RUN mkdir -p lib/model

# Add a note about model files
RUN echo "Please download model files and place them in lib/model/ directory:" > lib/model/README.txt && \
    echo "- AUROC0.5180_Loss0.6361_epoch47.bin (EfficientNet-B0 weights)" >> lib/model/README.txt && \
    echo "- AUROC0.5185_Loss0.3027_epoch39.bin (EVA-02 weights)" >> lib/model/README.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"] 