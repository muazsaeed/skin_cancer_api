"""
Simple entry point to run the FastAPI app directly.
This file makes it easier to run the application on any WSGI server.
"""
import os
import sys

# Add the root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, root_dir)

# Import the FastAPI app
from backend.main import app

# This allows running the app with any WSGI server like Gunicorn or Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 