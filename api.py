from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import io
from PIL import Image
import numpy as np
from predict import load_model, preprocess_image, predict
import time

app = FastAPI(
    title="Brain Tumor Detection API",
    description="API for detecting and classifying brain tumors from MRI scans",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        app.state.model = load_model('models/best_model.pth', device)
        app.state.device = device
        app.state.startup_time = time.time()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "uptime": time.time() - app.state.startup_time,
        "model_loaded": hasattr(app.state, "model")
    }

@app.get("/version")
async def version():
    """Get API version information"""
    return {
        "version": "1.0.0",
        "model_version": "ResNet50-v1",
        "supported_formats": ["jpg", "jpeg", "png"]
    }

def validate_image(image: Image.Image) -> bool:
    """Validate image dimensions and format"""
    min_size = 100
    max_size = 4096
    width, height = image.size
    return (min_size <= width <= max_size and 
            min_size <= height <= max_size)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict brain tumor from MRI scan
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        # Read image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )

        # Validate image
        if not validate_image(image):
            raise HTTPException(
                status_code=400,
                detail="Image dimensions must be between 100x100 and 4096x4096"
            )
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Preprocess image
        try:
            image_tensor = preprocess_image(image_array)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Failed to preprocess image"
            )
        
        # Make prediction
        try:
            prediction, confidence, class_probs = predict(
                app.state.model, 
                image_tensor, 
                app.state.device
            )
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Prediction failed"
            )
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "processing_time": time.time() - app.state.startup_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )