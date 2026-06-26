"""FastAPI server exposing LamSonOCR predictions."""
import os
import sys
from pathlib import Path
from PIL import Image
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from inference.predict import OCRPredictor
from config.settings import Settings

app = FastAPI(
    title="LamSonOCR API",
    description="API for Japanese/English/Numeric OCR using CRNN + CTC model",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor lazily
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        settings = Settings()
        checkpoint_path = settings.project_root / "checkpoints" / "best_model.pt"
        if not checkpoint_path.exists():
            # Try checking the parent checkpoints directory as a fallback
            fallback_path = settings.project_root.parent / "checkpoints" / "best_model.pt"
            if fallback_path.exists():
                checkpoint_path = fallback_path
            else:
                raise RuntimeError(
                    f"Model checkpoint not found at {checkpoint_path}. "
                    "Please train the model or provide a valid checkpoint."
                )
        
        device = "cuda" if os.environ.get("USE_CUDA", "").lower() == "true" or settings.device == "cuda" else "cpu"
        # If running on macOS, we can also support MPS
        if device == "cpu" and settings.device == "mps":
            device = "mps"
            
        print(f"Loading OCR model from {checkpoint_path} onto {device}...")
        predictor = OCRPredictor(
            checkpoint_path=checkpoint_path,
            device=device,
            img_height=settings.img_height,
            img_max_width=settings.img_max_width
        )
    return predictor


@app.on_event("startup")
async def startup_event():
    try:
        get_predictor()
        print("OCR API Server successfully initialized model.")
    except Exception as e:
        print(f"Warning during startup initialization: {e}")


@app.post("/predict")
async def predict_ocr(file: UploadFile = File(...)):
    """Perform OCR on the uploaded image.
    
    Accepts any common image format (PNG, JPEG, etc.).
    Returns:
        {"text": recognized_text, "confidence": confidence}
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        pred = get_predictor().predict(image)
        return pred
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Prediction failed: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    try:
        get_predictor()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "model_loaded": False, "error": str(e)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run("api:app", host=host, port=port, reload=False)
