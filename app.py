import os
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import LTXPipeline, LTXImageToVideoPipeline
from diffusers.utils import export_to_video
import uuid
import logging
from PIL import Image
import io

# Configuration from Env
API_KEY_ENV = os.getenv("API_KEY", "RTn9iVMzEymdRc_cG60lIbbsERjGz0ZQzlBegTZvI")
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "1000"))
MODEL_ID = os.getenv("MODEL_ID", "Lightricks/LTX-Video")
MAX_FRAMES_DEFAULT = int(os.getenv("MAX_FRAMES", "121")) # LTX default is often higher, but let's stick to a reasonable default
OUTPUT_RESOLUTION_HEIGHT = int(os.getenv("OUTPUT_HEIGHT", "480"))
OUTPUT_RESOLUTION_WIDTH = int(os.getenv("OUTPUT_WIDTH", "704"))
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
LOGGING_ENABLED = os.getenv("LOGGING", "true").lower() == "true"

# Configure logging
logging.basicConfig(level=logging.INFO if LOGGING_ENABLED else logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(title="LTXVideo Generator")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
models = {}
OUTPUT_DIR = "generated_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_pipeline_class(task_type):
    if task_type == "t2v":
        return LTXPipeline
    elif task_type == "i2v":
        return LTXImageToVideoPipeline
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def load_model(task_type):
    pipeline_class = get_pipeline_class(task_type)
    model_key = pipeline_class.__name__
    
    if model_key not in models:
        logger.info(f"Loading {task_type} model: {MODEL_ID}")
        try:
            # Check for GPU
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            pipe = pipeline_class.from_pretrained(MODEL_ID, torch_dtype=dtype)
            
            if torch.cuda.is_available():
                pipe.enable_model_cpu_offload()
                # pipe.enable_vae_tiling() # Enable if OOM occurs
            
            models[model_key] = pipe
        except Exception as e:
            logger.error(f"Failed to load model {MODEL_ID}: {e}")
            raise e
    return models[model_key]

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "models_loaded": list(models.keys()),
        "config": {
            "model_id": MODEL_ID,
            "max_frames": MAX_FRAMES_DEFAULT,
            "resolution": f"{OUTPUT_RESOLUTION_WIDTH}x{OUTPUT_RESOLUTION_HEIGHT}"
        }
    }

@app.post("/txt2vid")
async def text_to_video(prompt: str = Form(...), num_frames: int = Form(MAX_FRAMES_DEFAULT)):
    try:
        logger.info(f"Generating video for prompt: {prompt}")
        pipe = load_model("t2v")
        
        # Inference
        # LTXVideo specific parameters can be tuned here
        video_frames = pipe(
            prompt=prompt,
            width=OUTPUT_RESOLUTION_WIDTH,
            height=OUTPUT_RESOLUTION_HEIGHT,
            num_frames=num_frames,
            num_inference_steps=50, # Default is usually 50
        ).frames[0]
        
        # Save video
        filename = f"{uuid.uuid4()}.mp4"
        filepath = os.path.join(OUTPUT_DIR, filename)
        export_to_video(video_frames, filepath, fps=24)
        
        return FileResponse(filepath, media_type="video/mp4", filename=filename)
    except Exception as e:
        logger.error(f"Error in txt2vid: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/img2vid")
async def image_to_video(image: UploadFile = File(...), prompt: str = Form(None), num_frames: int = Form(MAX_FRAMES_DEFAULT)):
    try:
        logger.info(f"Generating video from image")
        pipe = load_model("i2v")
        
        # Read and process image
        image_content = await image.read()
        pil_image = Image.open(io.BytesIO(image_content)).convert("RGB")
        
        # Resize to match output resolution to avoid mismatches
        pil_image = pil_image.resize((OUTPUT_RESOLUTION_WIDTH, OUTPUT_RESOLUTION_HEIGHT))
        
        # Inference
        video_frames = pipe(
            image=pil_image,
            prompt=prompt if prompt else "", # LTX I2V can take a prompt
            width=OUTPUT_RESOLUTION_WIDTH,
            height=OUTPUT_RESOLUTION_HEIGHT,
            num_frames=num_frames,
            num_inference_steps=50,
        ).frames[0]
        
        # Save video
        filename = f"{uuid.uuid4()}.mp4"
        filepath = os.path.join(OUTPUT_DIR, filename)
        export_to_video(video_frames, filepath, fps=24)
        
        return FileResponse(filepath, media_type="video/mp4", filename=filename)
    except Exception as e:
        logger.error(f"Error in img2vid: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Rate limiting & Blocking
BLOCK_THRESHOLD = 10
usage_counter = {}
violation_counter = {}
blocked_keys = set()

@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    api_key = request.headers.get("X-API-Key")
    path = request.url.path
    
    if path in ["/txt2vid", "/img2vid"]:
        if not api_key or api_key != API_KEY_ENV:
             logger.warning(f'{{"event": "invalid_key", "path": "{path}", "ip": "{request.client.host}"}}')
             return JSONResponse(status_code=401, content={"detail": "Invalid or Missing API Key"})
        
        if api_key in blocked_keys:
            return JSONResponse(status_code=403, content={"detail": "API Key blocked due to abuse"})

        current_usage = usage_counter.get(api_key, 0)
        if current_usage >= RATE_LIMIT:
            violations = violation_counter.get(api_key, 0) + 1
            violation_counter[api_key] = violations
            if violations >= BLOCK_THRESHOLD:
                blocked_keys.add(api_key)
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
        
        usage_counter[api_key] = current_usage + 1
        
    response = await call_next(request)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
