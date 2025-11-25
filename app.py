import os
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import DiffusionPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
import uuid
import logging
from PIL import Image
import io

# Configuration from Env
API_KEY_ENV = os.getenv("API_KEY", "mysecretapikey123")
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "1000"))
MODEL_NAME_T2V = os.getenv("MODEL_NAME", "cerspense/zeroscope_v2_576w") # User requested MODEL_NAME
MODEL_ID_I2V = os.getenv("MODEL_ID_I2V", "stabilityai/stable-video-diffusion-img2vid-xt")
MAX_FRAMES_DEFAULT = int(os.getenv("MAX_FRAMES", "32"))
OUTPUT_RESOLUTION = int(os.getenv("OUTPUT_RESOLUTION", "256"))
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
LOGGING_ENABLED = os.getenv("LOGGING", "true").lower() == "true"
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# Configure logging
logging.basicConfig(level=logging.INFO if LOGGING_ENABLED else logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(title="CPU Video Generator")

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

def load_t2v_model():
    if "t2v" not in models:
        logger.info(f"Loading T2V model: {MODEL_NAME_T2V}")
        try:
            pipe = DiffusionPipeline.from_pretrained(MODEL_NAME_T2V, torch_dtype=torch.float32)
            # CPU optimizations
            pipe.enable_model_cpu_offload()
            pipe.unet.to(memory_format=torch.channels_last)
            models["t2v"] = pipe
        except Exception as e:
            logger.error(f"Failed to load T2V model {MODEL_NAME_T2V}: {e}")
            raise e
    return models["t2v"]

def load_i2v_model():
    if "i2v" not in models:
        logger.info(f"Loading I2V model: {MODEL_ID_I2V}")
        try:
            pipe = StableVideoDiffusionPipeline.from_pretrained(MODEL_ID_I2V, torch_dtype=torch.float32, variant="fp16")
            # CPU optimizations
            pipe.enable_model_cpu_offload()
            pipe.unet.to(memory_format=torch.channels_last)
            models["i2v"] = pipe
        except Exception as e:
             logger.error(f"Failed to load I2V model {MODEL_ID_I2V}: {e}")
             raise e
    return models["i2v"]

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "models_loaded": list(models.keys()),
        "config": {
            "model_t2v": MODEL_NAME_T2V,
            "max_frames": MAX_FRAMES_DEFAULT,
            "resolution": OUTPUT_RESOLUTION
        }
    }

@app.post("/txt2vid")
async def text_to_video(prompt: str = Form(...), num_frames: int = Form(MAX_FRAMES_DEFAULT)):
    try:
        # Enforce max frames from env if user requests more? 
        # For now, let's just use the default if not provided, or clamp it?
        # The user requested MAX_FRAMES=32, let's treat it as a default or cap.
        frames_to_gen = min(num_frames, MAX_FRAMES_DEFAULT)
        
        logger.info(f"Generating video for prompt: {prompt} ({frames_to_gen} frames)")
        pipe = load_t2v_model()
        
        # Inference
        # Note: zeroscope might expect specific height/width. 
        # If OUTPUT_RESOLUTION is 256, we might need to set it.
        # However, zeroscope is trained at 576w. Resizing might degrade quality.
        # We will pass it if the pipeline accepts it, otherwise we resize post-gen?
        # Diffusion pipelines usually take height/width.
        video_frames = pipe(prompt, num_frames=frames_to_gen, height=OUTPUT_RESOLUTION, width=OUTPUT_RESOLUTION).frames[0]
        
        # Save video
        filename = f"{uuid.uuid4()}.mp4"
        filepath = os.path.join(OUTPUT_DIR, filename)
        export_to_video(video_frames, filepath)
        
        return FileResponse(filepath, media_type="video/mp4", filename=filename)
    except Exception as e:
        logger.error(f"Error in txt2vid: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/img2vid")
async def image_to_video(image: UploadFile = File(...), motion_bucket_id: int = Form(127)):
    try:
        logger.info(f"Generating video from image")
        pipe = load_i2v_model()
        
        # Read and process image
        image_content = await image.read()
        pil_image = Image.open(io.BytesIO(image_content)).convert("RGB")
        
        # Resize to user specified resolution
        # SVD usually likes specific buckets (1024x576), but we'll try the user's resolution
        pil_image = pil_image.resize((OUTPUT_RESOLUTION, OUTPUT_RESOLUTION)) 
        
        # Inference
        frames = pipe(pil_image, decode_chunk_size=8, motion_bucket_id=motion_bucket_id).frames[0]
        
        # Save video
        filename = f"{uuid.uuid4()}.mp4"
        filepath = os.path.join(OUTPUT_DIR, filename)
        export_to_video(frames, filepath)
        
        return FileResponse(filepath, media_type="video/mp4", filename=filename)
    except Exception as e:
        logger.error(f"Error in img2vid: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Rate limiting & Blocking
BLOCK_THRESHOLD = 10  # Block after 10 rate limit violations
usage_counter = {}
violation_counter = {}
blocked_keys = set()

@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    api_key = request.headers.get("X-API-Key")
    path = request.url.path
    
    if path in ["/txt2vid", "/img2vid"]:
        # Check against configured API_KEY
        if not api_key or api_key != API_KEY_ENV:
             logger.warning(f'{{"event": "invalid_key", "path": "{path}", "ip": "{request.client.host}"}}')
             # For strict mode:
             return JSONResponse(status_code=401, content={"detail": "Invalid or Missing API Key"})
        
        if api_key in blocked_keys:
            logger.warning(f'{{"event": "blocked_access_attempt", "key": "{api_key}"}}')
            return JSONResponse(status_code=403, content={"detail": "API Key blocked due to abuse"})

        current_usage = usage_counter.get(api_key, 0)
        if current_usage >= RATE_LIMIT:
            # Track violations
            violations = violation_counter.get(api_key, 0) + 1
            violation_counter[api_key] = violations
            
            logger.warning(f'{{"event": "rate_limit_exceeded", "key": "{api_key}", "violations": {violations}}}')
            
            if violations >= BLOCK_THRESHOLD:
                blocked_keys.add(api_key)
                logger.warning(f'{{"event": "key_blocked", "key": "{api_key}"}}')
                
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
        
        usage_counter[api_key] = current_usage + 1
        logger.info(f'{{"event": "request_allowed", "key": "{api_key}", "usage": {usage_counter[api_key]}}}')
        
    response = await call_next(request)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
