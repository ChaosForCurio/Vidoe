import os
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from diffusers import DiffusionPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
import uuid
import logging
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CPU Video Generator")

# Global model cache
models = {}

# Environment variables
MODEL_ID_T2V = os.getenv("MODEL_ID_T2V", "cerspense/zeroscope_v2_576w")
MODEL_ID_I2V = os.getenv("MODEL_ID_I2V", "stabilityai/stable-video-diffusion-img2vid-xt")
OUTPUT_DIR = "generated_videos"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_t2v_model():
    if "t2v" not in models:
        logger.info(f"Loading T2V model: {MODEL_ID_T2V}")
        pipe = DiffusionPipeline.from_pretrained(MODEL_ID_T2V, torch_dtype=torch.float32)
        # CPU optimizations
        pipe.enable_model_cpu_offload()
        pipe.unet.to(memory_format=torch.channels_last)
        models["t2v"] = pipe
    return models["t2v"]

def load_i2v_model():
    if "i2v" not in models:
        logger.info(f"Loading I2V model: {MODEL_ID_I2V}")
        pipe = StableVideoDiffusionPipeline.from_pretrained(MODEL_ID_I2V, torch_dtype=torch.float32, variant="fp16")
        # CPU optimizations
        pipe.enable_model_cpu_offload()
        pipe.unet.to(memory_format=torch.channels_last)
        models["i2v"] = pipe
    return models["i2v"]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": list(models.keys())}

@app.post("/txt2vid")
async def text_to_video(prompt: str = Form(...), num_frames: int = Form(16)):
    try:
        logger.info(f"Generating video for prompt: {prompt}")
        pipe = load_t2v_model()
        
        # Inference
        video_frames = pipe(prompt, num_frames=num_frames).frames[0]
        
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
        pil_image = pil_image.resize((1024, 576)) # Resize to standard SVD resolution
        
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
USAGE_LIMIT = 1000
BLOCK_THRESHOLD = 10  # Block after 10 rate limit violations
usage_counter = {}
violation_counter = {}
blocked_keys = set()

@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    api_key = request.headers.get("X-API-Key")
    path = request.url.path
    
    if path in ["/txt2vid", "/img2vid"]:
        if not api_key:
             # Log warning in JSON
             logger.warning(f'{{"event": "missing_key", "path": "{path}", "ip": "{request.client.host}"}}')
             # return JSONResponse(status_code=401, content={"detail": "Missing API Key"})
             pass
        
        if api_key in blocked_keys:
            logger.warning(f'{{"event": "blocked_access_attempt", "key": "{api_key}"}}')
            return JSONResponse(status_code=403, content={"detail": "API Key blocked due to abuse"})

        current_usage = usage_counter.get(api_key, 0)
        if current_usage >= USAGE_LIMIT:
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
    # Configure JSON logging for uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
