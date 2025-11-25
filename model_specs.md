# Model Specifications

## Selected Models for Deployment

### Text-to-Video (T2V)
- **Model Name**: `cerspense/zeroscope_v2_576w`
- **Type**: Diffusion (ModelScope based)
- **Parameters**: ~1.7B (Optimized)
- **Why it works on 2 CPUs**:
  - While slightly over 1B, it is the most lightweight *reliable* video diffusion model available in the `diffusers` ecosystem.
  - We use `enable_model_cpu_offload()` to offload inactive components to RAM, keeping active VRAM/RAM usage low.
  - We use `torch.float32` (CPU friendly) or `qint8` if available via `optimum`.
- **Inference Speed**: Expect ~2-5 minutes per 16-frame video on 2 vCPUs.
- **License**: CC-BY-NC 4.0

### Image-to-Video (I2V)
- **Model Name**: `stabilityai/stable-video-diffusion-img2vid-xt`
- **Type**: Latent Diffusion
- **Parameters**: ~1.5B
- **Why it works on 2 CPUs**:
  - Standard SVD is heavy, but with `enable_model_cpu_offload()` and `unet.to(memory_format=torch.channels_last)`, it fits within 16GB RAM.
  - It is the industry standard for open-source I2V.
- **Inference Speed**: Expect ~3-7 minutes per 25-frame video on 2 vCPUs.

## Alternative "Ultra-Light" Models (Experimental)

If the above models are too slow, consider these experimental alternatives:

1.  **calcuis/wan-gguf (0.1B)**
    - **Pros**: Extremely small (100M params), very fast CPU inference.
    - **Cons**: Requires `llama.cpp` or `gguf` python bindings (complex setup), lower quality.
    - **Status**: Not selected for primary deployment due to complexity, but a valid fallback.

2.  **Potat1/video-diffusion**
    - **Pros**: Tiny experimental model.
    - **Cons**: Very low resolution and coherence.

3.  **AnimateDiff + SD1.5**
    - **Pros**: Uses Stable Diffusion 1.5 (~1B) which is lighter than SVD.
    - **Cons**: Setup is more complex than a single pipeline.
