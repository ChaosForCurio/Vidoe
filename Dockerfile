FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for ffmpeg and git (for cloning models if needed)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install python dependencies
# Using CPU-only PyTorch to save image size and ensure CPU compatibility
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose the port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_ID_T2V="cerspense/zeroscope_v2_576w"
ENV MODEL_ID_I2V="stabilityai/stable-video-diffusion-img2vid-xt"

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
