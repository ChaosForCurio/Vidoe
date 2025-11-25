# API Documentation

## Base URL
`https://<your-northflank-service-url>`

## Authentication
All endpoints (except `/health`) require an API Key.
- **Header**: `X-API-Key: <your-api-key>`

## Endpoints

### 1. Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Auth**: None
- **Response**:
  ```json
  {
    "status": "healthy",
    "models_loaded": ["t2v", "i2v"]
  }
  ```

### 2. Text to Video
- **URL**: `/txt2vid`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data` or `application/x-www-form-urlencoded`
- **Parameters**:
  - `prompt` (string, required): Description of the video.
  - `num_frames` (int, optional): Number of frames (default: 16).
- **Response**: Returns the generated `.mp4` video file.
- **Error Responses**:
  - `429`: Rate limit exceeded.
  - `403`: Key blocked.
  - `500`: Generation failed.

### 3. Image to Video
- **URL**: `/img2vid`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `image` (file, required): Input image (JPEG/PNG).
  - `motion_bucket_id` (int, optional): Motion intensity (default: 127).
- **Response**: Returns the generated `.mp4` video file.

## Rate Limits
- **Quota**: 1000 requests per month per key.
- **Blocking**: Keys are auto-blocked after 10 rate limit violations.
