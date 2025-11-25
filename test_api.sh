# Health Check
curl -X GET http://localhost:8000/health

# Text to Video
curl -X POST http://localhost:8000/txt2vid \
  -H "X-API-Key: test-key" \
  -F "prompt=A cyberpunk cat running in the rain" \
  -F "num_frames=16" \
  --output output_video.mp4

# Image to Video
curl -X POST http://localhost:8000/img2vid \
  -H "X-API-Key: test-key" \
  -F "image=@test_image.jpg" \
  --output output_video_from_image.mp4
