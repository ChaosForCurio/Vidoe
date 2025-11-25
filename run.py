import uvicorn
import os
import sys

# Explicitly import torch first to see if it works here
print("Importing torch...")
try:
    import torch
    print(f"Torch imported successfully: {torch.__version__}")
except Exception as e:
    print(f"Failed to import torch: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("Starting Uvicorn...")
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=False)
