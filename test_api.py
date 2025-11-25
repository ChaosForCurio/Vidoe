import requests
import sys
import os

API_URL = "http://localhost:8000" # Change to your Northflank URL
API_KEY = "RTn9iVMzEymdRc_cG60lIbbsERjGz0ZQzlBegTZvI"

def test_health():
    print("Testing /health...")
    try:
        response = requests.get(f"{API_URL}/health")
        data = response.json()
        print(data)
        if "Lightricks/LTX-Video" in data.get("config", {}).get("model_id", ""):
            print("Verified: LTXVideo model configured.")
        else:
            print("Warning: Model ID does not match LTXVideo.")
    except Exception as e:
        print(f"Failed: {e}")

def test_txt2vid():
    print("\nTesting /txt2vid...")
    prompt = "A cinematic shot of a cyberpunk city in rain"
    headers = {"X-API-Key": API_KEY}
    data = {"prompt": prompt, "num_frames": 16} # Keep frames low for testing
    
    try:
        print("Sending request... (This may take time for model download/inference)")
        response = requests.post(f"{API_URL}/txt2vid", headers=headers, data=data, stream=True)
        if response.status_code == 200:
            with open("output_txt2vid.mp4", "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Success! Saved to output_txt2vid.mp4")
        else:
            print(f"Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Failed: {e}")

def test_img2vid():
    print("\nTesting /img2vid...")
    headers = {"X-API-Key": API_KEY}
    # Create a dummy image if it doesn't exist
    if not os.path.exists('test_image.jpg'):
        from PIL import Image
        img = Image.new('RGB', (704, 480), color = 'blue')
        img.save('test_image.jpg')
        
    files = {'image': open('test_image.jpg', 'rb')} 
    data = {"prompt": "Camera pans right", "num_frames": 16}
    
    try:
        print("Sending request... (This may take time for model download/inference)")
        response = requests.post(f"{API_URL}/img2vid", headers=headers, files=files, data=data, stream=True)
        if response.status_code == 200:
            with open("output_img2vid.mp4", "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Success! Saved to output_img2vid.mp4")
        else:
            print(f"Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        API_URL = sys.argv[1]
    
    test_health()
    # Uncomment to run generation tests (warning: slow on CPU)
    # test_txt2vid()
    # test_img2vid()
