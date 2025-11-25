import requests
import sys

API_URL = "http://localhost:8000" # Change to your Northflank URL
API_KEY = "test-key"

def test_health():
    print("Testing /health...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(response.json())
    except Exception as e:
        print(f"Failed: {e}")

def test_txt2vid():
    print("\nTesting /txt2vid...")
    prompt = "A panda eating bamboo in a futuristic city"
    headers = {"X-API-Key": API_KEY}
    data = {"prompt": prompt, "num_frames": 16}
    
    try:
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
    files = {'image': open('test_image.jpg', 'rb')} # Ensure test_image.jpg exists
    
    try:
        response = requests.post(f"{API_URL}/img2vid", headers=headers, files=files, stream=True)
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
