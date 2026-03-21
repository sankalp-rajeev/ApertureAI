import requests
import sys
from pathlib import Path

def test_predict(image_path: str):
    url = "http://localhost:8000/api/predict"
    path = Path(image_path)
    
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
        
    print(f"Sending {path} to {url}...")
    with open(path, "rb") as f:
        files = {"file": f}
        try:
            response = requests.post(url, files=files)
            response.raise_for_status()
            data = response.json()
            
            print("\nResponse received:")
            print(f"Prediction: {data.get('prediction')}")
            print(f"Confidence: {data.get('confidence')}")
            
            has_gradcam = "gradcam_overlay" in data and bool(data["gradcam_overlay"])
            print(f"Has Grad-CAM overlay: {has_gradcam}")
            
            if has_gradcam:
                overlay_len = len(data["gradcam_overlay"])
                print(f"Overlay base64 length: {overlay_len}")
                
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to API. Is the server running with `uvicorn app.main:app`?")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # You can change this to any valid image path in your dataset
    default_image = "data/raw/images/0000.png"
    target_image = sys.argv[1] if len(sys.argv) > 1 else default_image
    test_predict(target_image)
