import requests
import os
import time

# API URL (Remote Google Colab via ngrok)
URL = "http://45.77.3.200:8000/ocr/predict"

# Directory containing test images
IMAGE_DIR = r"c:\Users\Admin\Desktop\cleanup service\test_case"

def test_batch_ocr():
    # 1. Collect some image files
    files = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    
    image_paths = [
        os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
        if f.lower().endswith(supported_extensions)
    ] # Removed [:5] limit to test all images
    
    if not image_paths:
        print("No images found in test_case directory!")
        return

    # 2. Prepare multipart files payload
    # Format: [('files', (filename, open_file, content_type)), ...]
    upload_files = []
    for path in image_paths:
        filename = os.path.basename(path)
        upload_files.append(
            ('files', (filename, open(path, 'rb'), 'image/jpeg'))
        )

    print(f"Sending batch of {len(upload_files)} images to {URL}...")
    
    # 3. Send request
    start_time = time.time()
    try:
        response = requests.post(URL, files=upload_files)
        response.raise_for_status()
        data = response.json()
        
        # 4. Show results
        total_latency_ms = data.get("total_latency_ms", 0)
        total_images = data.get("total_images", 0)
        throughput = total_images / (total_latency_ms / 1000) if total_latency_ms > 0 else 0
        
        print(f"\n✅ Batch Processing Complete!")
        print(f"Total Images: {total_images}")
        print(f"Total API Latency: {total_latency_ms:.2f}ms")
        print(f"Avg Speed: {total_latency_ms / total_images:.2f} ms/image")
        print(f"🚀 Throughput: {throughput:.2f} images/second")
        
        print("\nDetail Results:")
        for res in data.get("results", []):
            text_preview = " | ".join([item['text'] for item in res['results'][:3]])
            print(f"- {res['filename']}: {len(res['results'])} lines found. Preview: [{text_preview}...]")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Close all files
        for _, file_tuple in upload_files:
            file_tuple[1].close()

if __name__ == "__main__":
    test_batch_ocr()
