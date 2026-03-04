import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# API URL (Remote Google Colab via ngrok)
URL = "http://45.77.3.200:8000/ocr/predict"

# Directory containing test images
IMAGE_DIR = r"c:\Users\Admin\Desktop\cleanup service\test_case"

from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Parallel settings
MAX_WORKERS = 10  # Reduced slightly for stability with ngrok free tier
BATCH_SIZE = 8     

def send_request(batch_files):
    """Sends a single batch of images to the OCR API with retries."""
    upload_files = []
    opened_files = []
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        for path in batch_files:
            filename = os.path.basename(path)
            f = open(path, 'rb')
            opened_files.append(f)
            upload_files.append(('files', (filename, f, 'image/jpeg')))
        
        start = time.time()
        # Use session with retries
        response = session.post(URL, files=upload_files, timeout=60)
        response.raise_for_status()
        data = response.json()
        latency = (time.time() - start) * 1000
        return data, latency
    except Exception as e:
        print(f"❌ Worker Error: {e}")
        return None, 0
    finally:
        for f in opened_files:
            f.close()
        session.close()

def optimize_batch_ocr():
    # 1. Collect all images
    supported_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_paths = [
        os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
        if f.lower().endswith(supported_extensions)
    ]
    
    if not image_paths:
        print("No images found in test_case directory!")
        return

    print(f"🚀 Starting Parallel OCR Optimization")
    print(f"Total Images: {len(image_paths)}")
    print(f"Concurrency: {MAX_WORKERS} workers, {BATCH_SIZE} images/request")
    print("-" * 40)

    # 2. Divide images into batches
    batches = [image_paths[i:i + BATCH_SIZE] for i in range(0, len(image_paths), BATCH_SIZE)]
    
    start_total = time.time()
    all_results = []
    
    # 3. Process batches in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {executor.submit(send_request, batch): batch for batch in batches}
        
        for future in as_completed(future_to_batch):
            res_data, api_latency = future.result()
            if res_data:
                all_results.append(res_data)
                num_imgs = res_data.get("total_images", 0)
                print(f"✅ Received batch: {num_imgs} images in {api_latency:.0f}ms")

    total_time = (time.time() - start_total) * 1000
    total_images_processed = sum([r.get("total_images", 0) for r in all_results])
    
    # 4. Final Benchmark
    if total_images_processed > 0:
        throughput = total_images_processed / (total_time / 1000)
        avg_speed = total_time / total_images_processed
        
        print("-" * 40)
        print(f"📊 OPTIMIZED RESULTS (A100 + Parallel):")
        print(f"Total Time Taken: {total_time/1000:.2f} s")
        print(f"Total Images: {total_images_processed}")
        print(f"Avg Speed: {avg_speed:.2f} ms/image")
        print(f"🚀 Optimized Throughput: {throughput:.2f} images/second")
        print("-" * 40)

if __name__ == "__main__":
    optimize_batch_ocr()
