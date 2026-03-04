# Google Colab Setup for High-Performance OCR (EasyOCR GPU)

To achieve high speed and stable installation, follow these steps on [Google Colab](https://colab.research.google.com/):

### 1. Enable GPU
- Go to `Runtime` -> `Change runtime type`.
- Select `T4 GPU`.

### 2. Install Dependencies
Run this in the first cell:
```python
!pip install easyocr fastapi uvicorn nest-asyncio requests 
```

### 3. Upload Test Images
- Zip your `test_case` folder.
- Upload `test_case.zip` to Colab and unzip it:
```python
!unzip test_case.zip
```

### 4. Run the OCR Script (GPU Enabled)
Create a new cell and paste the following code to benchmark:

```python
import os
import time
import json
import easyocr
import cv2

# Initialize EasyOCR with GPU
# EasyOCR automatically detects and uses CUDA if available
reader = easyocr.Reader(['en'], gpu=True)

image_dir = '/content/test_case'  # Path to unzipped folder
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

print(f"Starting benchmark on {len(image_files)} images using EasyOCR + GPU...")
total_start_time = time.time()

for idx, filename in enumerate(image_files):
    img_path = os.path.join(image_dir, filename)
    start_time = time.time()
    
    # Run OCR
    result = reader.readtext(img_path)
    
    latency = (time.time() - start_time) * 1000
    print(f"[{idx+1}/{len(image_files)}] {filename}: {latency:.2f}ms ({len(result)} lines found)")

total_end_time = time.time()
avg_latency = ((total_end_time - total_start_time) / len(image_files)) * 1000
print("-" * 30)
print(f"Average latency on T4 GPU (EasyOCR): {avg_latency:.2f}ms")
```


### 5. Running as an API
If you want to run the FastAPI server inside Colab, use `nest_asyncio`:
```python
import nest_asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File
import threading

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Same logic as app.py
    pass

# Run in background thread
nest_asyncio.apply()
thread = threading.Thread(target=uvicorn.run, kwargs={'app': app, 'host': '0.0.0.0', 'port': 8000})
thread.start()
```
