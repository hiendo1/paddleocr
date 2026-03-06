import os
os.environ['FLAGS_use_mkldnn'] = '0'
from fastapi import FastAPI, UploadFile, File, HTTPException
from paddleocr import PaddleOCR
import numpy as np
import cv2
import time
from typing import List
from pydantic import BaseModel

app = FastAPI(title="PaddleOCR-Docker")

@app.get("/")
def home():
    return {"message": "PaddleOCR API is running. Visit /docs for API documentation."}

@app.get("/health")
def health():
    return {"status": "healthy", "gpu": paddle.is_compiled_with_cuda()}

import paddle

# Initialize PaddleOCR
# Auto-detect GPU availability
use_gpu = paddle.is_compiled_with_cuda()
print(f"Initializing PaddleOCR engine (use_gpu={use_gpu})...")
ocr = PaddleOCR(use_angle_cls=True, lang='latin', use_gpu=use_gpu)

class OCRResult(BaseModel):
    text: str
    confidence: float

class FileResult(BaseModel):
    filename: str
    results: List[OCRResult]

class BatchResponse(BaseModel):
    total_images: int
    total_latency_ms: float
    results: List[FileResult]

import asyncio
from concurrent.futures import ThreadPoolExecutor

# Semaphore to restrict GPU concurrency - only 1 batch at a time for performance stability
gpu_semaphore = asyncio.Semaphore(1)

@app.post("/ocr/predict", response_model=BatchResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Accepts multiple images and processes them as a true batch for GPU efficiency."""
    start_time = time.time()
    
    # 1. Read and decode all images in parallel (using threadpool for I/O and CPU)
    # Using async to not block the main event loop during file reads
    images = []
    filenames = []
    
    for file in files:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
                filenames.append(file.filename)
            else:
                print(f"Warning: Could not decode image {file.filename}")
        except Exception as e:
            print(f"Error reading {file.filename}: {e}")

    if not images:
        return BatchResponse(total_images=0, total_latency_ms=0, results=[])

    # 2. Perform Batch Inference on GPU
    # We wrap the blocking OCR call in a semaphore to prevent GPU contention
    async with gpu_semaphore:
        # Run the CPU-heavy/blocking OCR in a thread to keep FastAPI responsive
        loop = asyncio.get_event_loop()
        # PaddleOCR.ocr handles a list of images as a batch
        all_outputs = await loop.run_in_executor(None, lambda: ocr.ocr(images, cls=True))

    # 3. Parse results
    file_results = []
    for i, img_output in enumerate(all_outputs):
        extracted_data = []
        # img_output is a list of lines for this specific image
        if img_output:
            for line in img_output:
                extracted_data.append(OCRResult(
                    text=str(line[1][0]),
                    confidence=float(line[1][1])
                ))
        
        file_results.append(FileResult(
            filename=filenames[i],
            results=extracted_data
        ))
    
    latency = (time.time() - start_time) * 1000
    return BatchResponse(
        total_images=len(file_results),
        total_latency_ms=latency,
        results=file_results
    )
