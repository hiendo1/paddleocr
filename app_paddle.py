import os
# Disable MKLDNN (might not be needed on Linux but safe to have)
os.environ['FLAGS_use_mkldnn'] = '0' 
from fastapi import FastAPI, UploadFile, File, HTTPException
from paddleocr import PaddleOCR
import numpy as np
import cv2
import time
from typing import List
from pydantic import BaseModel

app = FastAPI(title="High-Performance OCR Service (PaddleOCR-Docker)")

import paddle

# Initialize PaddleOCR
# Auto-detect GPU availability
use_gpu = paddle.is_compiled_with_cuda()
print(f"Initializing PaddleOCR engine (use_gpu={use_gpu})...")
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)

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

@app.post("/ocr/predict", response_model=BatchResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Accepts multiple images and processes them as a batch."""
    start_time = time.time()
    
    images = []
    filenames = []
    
    # 1. Read and decode all images
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
            filenames.append(file.filename)
    
    if not images:
        raise HTTPException(status_code=400, detail="No valid images uploaded")

    try:
        # 2. Run Inference
        file_results = []
        
        # Note: Older PaddleOCR versions might have issues with direct list input 
        # for some model configurations. We'll iterate to ensure stability first.
        # If GPU is available, we would use a more optimized batching approach.
        for i, img in enumerate(images):
            # Run inference for single image
            img_output = ocr.ocr(img, cls=True)
            
            extracted_data = []
            if img_output and img_output[0]:
                for line in img_output[0]:
                    extracted_data.append(OCRResult(
                        text=line[1][0],
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
        
    except Exception as e:
        print(f"ERROR during OCR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
