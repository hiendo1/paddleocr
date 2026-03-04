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
def predict_batch(files: List[UploadFile] = File(...)):
    """Accepts multiple images and processes them as a batch.
    Uses sync 'def' to allow FastAPI to manage concurrency via threadpool,
    which is better for overlapping CPU-bound OCR and GPU inference.
    """
    start_time = time.time()
    file_results = []
    
    for file in files:
        try:
            # Synchronous read for 'def' endpoint
            contents = file.file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Process single image
                img_output = ocr.ocr(img, cls=True)
                
                extracted_data = []
                # PaddleOCR result is a list of lists (one per page)
                if img_output and img_output[0]:
                    for line in img_output[0]:
                        extracted_data.append(OCRResult(
                            text=str(line[1][0]),
                            confidence=float(line[1][1])
                        ))
                
                file_results.append(FileResult(
                    filename=file.filename,
                    results=extracted_data
                ))
            else:
                file_results.append(FileResult(filename=file.filename, results=[]))
                
        except Exception as img_err:
            print(f"Error processing image {file.filename}: {str(img_err)}")
            file_results.append(FileResult(filename=file.filename, results=[]))
    
    latency = (time.time() - start_time) * 1000
    return BatchResponse(
        total_images=len(file_results),
        total_latency_ms=latency,
        results=file_results
    )
