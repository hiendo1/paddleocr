from fastapi import FastAPI, UploadFile, File, HTTPException
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import numpy as np
import cv2
import time
from typing import List
from pydantic import BaseModel

app = FastAPI(title="High-Performance OCR Service (DocTR)")

# Initialize DocTR (reuse instance for performance)
# docTR will use GPU if available, otherwise CPU
print("Initializing DocTR predictor...")
predictor = ocr_predictor(pretrained=True)

class OCRResult(BaseModel):
    text: str
    confidence: float

class BatchResponse(BaseModel):
    filename: str
    latency_ms: float
    results: List[OCRResult]

@app.post("/ocr/predict", response_model=BatchResponse)
async def predict_ocr(file: UploadFile = File(...)):
    start_time = time.time()
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        # DocTR expects list of images
        # We convert the numpy array (BGR) to RGB as DocTR/PIL expects
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run inference
        result = predictor([img_rgb])
        
        extracted_data = []
        # DocTR structure: pages -> blocks -> lines -> words
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join([word.value for word in line.words])
                    # Average confidence for the line
                    conf = sum([word.confidence for word in line.words]) / len(line.words) if line.words else 0
                    
                    extracted_data.append(OCRResult(
                        text=line_text,
                        confidence=float(conf)
                    ))
        
        latency = (time.time() - start_time) * 1000
        
        return BatchResponse(
            filename=file.filename,
            latency_ms=latency,
            results=extracted_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
