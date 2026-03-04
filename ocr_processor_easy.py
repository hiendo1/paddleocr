import os
import time
import json
import easyocr
import cv2

def run_ocr_benchmark(image_dir):
    # Initialize EasyOCR
    reader = easyocr.Reader(['en']) 
    
    results = {}
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    print(f"Starting benchmark on {len(image_files)} images using EasyOCR...")
    
    total_start_time = time.time()
    
    for idx, filename in enumerate(image_files):
        img_path = os.path.join(image_dir, filename)
        
        start_time = time.time()
        
        # Run OCR
        # EasyOCR returns a list of tuples: (bbox, text, confidence)
        result = reader.readtext(img_path)
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000 # ms
        
        # Format result
        extracted_text = []
        for (bbox, text, prob) in result:
            extracted_text.append({
                "text": text,
                "confidence": float(prob)
            })
        
        print(f"[{idx+1}/{len(image_files)}] {filename}: {latency:.2f}ms ({len(extracted_text)} lines found)")
        
        results[filename] = {
            "latency_ms": latency,
            "data": extracted_text
        }
        
    total_end_time = time.time()
    avg_latency = ((total_end_time - total_start_time) / len(image_files)) * 1000
    
    print("-" * 30)
    print(f"Total time: {total_end_time - total_start_time:.2f}s")
    print(f"Average latency: {avg_latency:.2f}ms")
    
    with open("benchmark_results_easyocr.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Results saved to benchmark_results_easyocr.json")

if __name__ == "__main__":
    TEST_CASE_DIR = r"c:\Users\Admin\Desktop\cleanup service\test_case"
    run_ocr_benchmark(TEST_CASE_DIR)
