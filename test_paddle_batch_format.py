import os
os.environ['FLAGS_use_mkldnn'] = '0'
import numpy as np
from paddleocr import PaddleOCR
import cv2

def test_batch_format():
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
    
    # Create two dummy images with some text-like features
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(img1, "Hello", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(img2, "World", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    images = [img1, img2]
    
    print("Running batch OCR...")
    all_outputs = ocr.ocr(images, cls=True)
    
    print(f"Number of list levels in all_outputs: {type(all_outputs)}")
    print(f"Length of all_outputs: {len(all_outputs)}")
    
    for i, img_output in enumerate(all_outputs):
        print(f"\nImage {i} output type: {type(img_output)}")
        if img_output:
            print(f"Image {i} output length: {len(img_output)}")
            print(f"Image {i} first element type: {type(img_output[0])}")
            print(f"Image {i} first element: {img_output[0]}")
            
            # Check the nesting
            if isinstance(img_output[0], list) and len(img_output[0]) > 0:
                print(f"Image {i} first element of first element type: {type(img_output[0][0])}")
                print(f"Image {i} first element of first element: {img_output[0][0]}")

if __name__ == "__main__":
    test_batch_format()
