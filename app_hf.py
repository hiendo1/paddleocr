import os
import gradio as gr
from paddleocr import PaddleOCR
import numpy as np
import cv2
import time

# Hugging Face ZeroGPU support
# This only works on HF Spaces with the Gradio SDK
try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

# Initialize PaddleOCR
# In Space, we'll initialize it here. The GPU will be allocated per-call.
print("Initializing PaddleOCR for Hugging Face...")
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

def ocr_process_logic(image):
    if image is None:
        return "No image uploaded"
    
    # Convert Gradio image (numpy) if needed (Gradio usually provides RGB numpy)
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    start_time = time.time()
    
    # Run Inference
    # Note: Hugging Face ZeroGPU will manage the GPU allocation here
    result = ocr.ocr(img, cls=True)
    
    latency = (time.time() - start_time) * 1000
    
    # Format text output
    extracted_text = []
    if result and result[0]:
        for line in result[0]:
            extracted_text.append(f"{line[1][0]} (Conf: {line[1][1]:.2f})")
    
    output_str = "\n".join(extracted_text)
    return f"Processing Time: {latency:.2f}ms\n\nExtracted Text:\n{output_str}"

# Define the GPU-decorated function
if has_spaces:
    @spaces.GPU
    def predict(image):
        return ocr_process_logic(image)
else:
    def predict(image):
        return ocr_process_logic(image)

# Build Gradio UI
with gr.Blocks(title="PaddleOCR on ZeroGPU") as demo:
    gr.Markdown("# 🚀 High-Performance OCR (PaddleOCR + ZeroGPU)")
    gr.Markdown("Upload an image to extract text using PaddleOCR on an NVIDIA A100 (ZeroGPU).")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Input Image")
            btn = gr.Button("Extract Text")
        with gr.Column():
            output_text = gr.Textbox(label="OCR Result", lines=15)
            
    btn.click(fn=predict, inputs=input_img, outputs=output_text)

# Launch
if __name__ == "__main__":
    demo.launch()
