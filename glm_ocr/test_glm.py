import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
import time
import os

# Model path on Hugging Face
model_id = 'zai-org/GLM-OCR'

def run_glm_ocr(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    print(f"Loading model {model_id} (this may take a while)...")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️ WARNING: No GPU detected. Running on CPU will be extremely slow.")

    # Load tokenizer and model
    # Use AutoModel + trust_remote_code=True for custom architectures
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

    # Diagnostics
    print(f"Model class: {type(model)}")

    # Load and process image
    image = Image.open(image_path).convert('RGB')
    
    # Prompt for OCR
    query = "Read all text and structural information from this image accurately."

    print("Running inference...")
    start_time = time.time()
    
    with torch.no_grad():
        if hasattr(model, 'chat'):
            response, _ = model.chat(tokenizer, image, query, history=[])
        else:
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "image": image, "content": query}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
            ).to(device)
            
            generate_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            generate_ids = [cur_ids[len(inputs.input_ids[0]):] for cur_ids in generate_ids]
            response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
    
    latency = (time.time() - start_time) * 1000
    
    print(f"\n✅ Inference Complete ({latency:.2f}ms)")
    print("-" * 30)
    print("Output:")
    print(response)
    print("-" * 30)

if __name__ == "__main__":
    # Test with an image from the test_case directory
    test_image = r"c:\Users\Admin\Desktop\cleanup service\test_case\il_1588xN.6684182862_4wk7.jpg.webp"
    run_glm_ocr(test_image)
