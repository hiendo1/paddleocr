# OCR Deployment Requirements & Checklist

To successfully deploy the PaddleOCR service on your GPU server, please ensure the following requirements are met.

## 🟢 Server Hardware Requirements
- **GPU:** NVIDIA A100 / T4 / RTX Series (8GB+ VRAM recommended).
- **Driver:** NVIDIA Driver >= 525.xx (supporting CUDA 11.8).

## 🟢 Software Prerequisites
- **Docker:** Installed and running.
- **NVIDIA Container Toolkit:** Installed to allow Docker to access the GPU.
  - Verify with: `docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

## 🟢 Python Environment (if not using Docker)
- **Python:** 3.8 - 3.10.
- **NumPy:** MUST be version 1.x (e.g., `1.26.4`). **Do NOT install NumPy 2.0.**
- **CUDA/cuDNN:** CUDA 11.8 and cuDNN 8.x must be installed in the system.

## 🟢 Deployment Steps
1. **Clone the repository.**
2. **Build the Docker Image:**
   ```bash
   docker build -t ocr-paddle-gpu -f Dockerfile.gpu .
   ```
3. **Run the Container:**
   ```bash
   docker run -d --name ocr-service --gpus all -p 8000:8000 ocr-paddle-gpu
   ```

## 🟢 Verification
- Once running, check the health endpoint: `http://<server-ip>:8000/health` (if implemented) or just visit the base URL.
- Test the throughput using the provided `test_batch_parallel.py` from a client machine.
