# Hướng dẫn chạy PaddleOCR bằng Docker trên Windows

Để chạy được PaddleOCR ổn định mà không gặp lỗi Windows (AVX/MKLDNN), hãy làm theo các bước sau:

### 1. Tải và Cài đặt Docker
- Vào trang [Docker Desktop](https://www.docker.com/products/docker-desktop/).
- Tải bản cho Windows và cài đặt.
- **Quan trọng**: Nếu được hỏi, hãy chọn `Use WSL2 instead of Hyper-V`.
- Restart máy nếu Docker yêu cầu.

### 2. Khởi động Docker
- Mở ứng dụng `Docker Desktop`. Đợi đến khi icon hình con cá voi ở thanh cuộn phía dưới báo màu xanh (Engine running).

### 3. Build và Chạy Service
Mở Terminal (PowerShell hoặc CMD) tại thư mục dự án và chạy duy nhất lệnh sau:

```powershell
docker-compose up --build
```

- **Bước này làm gì?**: Docker sẽ tự động tải môi trường Linux, cài đặt PaddleOCR và các thư viện cần thiết. Lần đầu tiên chạy có thể mất vài phút.

### 4. Kiểm tra
Khi thấy dòng chữ `Uvicorn running on http://0.0.0.0:8000` trong terminal, service đã sẵn sàng.
Bạn có thể gọi API tại địa chỉ:
- **URL**: `http://localhost:8001/ocr/predict` (Sử dụng cổng 8001 để tránh trùng với app local đang chạy cổng 8000).

---

### Lưu ý:
- Toàn bộ code và thư viện của PaddleOCR bây giờ nằm trong "chiếc hộp" Docker, không làm ảnh hưởng đến máy thật của bạn.
- Hiệu năng xử lý ảnh trên Docker Linux sẽ ổn định và chính xác hơn bản Windows "hack" rất nhiều.
