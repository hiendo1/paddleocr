import requests
import os
import time
import json
import sys
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Đảm bảo stdout hỗ trợ utf-8 để in tiếng Việt trên console Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- CẤU HÌNH ---
URL = "http://45.77.3.200:8000/ocr/predict"
IMAGE_DIR = r"c:\Users\Admin\Desktop\cleanup service\test_case"
OUTPUT_JSON = "ocr_results.json"

# Cấu hình song song (Dựa trên cấu hình đã tối ưu cho A16)
MAX_WORKERS = 4
BATCH_SIZE = 32

# ----------------

def send_request(batch_files):
    """Gửi một batch ảnh đến OCR API và tính thời gian."""
    upload_files = []
    opened_files = []
    
    # Cấu hình chiến lược retry
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        for path in batch_files:
            filename = os.path.basename(path)
            f = open(path, 'rb')
            opened_files.append(f)
            upload_files.append(('files', (filename, f, 'image/jpeg')))
        
        start_batch = time.time()
        response = session.post(URL, files=upload_files, timeout=90)
        response.raise_for_status()
        
        data = response.json()
        end_batch = time.time()
        
        batch_duration_ms = (end_batch - start_batch) * 1000
        avg_per_image_ms = batch_duration_ms / len(batch_files)
        
        return {
            "data": data,
            "batch_time_ms": batch_duration_ms,
            "avg_image_ms": avg_per_image_ms,
            "count": len(batch_files)
        }
    except Exception as e:
        print(f"❌ Lỗi khi xử lý Batch: {e}")
        return None
    finally:
        for f in opened_files:
            f.close()
        session.close()

def main():
    # 1. Thu thập danh sách ảnh
    supported_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_paths = [
        os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
        if f.lower().endswith(supported_extensions)
    ]
    
    if not image_paths:
        print(f"[-] Không tìm thấy ảnh trong thư mục: {IMAGE_DIR}")
        return

    print(f"🚀 Bắt đầu quá trình OCR & Xuất báo cáo")
    print(f"Tổng số ảnh: {len(image_paths)}")
    print(f"Cấu hình: {MAX_WORKERS} workers, {BATCH_SIZE} ảnh/batch")
    print("-" * 50)

    # 2. Chia batch
    batches = [image_paths[i:i + BATCH_SIZE] for i in range(0, len(image_paths), BATCH_SIZE)]
    
    start_total = time.time()
    final_results = []
    
    # 3. Chạy song song
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {executor.submit(send_request, batch): i for i, batch in enumerate(batches)}
        
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            result = future.result()
            
            if result:
                # Lưu kết quả OCR của từng ảnh vào danh sách tổng
                final_results.extend(result["data"]["results"])
                
                print(f"✅ Batch #{batch_idx+1:02d}: {result['count']} ảnh  | "
                      f"Tổng: {result['batch_time_ms']:.0f}ms | "
                      f"Tb/ảnh: {result['avg_image_ms']:.0f}ms")

    end_total = time.time()
    total_time_s = end_total - start_total
    total_images_processed = len(final_results)

    # 4. Xuất file JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    # 5. Thống kê cuối cùng
    if total_images_processed > 0:
        avg_total_ms = (total_time_s * 1000) / total_images_processed
        
        print("-" * 50)
        print(f"📊 THÔNG KÊ KẾT THÚC:")
        print(f"✔️ Tổng thời gian: {total_time_s:.2f} giây")
        print(f"✔️ Tổng số ảnh đã xử lý: {total_images_processed}")
        print(f"✔️ Thời gian trung bình 1 ảnh: {avg_total_ms:.2f} ms")
        print(f"✔️ Tốc độ thực tế: {total_images_processed / total_time_s:.2f} ảnh/giây")
        print(f"📂 Đã lưu kết quả vào file: {OUTPUT_JSON}")
        print("-" * 50)

if __name__ == "__main__":
    main()
