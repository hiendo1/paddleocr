import requests
import os
import json

import sys
import io

# Đảm bảo stdout hỗ trợ utf-8 để in tiếng Việt trên console Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# URL API của Server
URL = "http://45.77.3.200:8000/ocr/predict"

# Thư mục chứa ảnh test
IMAGE_DIR = r"c:\Users\Admin\Desktop\cleanup service\test_case"

def test_single_image(image_path):
    print(f"\n--- Dang xu ly: {os.path.basename(image_path)} ---")
    
    try:
        with open(image_path, 'rb') as f:
            files = [('files', (os.path.basename(image_path), f, 'image/jpeg'))]
            response = requests.post(URL, files=files)
            
        if response.status_code == 200:
            result = response.json()
            # Lấy danh sách kết quả của tệp đầu tiên (vì chúng ta chỉ gửi 1 tệp)
            img_results = result.get("results", [])[0].get("results", [])
            
            if not img_results:
                print("[-] Khong tim thay van ban nao.")
            else:
                print(f"[+] Tim thay {len(img_results)} dong van ban:")
                for i, line in enumerate(img_results, 1):
                    text = line.get("text")
                    conf = line.get("confidence")
                    # Có thể in tiếng Việt thoải mái bây giờ
                    print(f"  [{i}] Noi dung: {text} (Do tin cay: {conf:.2%})")
        else:
            print(f"[-] Loi Server: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"[-] Loi ket noi: {str(e)}")

def main():
    # Lấy danh sách 3 ảnh đầu tiên trong thư mục test_case để xem ví dụ
    supported_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    images = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
              if f.lower().endswith(supported_extensions)]
    
    if not images:
        print(f"[-] Khong tim thay anh trong thu muc: {IMAGE_DIR}")
        return

    print(f"[*] Tim thay tong cong {len(images)} anh. Se hien thi ket qua cua 10 anh dau tien.")
    
    # Chạy thử 3 ảnh đầu để xem cấu trúc kết quả
    for img_path in images[:1]:
        test_single_image(img_path)

if __name__ == "__main__":
    main()
