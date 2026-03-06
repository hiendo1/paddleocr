import requests
import io
# Giả sử bạn dùng thư viện này để kết nối SQL (MySQL/PostgreSQL/SQLite)
# import pymysql 

URL = "http://45.77.3.200:8000/ocr/predict"

def call_ocr_from_db_records(db_records):
    """
    db_records: Danh sách các bản ghi từ Database.
    Mỗi record giả sử có dạng: {'id': 1, 'image_blob': b'\x89PNG...', 'filename': 'image1.png'}
    """
    
    upload_files = []
    
    for record in db_records:
        # Chuyển dữ liệu Binary (BLOB) từ DB thành một đối tượng File-like trong bộ nhớ
        image_data = record['image_blob']
        filename = record['filename']
        
        # Thêm vào danh sách gửi đi mà không cần lưu xuống ổ đĩa
        upload_files.append(
            ('files', (filename, io.BytesIO(image_data), 'application/octet-stream'))
        )

    try:
        # Gửi request multipart/form-data
        response = requests.post(URL, files=upload_files, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

# --- VÍ DỤ MINH HỌA ---
if __name__ == "__main__":
    # Mô phỏng dữ liệu lấy từ SQL
    mock_db_records = [
        {
            'id': 101, 
            'filename': 'db_image_1.png', 
            'image_blob': open(r"c:\Users\Admin\Desktop\cleanup service\test_case\aaa.png", "rb").read()
        }
    ]
    
    print("Đang gọi OCR từ dữ liệu Database...")
    result = call_ocr_from_db_records(mock_db_records)
    
    if result:
        for img_res in result['results']:
            print(f"\nKết quả cho {img_res['filename']}:")
            for line in img_res['results']:
                print(f"- {line['text']} (Conf: {line['confidence']:.2f})")
