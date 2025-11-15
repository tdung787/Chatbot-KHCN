import os
import sys
from pathlib import Path

# Đảm bảo import được module
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.parsers.pdf_parser import parse_pdf

def main():
    # Cấu hình đường dẫn
    input_path = "data/input/sinh12.pdf"
    output_dir = "data/output"
    
    # Kiểm tra file tồn tại
    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy file: {input_path}")
        return
    
    print(f"📄 Đang xử lý: {input_path}")
    
    # Gọi hàm parse
    result_path = parse_pdf(filepath=input_path, output_dir=output_dir)
    
    print(f"✅ Kết quả lưu tại: {result_path}")

if __name__ == "__main__":
    main()