import os
import sys
import pdfplumber
from tqdm import tqdm

def extract_text_per_page(filepath: str, output_dir: str = "data/output"):
    """Trích xuất text từ PDF và lưu riêng từng trang"""
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {filepath}")
    
    pages_dir = os.path.join(output_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    
    pdf = pdfplumber.open(filepath)
    num_pages = len(pdf.pages)
    
    print(f"📄 Đang đọc PDF: {os.path.basename(filepath)}")
    print(f"📊 Tổng số trang: {num_pages}\n")
    
    for i, page in enumerate(tqdm(pdf.pages, desc="Đang trích xuất text"), start=1):
        text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
        
        # Lưu từng trang
        page_file = os.path.join(pages_dir, f"page_{i:03d}.txt")
        with open(page_file, "w", encoding="utf-8") as f:
            f.write(text.strip())
    
    pdf.close()
    
    print(f"\n✅ Hoàn tất!")
    print(f"📁 Text từng trang lưu tại: {pages_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "data/input/bt10.pdf"
    
    extract_text_per_page(input_file)