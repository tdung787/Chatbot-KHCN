import os
import sys
import json
import base64
import logging
import warnings
import fitz  # PyMuPDF để render ảnh
import pdfplumber  # để trích xuất text giữ layout
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def create_directories(base_dir: str):
    """Tạo thư mục con cho ảnh và text"""
    os.makedirs(os.path.join(base_dir, "page_images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "text"), exist_ok=True)


def process_page(pdfplumber_page, fitz_page, page_num, base_dir, items, zoom=2.0):
    """Render ảnh (PyMuPDF) và trích xuất text (pdfplumber)"""
    # --- Render ảnh bằng fitz ---
    matrix = fitz.Matrix(zoom, zoom)
    pix = fitz_page.get_pixmap(matrix=matrix, alpha=False)
    image_path = os.path.join(base_dir, f"page_images/page_{page_num:03d}.png")
    pix.save(image_path)

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf8")

    # --- Trích xuất text bằng pdfplumber ---
    text_content = pdfplumber_page.extract_text(x_tolerance=2, y_tolerance=3) or ""
    text_content = text_content.strip()

    text_path = os.path.join(base_dir, f"text/page_{page_num:03d}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    # --- Ghi thông tin vào danh sách ---
    items.append({
        "page": page_num,
        "type": "page",
        "image_path": image_path,
        "text_path": text_path,
        "image_base64": image_b64,
        "text_content": text_content
    })


def parse_pdf(filepath: str, output_dir: str = "data/output"):
    """Hàm chính — xuất ảnh + text từng trang PDF"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {filepath}")

    os.makedirs(output_dir, exist_ok=True)
    create_directories(output_dir)

    fitz_doc = fitz.open(filepath)
    plumber_doc = pdfplumber.open(filepath)

    num_pages = len(fitz_doc)
    items = []

    for page_num in tqdm(range(1, num_pages + 1), desc="📄 Đang xử lý PDF"):
        fitz_page = fitz_doc[page_num - 1]
        plumber_page = plumber_doc.pages[page_num - 1]
        process_page(plumber_page, fitz_page, page_num, output_dir, items)

    fitz_doc.close()
    plumber_doc.close()

    # --- Ghi file JSON tổng hợp ---
    output_json = os.path.join(output_dir, f"{os.path.basename(filepath)}_pages.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Hoàn tất! Kết quả lưu tại: {output_json}")
    print(f"📂 Ảnh: {os.path.join(output_dir, 'page_images')}")
    print(f"📂 Văn bản: {os.path.join(output_dir, 'text')}")
    return output_json


# --- Chạy trực tiếp ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "data/input/bt10.pdf"

    parse_pdf(input_file)
