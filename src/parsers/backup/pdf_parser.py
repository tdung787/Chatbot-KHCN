import os
import sys
import json
import base64
import logging
import warnings
import fitz  # PyMuPDF
import tabula
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def create_directories(base_dir: str):
    """Tạo các thư mục con cần thiết trong data/output"""
    directories = ["images", "text", "tables", "page_images"]
    for dir_name in directories:
        os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)


def process_tables(filepath, page_num, base_dir, items):
    """Trích xuất bảng từ PDF"""
    try:
        tables = tabula.read_pdf(filepath, pages=page_num, multiple_tables=True)
        if not tables:
            return
        for table_idx, table in enumerate(tables):
            table_text = "\n".join([" | ".join(map(str, row)) for row in table.values])
            table_file_name = f"{base_dir}/tables/{os.path.basename(filepath)}_table_{page_num}_{table_idx}.txt"
            with open(table_file_name, "w", encoding="utf-8") as f:
                f.write(table_text)
            items.append({
                "page": page_num,
                "type": "table",
                "text": table_text,
                "path": table_file_name
            })
    except Exception as e:
        print(f"⚠️ Error extracting tables from page {page_num}: {str(e)}")


def process_text_chunks(text, text_splitter, page_num, base_dir, items, filepath):
    """Chia nhỏ văn bản để xử lý"""
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        text_file_name = f"{base_dir}/text/{os.path.basename(filepath)}_text_{page_num}_{i}.txt"
        with open(text_file_name, "w", encoding="utf-8") as f:
            f.write(chunk)
        items.append({
            "page": page_num,
            "type": "text",
            "text": chunk,
            "path": text_file_name
        })


def process_images(doc, page, page_num, base_dir, items, filepath):
    """Trích xuất ảnh nhúng trong PDF"""
    images = page.get_images(full=True)
    for idx, image in enumerate(images):
        xref = image[0]
        pix = fitz.Pixmap(doc, xref)
        image_name = f"{base_dir}/images/{os.path.basename(filepath)}_image_{page_num}_{idx}_{xref}.png"

        # Giữ chất lượng cao nhất
        if pix.n - pix.alpha < 4:
            pix.save(image_name)
        else:
            pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
            pix_rgb.save(image_name)
            pix_rgb = None
        pix = None

        with open(image_name, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf8")
        items.append({
            "page": page_num,
            "type": "image",
            "path": image_name,
            "image": encoded_image
        })


def process_page_images(page, page_num, base_dir, items, zoom=2.0):
    """Render toàn bộ trang PDF thành PNG chất lượng cao"""
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    page_path = os.path.join(base_dir, f"page_images/page_{page_num:03d}.png")
    pix.save(page_path)

    with open(page_path, "rb") as f:
        page_image = base64.b64encode(f.read()).decode("utf8")

    items.append({
        "page": page_num,
        "type": "page",
        "path": page_path,
        "image": page_image
    })


def parse_pdf(filepath: str, output_dir: str = "data/output"):
    """Hàm chính để trích xuất nội dung PDF"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {filepath}")

    os.makedirs(output_dir, exist_ok=True)
    create_directories(output_dir)

    doc = fitz.open(filepath)
    num_pages = len(doc)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=200, length_function=len
    )

    items = []

    for page_num in tqdm(range(1, num_pages + 1), desc="📄 Processing PDF pages"):
        page = doc[page_num - 1]  # PyMuPDF vẫn đánh index từ 0
        text = page.get_text("text")

        process_tables(filepath, page_num, output_dir, items)
        process_text_chunks(text, text_splitter, page_num, output_dir, items, filepath)
        process_images(doc, page, page_num, output_dir, items, filepath)
        process_page_images(page, page_num, output_dir, items)

    output_json = os.path.join(output_dir, f"{os.path.basename(filepath)}_extracted.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Hoàn tất! Kết quả được lưu trong: {output_json}")
    return output_json


# Cho phép chạy độc lập nếu cần
if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "data/input/bt10.pdf"

    parse_pdf(input_file)