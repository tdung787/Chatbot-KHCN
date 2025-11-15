import os
import fitz  # PyMuPDF
from tqdm import tqdm

def parse_pdf(filepath: str, output_dir: str = "data/output"):
    """Trích xuất chỉ text, bỏ qua ảnh hoàn toàn"""
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {filepath}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    doc = fitz.open(filepath)
    num_pages = len(doc)
    
    all_text = []
    
    print(f"📄 Đang đọc PDF: {os.path.basename(filepath)}")
    print(f"📊 Tổng số trang: {num_pages}\n")
    
    for page_num in tqdm(range(num_pages), desc="Đang trích xuất text"):
        page = doc[page_num]
        
        # Chỉ lấy text, bỏ qua ảnh
        text = page.get_text("text")  # Không lấy ảnh
        
        if text.strip():
            all_text.append(f"{'='*60}")
            all_text.append(f"TRANG {page_num + 1}")
            all_text.append(f"{'='*60}")
            all_text.append(text.strip())
            all_text.append("")
    
    doc.close()
    
    full_text = "\n".join(all_text)
    
    output_file = os.path.join(
        output_dir, 
        f"{os.path.splitext(os.path.basename(filepath))[0]}_text.txt"
    )
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"\n✅ Hoàn tất!")
    print(f"📁 File text: {output_file}")
    print(f"📝 Tổng ký tự: {len(full_text):,}")
    
    return output_file