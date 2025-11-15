import cv2
from pathlib import Path
import easyocr


def parse_ocr_all_text(input_path, output_folder="data/output/answers"):
    """
    Đọc tất cả các ảnh hoặc folder ảnh bằng EasyOCR
    và xuất kết quả OCR ra từng file .txt riêng
    (Đọc toàn bộ text trong ảnh, không giới hạn vùng màu)
    
    Args:
        input_path: Đường dẫn đến file ảnh hoặc thư mục chứa ảnh
        output_folder: Thư mục lưu các file txt kết quả
    """
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ Không tìm thấy đường dẫn: {input_path}")
        return None

    # Nếu là file ảnh đơn
    if input_path.is_file():
        image_files = [input_path]
    else:
        # Nếu là thư mục => lấy tất cả ảnh trong đó
        image_files = sorted([
            p for p in input_path.glob("*")
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ])

    if not image_files:
        print("❌ Không có file ảnh hợp lệ trong thư mục.")
        return None

    print(f"📂 Tổng số ảnh sẽ xử lý: {len(image_files)}")

    # Khởi tạo EasyOCR
    reader = easyocr.Reader(['vi', 'en'], gpu=False)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    all_results = []

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n=== 🔍 Ảnh {idx}/{len(image_files)}: {image_path.name} ===")

        # Đọc ảnh
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"⚠️ Không thể đọc ảnh {image_path}")
            continue

        # --- Đọc toàn bộ text trên ảnh ---
        results = reader.readtext(image, detail=1, paragraph=True)  # paragraph=True để gom câu liên tiếp

        page_texts = [text for (bbox, text, conf) in results if conf > 0.3]

        if page_texts:
            txt_filename = Path(output_folder) / f"{image_path.stem}.txt"
            with open(txt_filename, "w", encoding="utf-8") as f:
                for line in page_texts:
                    f.write(line + "\n")
            print(f"💾 Lưu kết quả tại: {txt_filename}")
            all_results.append(str(txt_filename))
        else:
            print(f"⚠️ Không có text nào để lưu cho {image_path.name}")

    print(f"\n✅ Đã xử lý xong {len(all_results)} ảnh, kết quả nằm trong: {output_folder}")
    return all_results


if __name__ == "__main__":
    # Ví dụ sử dụng:
    input_path = "data/images"  # Thay bằng folder hoặc file ảnh của bạn
    parse_ocr_all_text(input_path)
