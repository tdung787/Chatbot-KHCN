import cv2
import numpy as np
import easyocr
from pathlib import Path


def merge_nearby_boxes(bounding_boxes, vertical_threshold=20, horizontal_threshold=50):
    """Gộp các bounding box gần nhau thành 1 box lớn"""
    if not bounding_boxes:
        return []

    boxes = sorted(bounding_boxes, key=lambda b: b[1])
    merged = []
    current_group = [boxes[0]]

    for box in boxes[1:]:
        x, y, w, h = box
        last_box = current_group[-1]
        last_x, last_y, last_w, last_h = last_box

        vertical_distance = y - (last_y + last_h)
        x_overlap = not (x > last_x + last_w + horizontal_threshold or
                         last_x > x + w + horizontal_threshold)

        if vertical_distance <= vertical_threshold and x_overlap:
            current_group.append(box)
        else:
            merged.append(_merge_boxes(current_group))
            current_group = [box]

    if current_group:
        merged.append(_merge_boxes(current_group))

    return merged


def _merge_boxes(boxes):
    """Gộp list các box thành 1 box lớn"""
    if len(boxes) == 1:
        return boxes[0]

    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    max_x = max(b[0] + b[2] for b in boxes)
    max_y = max(b[1] + b[3] for b in boxes)

    return (min_x, min_y, max_x - min_x, max_y - min_y)


def parse_ocr(input_path, output_folder="data/output/answers"):
    """
    Đọc tất cả các ảnh hoặc folder ảnh bằng EasyOCR
    và xuất kết quả OCR ra từng file .txt riêng

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

    reader = easyocr.Reader(['vi', 'en'], gpu=False)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    all_results = []

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n=== 🔍 Ảnh {idx}/{len(image_files)}: {image_path.name} ===")

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"⚠️ Không thể đọc ảnh {image_path}")
            continue

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([50, 120, 120])
        upper_green = np.array([70, 255, 255])
        mask = cv2.inRange(hsv_image, lower_green, upper_green)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        if not valid_contours:
            print("⚠️ Không tìm thấy vùng xanh.")
            continue

        bounding_boxes = [cv2.boundingRect(cnt) for cnt in valid_contours]
        merged_boxes = merge_nearby_boxes(bounding_boxes, vertical_threshold=25, horizontal_threshold=50)
        merged_boxes = sorted(merged_boxes, key=lambda b: b[1])

        page_texts = []

        for i, (x, y, w, h) in enumerate(merged_boxes):
            padding = 5
            y1 = max(0, y - padding)
            y2 = min(image.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(image.shape[1], x + w + padding)

            roi = image[y1:y2, x1:x2]
            results = reader.readtext(roi, detail=1, paragraph=False)

            text_parts = [text for (bbox, text, conf) in results if conf > 0.3]
            full_text = " ".join(text_parts).strip()

            if full_text:
                page_texts.append(full_text)
                print(f"✅ {i+1}: {full_text}")
            else:
                print(f"⚠️ {i+1}: [KHÔNG ĐỌC ĐƯỢC]")

        # === Lưu từng file TXT riêng ===
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
