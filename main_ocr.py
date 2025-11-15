from src.parsers import parse_ocr_all_text
# from src.parsers.pdf_parser import parse_pdf

def main():
    # OCR toàn bộ folder ảnh
    result = parse_ocr_all_text("data/input/page_images")
    print(result)

if __name__ == "__main__":
    main()
