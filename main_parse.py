from src.parsers import parse_ocr
from src.parsers.pdf_parser import parse_pdf

def main():
    input_path = "data/input/bt10-full.pdf"
    output_dir = "data/output"
    
    result_path = parse_pdf(filepath=input_path, output_dir=output_dir)
    print(f"Kết quả lưu tại: {result_path}")

if __name__ == "__main__":
    main()