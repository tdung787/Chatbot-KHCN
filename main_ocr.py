from src.parsers import parse_ocr
from src.parsers.pdf_parser import parse_pdf

def main():
    
    result = parse_ocr(
        "data/input/page_images",
    )

if __name__ == "__main__":
    main()