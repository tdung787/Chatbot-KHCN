from .pdf_answer_extractor import parse_ocr_all_text

# Backwards-compatible alias (some callers expect `parse_ocr`)
parse_ocr = parse_ocr_all_text

__all__ = [
	'parse_ocr_all_text',
	'parse_ocr',
]