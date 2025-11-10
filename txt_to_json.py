import os
import json
import re
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from tqdm import tqdm

# ================== CONFIG ==================
INPUT_FOLDER = "data/output/assigned_answers"
OUTPUT_JSON = "database/parsed_questions.json"
STATS_JSON = "database/parsing_statistics.json"

# ================== LOGGING SETUP ==================
def setup_logging():
    """Setup logging"""
    log_folder = Path("database/logs")
    log_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_folder / f"parsing_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ================== PARSE QUESTION FILE ==================
def normalize_text(text: str) -> str:
    """
    Chuẩn hóa text: loại bỏ xuống dòng thừa, khoảng trắng thừa
    """
    # Thay \n bằng space
    text = text.replace('\n', ' ')
    # Loại bỏ nhiều space liên tiếp
    text = re.sub(r'\s+', ' ', text)
    # Trim
    return text.strip()

def parse_all_questions(content: str, page_markers: List[tuple]) -> List[Dict]:
    """
    Parse toàn bộ content thành list các câu hỏi với format chuẩn
    
    Args:
        content: Toàn bộ text đã gộp từ tất cả các file
        page_markers: List of (start_pos, end_pos, page_name) để track vị trí
    
    Returns:
        List[Dict]: Mỗi dict chứa question, options, correct_answer
    """
    
    def get_page_info(start_pos: int, end_pos: int) -> Dict:
        """Xác định câu hỏi nằm trong page nào"""
        pages = []
        for page_start, page_end, page_name in page_markers:
            # Check if question overlaps with this page
            if not (end_pos < page_start or start_pos > page_end):
                pages.append(page_name)
        
        if not pages:
            return {"primary_page": "unknown", "spans_pages": []}
        elif len(pages) == 1:
            return {"primary_page": pages[0].replace('.txt', ''), "spans_pages": []}
        else:
            return {"primary_page": pages[0].replace('.txt', ''), "spans_pages": pages}
    
    
    # Regex để tách từng câu hỏi với vị trí
    # Pattern: Câu X: ... A. ... B. ... C. ... D. ... <Đáp án: Y>
    pattern = r'Câu\s+(\d+)[.:]?\s*(.*?)(?=Câu\s+\d+[.:]?|$)'
    matches = [(m.group(1), m.group(2), m.start(), m.end()) 
               for m in re.finditer(pattern, content, re.DOTALL)]
    
    questions = []
    
    for match in matches:
        question_num = match[0]
        question_block = match[1].strip()
        start_pos = match[2]
        end_pos = match[3]
        
        # Get page info
        page_info = get_page_info(start_pos, end_pos)
        
        # Tìm phần câu hỏi (trước các lựa chọn A, B, C, D)
        question_text_match = re.match(r'(.*?)(?=\s*[A-D]\.)', question_block, re.DOTALL)
        if not question_text_match:
            logger.warning(f"⚠️  Không parse được câu hỏi {question_num} (page: {page_info['primary_page']}) - Không tìm thấy pattern câu hỏi")
            continue
        
        question_text = question_text_match.group(1).strip()
        
        # Normalize question text
        question_text = normalize_text(question_text)
        
        # Tìm các lựa chọn A, B, C, D
        options = {}
        option_pattern = r'([A-D])\.\s*(.*?)(?=\s*[A-D]\.|<Đáp án:|$)'
        option_matches = re.findall(option_pattern, question_block, re.DOTALL)
        
        for opt_letter, opt_text in option_matches:
            # Normalize option text
            options[opt_letter] = normalize_text(opt_text)
        
        # Tìm đáp án đúng
        answer_match = re.search(r'<Đáp án:\s*([A-D])\s*>', question_block)
        if not answer_match:
            logger.warning(f"⚠️  Không tìm thấy đáp án cho câu {question_num} (page: {page_info['primary_page']})")
            continue
        
        correct_answer = answer_match.group(1)
        
        # Validate
        if len(options) != 4:
            logger.warning(f"⚠️  Câu {question_num} (page: {page_info['primary_page']}) không đủ 4 lựa chọn (có {len(options)}: {list(options.keys())})")
            continue
        
        if correct_answer not in options:
            logger.error(f"❌ Đáp án {correct_answer} không có trong options câu {question_num} (page: {page_info['primary_page']})")
            raise ValueError(f"Invalid answer key in question {question_num}")
        
        # Create unique ID based on primary page
        question_id = f"{page_info['primary_page']}_cau_{question_num}"
        
        question_data = {
            "id": question_id,
            "question": question_text,
            "options": options,
            "correct_answer": correct_answer,
            "correct_answer_text": options[correct_answer],
            "question_number": int(question_num),
            "primary_page": page_info['primary_page'],
            "subject": "Vật lý"  # Hardcode cho test
        }
        
        # Add spans_pages if question is split across pages
        if page_info['spans_pages']:
            question_data["spans_pages"] = page_info['spans_pages']
            logger.info(f"📄 Câu {question_num} bị ngắt qua các trang: {page_info['spans_pages']}")
        
        questions.append(question_data)
    
    return questions

# ================== MAIN PROCESS ==================
def main():
    logger.info("=" * 70)
    logger.info("BẮT ĐẦU PARSE CÂU HỎI TỪ TXT → JSON")
    logger.info("=" * 70)
    
    # Check input folder
    input_path = Path(INPUT_FOLDER)
    if not input_path.exists():
        logger.error(f"❌ Không tìm thấy folder: {INPUT_FOLDER}")
        return
    
    txt_files = sorted(input_path.glob("*.txt"))
    logger.info(f"📂 Folder input: {INPUT_FOLDER}")
    logger.info(f"📄 Tìm thấy {len(txt_files)} file txt")
    
    # Statistics
    stats = {
        "total_files": len(txt_files),
        "success_files": 0,
        "failed_files": 0,
        "total_questions": 0,
        "total_chars": 0
    }
    
    # Concatenate all files into one large text
    logger.info("🔗 Đang gộp tất cả file txt thành 1 file lớn...")
    full_text = ""
    page_markers = []  # Track position of each page: (start_pos, end_pos, page_name)
    
    for txt_file in tqdm(txt_files, desc="Reading files"):
        try:
            start_pos = len(full_text)
            content = txt_file.read_text(encoding='utf-8', errors='replace')
            full_text += content + "\n\n"  # Add spacing between files
            end_pos = len(full_text)
            
            page_markers.append((start_pos, end_pos, txt_file.name))
            stats["success_files"] += 1
        except Exception as e:
            logger.error(f"❌ Lỗi đọc file {txt_file.name}: {e}")
            stats["failed_files"] += 1
    
    logger.info(f"✓ Đã gộp {stats['success_files']} files")
    logger.info(f"📏 Tổng độ dài: {len(full_text):,} ký tự")
    
    # Update stats with total chars
    stats["total_chars"] = len(full_text)
    
    # Parse all questions from concatenated text
    logger.info("🔍 Đang parse câu hỏi từ text đã gộp...")
    
    try:
        all_questions = parse_all_questions(full_text, page_markers)
        stats["total_questions"] = len(all_questions)
        logger.info(f"✓ Đã parse thành công {len(all_questions)} câu hỏi")
        
    except Exception as e:
        logger.error(f"❌ Lỗi parse: {e}")
        logger.error("🛑 DỪNG TOÀN BỘ QUÁ TRÌNH DO LỖI PARSE")
        raise
    
    # Check for duplicate IDs
    ids = [q["id"] for q in all_questions]
    duplicate_ids = [id for id in ids if ids.count(id) > 1]
    
    if duplicate_ids:
        logger.warning(f"⚠️  Tìm thấy {len(set(duplicate_ids))} ID trùng lặp!")
        for dup_id in set(duplicate_ids):
            dup_questions = [q for q in all_questions if q["id"] == dup_id]
            logger.warning(f"   - {dup_id}: Xuất hiện {len(dup_questions)} lần")
            for q in dup_questions:
                logger.warning(f"     Question #{q['question_number']}: {q['question'][:50]}...")
    
    # Count questions that span multiple pages
    split_questions = [q for q in all_questions if "spans_pages" in q]
    if split_questions:
        logger.info(f"📄 Tìm thấy {len(split_questions)} câu hỏi bị ngắt qua nhiều trang")
        stats["split_questions"] = len(split_questions)
    
    # Save to JSON
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ Đã lưu {len(all_questions)} câu hỏi vào: {OUTPUT_JSON}")
    
    # Save statistics
    stats["timestamp"] = datetime.now().isoformat()
    stats["output_file"] = OUTPUT_JSON
    
    with open(STATS_JSON, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ Đã lưu thống kê vào: {STATS_JSON}")
    
    # Print summary
    logger.info("=" * 70)
    logger.info("KẾT QUẢ TỔNG KẾT")
    logger.info("=" * 70)
    logger.info(f"📁 Tổng số file đọc: {stats['total_files']}")
    logger.info(f"✓ Đọc thành công: {stats['success_files']}")
    logger.info(f"❌ Đọc thất bại: {stats['failed_files']}")
    logger.info(f"📏 Tổng ký tự: {stats['total_chars']:,}")
    logger.info(f"📝 Tổng số câu hỏi: {stats['total_questions']}")
    if duplicate_ids:
        logger.info(f"⚠️  ID trùng lặp: {len(set(duplicate_ids))}")
    if stats.get("split_questions"):
        logger.info(f"📄 Câu hỏi bị ngắt trang: {stats['split_questions']}")
    logger.info(f"💾 Output JSON: {OUTPUT_JSON}")
    logger.info(f"📊 Statistics: {STATS_JSON}")
    logger.info("=" * 70)
    
    # Sample output
    logger.info("\n📋 SAMPLE - 3 câu hỏi đầu tiên:")
    for i, q in enumerate(all_questions[:3], 1):
        logger.info(f"\n{i}. ID: {q['id']}")
        logger.info(f"   Câu hỏi: {q['question'][:80]}...")
        logger.info(f"   Đáp án: {q['correct_answer']} - {q['correct_answer_text'][:50]}...")
    
    logger.info("\n✅ HOÀN TẤT! Hãy kiểm tra file JSON trước khi embed.")

# ================== CLI ==================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Người dùng dừng chương trình")
    except Exception as e:
        logger.error(f"\n❌ Lỗi nghiêm trọng: {e}", exc_info=True)