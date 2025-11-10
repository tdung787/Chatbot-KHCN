"""
tools/quiz_generator.py

Tool để tạo đề kiểm tra trắc nghiệm bằng AI
Cost: ~$0.015/quiz (10 câu) - optimized
"""

import re
import json
from pathlib import Path
from typing import Dict, Optional
from openai import OpenAI 


# ================== STUDENT PROFILE LOADER ==================
def load_student_profile(profile_path: str = "data/api/student.json") -> Optional[Dict]:
    """Load student profile from JSON file"""
    try:
        path = Path(profile_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        
        if not path.exists():
            print(f"⚠️  Profile not found: {path}")
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            response = json.load(f)
        
        if not response.get("success"):
            print(f"⚠️  Load failed: {response.get('message')}")
            return None
        
        return response.get("data")
        
    except Exception as e:
        print(f"⚠️  Error: {e}")
        return None


def get_difficulty_vietnamese(difficulty_pref: str) -> str:
    """Convert difficulty to Vietnamese"""
    mapping = {"easy": "dễ", "medium": "trung bình", "hard": "khó"}
    return mapping.get(difficulty_pref.lower(), "trung bình")


class QuizGenerator:
    """Generate quiz using AI"""
    
    def __init__(self, openai_client: OpenAI, student_profile_path: str = "data/api/student.json"):
        self.client = openai_client
        self.student_profile = load_student_profile(student_profile_path)
        
        if self.student_profile:
            full_name = self.student_profile.get("user_id", {}).get("full_name", "")
            grade = self.student_profile.get("grade_level", "")
            diff = self.student_profile.get("difficulty_preference", "medium")
            print(f"✓ Profile: {full_name} - Lớp {grade} - Độ khó: {get_difficulty_vietnamese(diff)}")
    
    def get_student_info(self) -> Dict:
        """Get formatted student info"""
        if not self.student_profile:
            return {
                "full_name": "........................",
                "current_class": "........................",
                "difficulty": "trung bình",
                "grade_level": None
            }
        
        user_info = self.student_profile.get("user_id", {})
        return {
            "full_name": user_info.get("full_name", "........................"),
            "current_class": self.student_profile.get("current_class", "........................"),
            "difficulty": get_difficulty_vietnamese(self.student_profile.get("difficulty_preference", "medium")),
            "grade_level": self.student_profile.get("grade_level")
        }
    
    def generate_quiz(
        self,
        subject: str,
        topic: str,
        num_questions: int = 10,
        difficulty: str = None,
        time_limit: int = 15,
        use_student_difficulty: bool = True
    ) -> Dict:
        """Generate quiz - Fixed for 15-min, 10 questions"""
        
        # Force 15-min, 10 questions format
        num_questions = 10
        time_limit = 15
        
        student_info = self.get_student_info()

        # Use student difficulty preference
        if use_student_difficulty or difficulty is None:
            difficulty = student_info["difficulty"]
        else:
            if difficulty.lower() in ["easy", "medium", "hard"]:
                difficulty = get_difficulty_vietnamese(difficulty)
        
        print(f"\n📝 Tạo đề: {subject} - {topic}")
        print(f"   👤 {student_info['full_name']} ({student_info['current_class']})")
        print(f"   📊 10 câu - 15 phút - Độ khó: {difficulty}")
        
        # Optimized system prompt (reduced tokens)
        system_prompt = """Chuyên gia ra đề trắc nghiệm THPT. Tạo đề 15 phút, 10 câu.

QUY TẮC:
1. BẮT BUỘC: Đúng 10 câu (Câu 1→10)
2. Câu hỏi chính xác khoa học
3. Đáp án nhiễu hợp lý
4. CHỈ ĐỀ, KHÔNG ĐÁP ÁN

ĐỘ KHÓ (15 phút):
- Dễ: Nhớ định nghĩa, 1 bước tính, số đẹp. VD: "v=s/t với s=100m, t=10s"
- TB: 2-3 bước, so sánh khái niệm. VD: "v trung bình khi v đổi"
- Khó: 3-4 bước, kết hợp 2-3 công thức, bẫy nhỏ. VD: "đi-về khác v, tính s"

FORMAT:
# ĐỀ KIỂM TRA 15 PHÚT - [MÔN]
**Chủ đề**: [topic]
**Độ khó**: [level]
**Thời gian**: 15 phút
**Tổng điểm**: 10 điểm
**Họ và tên**: [name]
**Lớp**: [class]
---
## **Câu 1**: [question]
**A.** [option]  
**B.** [option]  
**C.** [option]  
**D.** [option]
...
## **Câu 10**: [question]
**A.** [option]  
**B.** [option]  
**C.** [option]  
**D.** [option]
---
_Hết_"""
        
        # Optimized user prompt (reduced tokens)
        difficulty_extra = ""
        if difficulty == "khó":
            difficulty_extra = "\n⚠️ Độ khó 'khó': 6-7 câu bài tập 3-4 bước, đáp án gần nhau, tối đa 2-3 câu lý thuyết."
        
        user_prompt = f"""Đề thi:
- Môn: {subject} | Chủ đề: {topic}
- Học sinh: {student_info['full_name']} - {student_info['current_class']}
- 10 câu, 15 phút, mỗi câu 1 điểm
- Độ khó: {difficulty}{difficulty_extra}

Yêu cầu: Đúng 10 câu, 4 đáp án/câu, không đáp án. Tập trung chủ đề "{topic}"."""
        
        try:
            print("   🤖 Đang sinh đề...")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=3000
            )
            
            quiz_markdown = response.choices[0].message.content.strip()
            
            # Validate
            if not self._validate_quiz(quiz_markdown, num_questions):
                print("   ⚠️ Retry...")
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt + "\n\nCHÚ Ý: Đúng format ## **Câu X**: và **A.**, **B.**, **C.**, **D.**"}
                    ],
                    temperature=0.7,
                    max_tokens=3000
                )
                quiz_markdown = response.choices[0].message.content.strip()
            
            print("   ✓ Hoàn thành!")
            
            metadata = self._extract_metadata(quiz_markdown)
            
            return {
                "success": True,
                "quiz_markdown": quiz_markdown,
                "metadata": {
                    "subject": subject,
                    "topic": topic,
                    "num_questions": num_questions,
                    "difficulty": difficulty,
                    "time_limit": time_limit,
                    "student_info": student_info,
                    **metadata
                }
            }
            
        except Exception as e:
            print(f"   ✗ Lỗi: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_quiz(self, quiz_markdown: str, expected_questions: int) -> bool:
        """Validate quiz format"""
        question_pattern = r'##\s+\*\*Câu\s+\d+\*\*:'
        questions = re.findall(question_pattern, quiz_markdown)
        
        if len(questions) != expected_questions:
            print(f"   ⚠️ Số câu: {len(questions)}/{expected_questions}")
            return False
        
        option_pattern = r'\*\*[A-D]\.\*\*'
        options = re.findall(option_pattern, quiz_markdown)
        
        if len(options) != expected_questions * 4:
            print(f"   ⚠️ Số đáp án: {len(options)}/{expected_questions * 4}")
            return False
        
        if "ĐÁP ÁN" in quiz_markdown.upper():
            print("   ⚠️ Có đáp án")
            return False
        
        return True
    
    def _extract_metadata(self, quiz_markdown: str) -> Dict:
        """Extract metadata"""
        question_pattern = r'##\s+\*\*Câu\s+\d+\*\*:'
        questions = re.findall(question_pattern, quiz_markdown)
        return {"total_questions_found": len(questions)}


def extract_topic_from_query(query: str, openai_client: OpenAI) -> Optional[Dict]:
    """Extract subject and topic from query"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Trích xuất từ: "{query}"

JSON:
{{
    "subject": "Toán"|"Vật lý"|"Hóa học"|"Sinh học"|null,
    "topic": "chủ đề CỤ THỂ (không chung chung)",
    "num_questions": 10,
    "user_difficulty": "dễ"|"trung bình"|"khó"|null (chỉ set nếu user NÓI RÕ)
}}

QUY TẮC QUAN TRỌNG:
- CHỈ trả về subject nếu là 1 trong 4 môn: Toán, Vật lý, Hóa học, Sinh học
- Nếu là môn khác (Văn, Anh, Sử, Địa, ...) → subject = null
- Nếu chủ đề chung chung (VD: "Động lực học"), hãy cụ thể hóa (VD: "Ba định luật Newton")
- Chỉ set "user_difficulty" khi user NÓI RÕ (dễ/TB/khó), còn không thì null

VD:
"Đề 15p Động lực học độ khó TB" → {{"subject":"Vật lý","topic":"Ba định luật Newton","num_questions":10,"user_difficulty":"trung bình"}}
"Tạo đề Văn về Chiếc lược ngà" → {{"subject":null,"topic":"Chiếc lược ngà","num_questions":10,"user_difficulty":null}}
"Tạo đề Vật lý Tốc độ" → {{"subject":"Vật lý","topic":"Tốc độ và vận tốc","num_questions":10,"user_difficulty":null}}
"15 câu Toán khó Hệ BPT" → {{"subject":"Toán","topic":"Hệ bất phương trình bậc nhất hai ẩn","num_questions":10,"user_difficulty":"khó"}}
"Cho tôi bài kiểm tra Tiếng Anh" → {{"subject":null,"topic":"Grammar","num_questions":10,"user_difficulty":null}}

Chỉ JSON."""
            }],
            temperature=0,
            max_tokens=150
        )
        
        content = response.choices[0].message.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        
        if result.get("subject") and result.get("topic"):
            return result
        
        return None
        
    except Exception as e:
        print(f"⚠️ Extract error: {e}")
        return None