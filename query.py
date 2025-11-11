import os
import json
import re
import platform
import subprocess
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Import graph tool and quiz tool
from src.tools.graph_generator import (
    GraphGenerator,
    extract_equation_from_query,
    extract_range_from_query
)
from src.tools.quiz_generator import (
    QuizGenerator,
    extract_topic_from_query
)
from src.tools.quiz_storage import QuizStorage
from src.tools.quiz_guard import QuizGuard
from src.tools.submission_manager import SubmissionManager

load_dotenv()

# ================== CONFIG ==================
OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
QDRANT_PATH = "database/qdrant_storage"
COLLECTION_NAME = "KHTN_QA"

# Supported subjects
SUBJECTS = {
    "Vật lý": ["vật lý", "physics", "lực", "năng lượng", "điện", "từ", "quang", "nhiệt"],
    "Hóa học": ["hóa học", "chemistry", "phản ứng", "nguyên tố", "hợp chất", "ion"],
    "Sinh học": ["sinh học", "biology", "tế bào", "gen", "protein", "DNA"],
    "Toán": ["toán", "math", "phương trình", "hàm số", "đồ thị", "số học"]
}
# Allowed subjects for quiz generation
ALLOWED_QUIZ_SUBJECTS = ["Toán", "Vật lý", "Hóa học", "Sinh học"]

# ================== INTENT CLASSIFIER ==================
class IntentClassifier:
    """Classify user query intent using LLM"""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def classify(self, query: str) -> Dict:
        """Classify query intent"""
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """Bạn là trợ lý phân loại câu hỏi học sinh.

Phân tích câu hỏi và xác định:
1. Có phải câu hỏi về môn học tự nhiên không? (Toán, Lý, Hóa, Sinh)
2. Nếu có, thuộc môn nào?

Trả về JSON với format:
{
    "is_subject_question": true/false,
    "subject": "Vật lý" | "Hóa học" | "Sinh học" | "Toán" | null,
    "confidence": 0.0-1.0,
    "reasoning": "lý do ngắn gọn"
}

Ví dụ:
- "Định luật Newton là gì?" → {"is_subject_question": true, "subject": "Vật lý", "confidence": 0.95, "reasoning": "Câu hỏi về định luật vật lý"}
- "Hôm nay thời tiết thế nào?" → {"is_subject_question": false, "subject": null, "confidence": 0.9, "reasoning": "Không liên quan môn học"}
"""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=0
            )
            
            # Parse JSON from response
            content = response.choices[0].message.content.strip()
            
            # Extract JSON if wrapped in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            return {
                "is_subject_question": result.get("is_subject_question", False),
                "subject": result.get("subject"),
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", "")
            }
            
        except Exception as e:
            print(f"⚠️  Lỗi classify: {e}")
            return {
                "is_subject_question": False,
                "subject": None,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}"
            }

# ================== RETRIEVAL TOOL ==================
class QuestionRetriever:
    """Retrieve relevant questions from Qdrant"""
    
    def __init__(self, client: OpenAI, qdrant_path: str, collection_name: str):
        self.openai_client = client
        self.qdrant_client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name
    
    def _embed_text(self, text: str) -> List[float]:
        """Embed text using OpenAI"""
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    
    def search(
        self, 
        query: str, 
        subject: Optional[str] = None,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Search for relevant questions
        
        Args:
            query: User query
            subject: Filter by subject (optional)
            top_k: Number of results to return
            
        Returns:
            List of relevant questions with metadata
        """
        try:
            # Embed query
            query_vector = self._embed_text(query)
            
            # Build filter if subject specified
            search_filter = None
            if subject:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="subject",
                            match=MatchValue(value=subject)
                        )
                    ]
                )
            
            # Search
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=top_k,
                with_payload=True
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "question": result.payload.get("question", ""),
                    "options": result.payload.get("options", {}),
                    "correct_answer": result.payload.get("correct_answer", ""),
                    "correct_answer_text": result.payload.get("correct_answer_text", ""),
                    "question_id": result.payload.get("id", ""),
                    "primary_page": result.payload.get("primary_page", ""),
                    "subject": result.payload.get("subject", ""),
                    "score": result.score
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"⚠️  Lỗi search: {e}")
            return []

# ================== TOOL FUNCTION ==================
def search_questions_tool(
    query: str, 
    intent_classifier: IntentClassifier,
    retriever: QuestionRetriever
) -> str:
    """
    Tool function to search questions
    
    Args:
        query: User query
        intent_classifier: Intent classifier instance
        retriever: Question retriever instance
        
    Returns:
        Formatted search results
    """
    # Classify intent
    intent = intent_classifier.classify(query)
    
    print(f"\n🔍 Intent Classification:")
    print(f"   - Is subject question: {intent['is_subject_question']}")
    print(f"   - Subject: {intent['subject']}")
    print(f"   - Confidence: {intent['confidence']:.2f}")
    print(f"   - Reasoning: {intent['reasoning']}")
    
    if not intent['is_subject_question'] or intent['confidence'] < 0.7:
        return "Câu hỏi này không liên quan đến môn học tự nhiên. Tôi không thể tìm kiếm trong database."
    
    # Search with subject filter
    results = retriever.search(
        query=query,
        subject=intent['subject'],
        top_k=3
    )
    
    if not results:
        return f"Không tìm thấy câu hỏi liên quan về {intent['subject']}."
    
    # Format results
    output = f"Tìm thấy {len(results)} câu hỏi liên quan:\n\n"
    
    for i, result in enumerate(results, 1):
        output += f"--- Câu hỏi {i} (Độ tương đồng: {result['score']:.2f}) ---\n"
        output += f"ID: {result['question_id']}\n"
        output += f"Môn: {result['subject']}\n"
        output += f"Câu hỏi: {result['question']}\n"
        output += f"Các lựa chọn:\n"
        for key, value in result['options'].items():
            marker = "✓" if key == result['correct_answer'] else " "
            output += f"  [{marker}] {key}. {value}\n"
        output += f"Đáp án đúng: {result['correct_answer']} - {result['correct_answer_text']}\n\n"
    
    return output

# ================== SIMPLE AGENT (without LangChain) ==================
class SimpleAgent:
    """Simple agent implementation without LangChain"""
    
    def __init__(self, client: OpenAI, intent_classifier: IntentClassifier, retriever: QuestionRetriever):
        self.client = client
        self.intent_classifier = intent_classifier
        self.retriever = retriever
        self.graph_generator = GraphGenerator(client)
        self.quiz_generator = QuizGenerator(client)
        self.quiz_storage = QuizStorage()
        self.quiz_guard = QuizGuard(client)
        self.submission_manager = SubmissionManager()
        self.conversation_history = []
    
    def _get_system_prompt(self, mode: str = "general") -> str:
        """
        Get system prompt with real-time pending quiz check
        
        Args:
            mode: "general" | "search" - prompt mode
        """
        
        # Get student profile
        student_info = ""
        student_id = "unknown"
        if self.quiz_generator.student_profile:
            profile = self.quiz_generator.student_profile
            student_id = profile.get('_id', 'unknown')
            student_info = f"""
THÔNG TIN HỌC SINH:
- Họ tên: {profile.get('name', 'N/A')}
- Lớp: {profile.get('grade', 'N/A')}
- Độ khó phù hợp: {profile.get('difficulty_level', 'N/A')}
"""
        
        # Check pending quiz
        pending_quiz = self.quiz_storage.get_latest_pending_quiz(student_id)
        
        pending_warning = ""
        if pending_quiz:
            pending_warning = f"""
⚠️⚠️⚠️ CẢNH BÁO QUAN TRỌNG ⚠️⚠️⚠️

HỌC SINH ĐANG CÓ BÀI KIỂM TRA CHƯA NỘP!
- Quiz ID: {pending_quiz['id']}
- Môn: {pending_quiz.get('subject', 'N/A')}
- Chủ đề: {pending_quiz.get('topic', 'N/A')}

QUY TẮC BẮT BUỘC (NGHIÊM NGẶT):
1. ❌ KHÔNG được tạo đề kiểm tra mới
2. ❌ KHÔNG được giải thích nội dung liên quan đến đề đang làm
3. ❌ KHÔNG được đưa ra gợi ý giúp làm bài
4. ✅ CHỈ được chat về: thời tiết, câu chuyện, định nghĩa TỔNG QUÁT không liên quan đến đề

Nếu học sinh yêu cầu tạo đề hoặc hỏi nội dung đề:
→ TỪ CHỐI lịch sự và nhắc nhở nộp bài trước.

Ví dụ từ chối:
"Bạn cần nộp bài kiểm tra hiện tại trước khi tạo đề mới! Quiz ID: {pending_quiz['id']}"
"""
        
        # Build prompt based on mode
        if mode == "search":
            return f"""Bạn là trợ lý giáo dục thông minh.

{student_info}

{pending_warning}

NHIỆM VỤ:
1. Dựa vào kết quả tìm kiếm, trả lời câu hỏi của học sinh
2. Giải thích rõ ràng, dễ hiểu
3. Trích dẫn nguồn (ID câu hỏi) khi trả lời
4. Không copy nguyên văn, hãy diễn giải

PHONG CÁCH: Thân thiện, khuyến khích học sinh tư duy

Ví dụ trích dẫn: "Theo câu hỏi page_002_cau_5..."
"""
        else:  # general mode
            return f"""Bạn là trợ lý học tập AI cho học sinh THPT Việt Nam.

{student_info}

{pending_warning}

NHIỆM VỤ:
- Giải đáp thắc mắc học tập (trừ khi có quiz pending và câu hỏi liên quan)
- KHÔNG tạo đề kiểm tra nếu có quiz pending
- Vẽ đồ thị minh họa (nếu cần)
- Tìm kiếm thông tin (nếu cần)

PHONG CÁCH:
- Thân thiện, dễ hiểu
- Giải thích rõ ràng với ví dụ
- Khuyến khích tư duy độc lập

Hãy giúp học sinh học tốt hơn! 📚✨"""
    
    def _should_use_tool(self, query: str) -> bool:
        """Decide if should use search tool"""
        # Quick keyword check first
        keywords = ["gì", "nào", "như thế nào", "tại sao", "là gì", "?"]
        has_question = any(kw in query.lower() for kw in keywords)
        
        if not has_question:
            return False
        
        # Check if related to subjects
        for subject, keywords in SUBJECTS.items():
            if any(kw in query.lower() for kw in keywords):
                return True
        
        return False
    
    def _should_draw_graph(self, query: str) -> bool:
        """Detect if query asks for graph"""
        graph_keywords = ["vẽ đồ thị", "vẽ đồ", "đồ thị", "graph", "plot", "vẽ hàm"]
        return any(kw in query.lower() for kw in graph_keywords)
    
    def _should_create_quiz(self, user_query: str) -> bool:
        """
        Detect quiz creation intent
        
        Uses hybrid approach:
        1. Keyword matching (primary - fast & reliable)
        2. Regex patterns (backup - catch edge cases)
        
        Returns:
            True if user wants to create a quiz
        """
        query_lower = user_query.lower()
        
        # ========== METHOD 1: KEYWORD MATCHING ==========
        # Simple, fast, covers 95% of cases
        quiz_keywords = [
            # Core keywords
            "tạo đề", "ra đề", "đề kiểm tra", "đề thi", "bài kiểm tra",
            
            # English
            "quiz", "test",
            
            # Variants
            "trắc nghiệm", "15 phút", "30 phút",
            
            # Short forms
            "kiểm tra", "bài thi",
            
            # Request patterns
            "cho tôi bài", "cho em bài", "cho mình bài",
            "cho tôi đề", "cho em đề", "cho mình đề",
            
            # Action verbs
            "tạo bài", "ra bài", "làm bài",
            "muốn bài", "cần bài", "muốn đề", "cần đề"
        ]
        
        for keyword in quiz_keywords:
            if keyword in query_lower:
                print(f"   ✓ Matched keyword: '{keyword}'")
                return True
        
        # ========== METHOD 2: REGEX PATTERNS ==========
        # Backup for complex cases
        patterns = [
            r'cho\s+(tôi|em|mình)\s+(một|1)?\s*(bài|đề)',
            r'(tạo|ra|làm)\s+(cho\s+)?(tôi|em|mình)?\s*(một|1)?\s*(bài|đề)',
            r'(muốn|cần|được)\s+(làm|có)?\s*(bài|đề)',
        ]
        
        for pattern in patterns:
            if re.search(pattern, query_lower):
                print(f"   ✓ Matched regex pattern")
                return True
        
        print("   ✗ No quiz creation intent detected")
        return False
    
    def _extract_equation(self, query: str) -> Optional[str]:
        """Extract equation from query"""
        return extract_equation_from_query(query, self.client)
    
    def query(self, user_query: str) -> str:
        """Process user query"""
        try:
            print(f"\n{'='*70}")
            print(f"USER QUERY: {user_query}")
            print(f"{'='*70}")
            
            # Get student ID from profile
            student_id = "unknown"
            if self.quiz_generator.student_profile:
                student_id = self.quiz_generator.student_profile.get("_id", "unknown")
            
            # ========== CHECK PENDING QUIZ (EARLY RETURN) ==========
            pending_quiz = self.quiz_storage.get_latest_pending_quiz(student_id)
            
            if pending_quiz:
                print(f"\n⚠️  Student có quiz đang làm: {pending_quiz['id']}")
                
                # PRIORITY 1: Block new quiz creation
                if self._should_create_quiz(user_query):
                    print("   🚫 BLOCKED: Cannot create new quiz")
                    
                    return f"""❌ Bạn không thể tạo đề mới khi đang có bài chưa nộp!

📋 **Bài kiểm tra đang chờ:**
- Quiz ID: `{pending_quiz['id']}`
- Môn: {pending_quiz.get('subject', 'N/A')}
- Chủ đề: {pending_quiz.get('topic', 'N/A')}
- Ngày tạo: {pending_quiz.get('date', 'N/A')[:10]}

💡 **Hướng dẫn nộp bài:**
```bash
POST /api/submission/submit?quiz_id={pending_quiz['id']}&student_id={student_id}&answers=1-A,2-B,3-C,4-D,5-A,6-B,7-C,8-D,9-A,10-B
```

Sau khi nộp xong, bạn có thể tạo đề mới! 📝"""
                
                # PRIORITY 2: Check if cheating
                guard_result = self.quiz_guard.is_cheating(user_query, pending_quiz)
                
                if guard_result["is_blocked"]:
                    print(f"   🚫 BLOCKED: {guard_result['reason']} (method: {guard_result['method']})")
                    
                    return f"""🚫 **Không thể trả lời câu hỏi này!**

**Lý do:** {guard_result['reason']}

Bạn đang làm bài kiểm tra về **{pending_quiz.get('topic', 'N/A')}** (Môn {pending_quiz.get('subject', 'N/A')}).

💡 Hãy hoàn thành và nộp bài để có thể hỏi lại! 📝"""
                else:
                    print(f"   ✓ ALLOWED: {guard_result['reason']} (method: {guard_result['method']})")
            # =======================================================
            
            # Debug: Check all conditions
            print(f"\n🔍 Debug:")
            print(f"   - Should create quiz: {self._should_create_quiz(user_query)}")
            print(f"   - Should draw graph: {self._should_draw_graph(user_query)}")
            print(f"   - Should use search: {self._should_use_tool(user_query)}")
            
            # Check if quiz request
            if self._should_create_quiz(user_query):
                print("\n📝 Phát hiện yêu cầu tạo đề kiểm tra!")
                
                # Extract subject and topic
                quiz_info = extract_topic_from_query(user_query, self.client)
                
                # ========== CHECK 1: Tool failure ==========
                if not quiz_info:
                    return """Xin lỗi, mình chưa hiểu rõ yêu cầu của bạn 😅

📚 **Hệ thống hiện hỗ trợ 4 môn tự nhiên:**
- Toán
- Vật lý  
- Hóa học
- Sinh học

💡 **Bạn có thể thử:**
- "Tạo đề Vật lý về Động lực học"
- "Ra đề kiểm tra Toán về Hệ bất phương trình"
- "Tạo đề Hóa học về Bảng tuần hoàn"
"""
                
                # ========== CHECK 2: No subject detected ==========
                if not quiz_info.get("subject"):
                    return """⚠️ Không xác định được môn học.

💡 **Các môn hỗ trợ:** Toán, Vật lý, Hóa học, Sinh học

**Ví dụ câu hỏi đúng:**
- "Tạo đề Toán về Hàm số bậc hai"
- "Đề kiểm tra Vật lý về Dao động điều hòa"
- "Ra 10 câu Hóa về Axit - Bazơ - Muối"
"""
                
                # ========== CHECK 3: Subject not in allowed list ==========
                detected_subject = quiz_info.get("subject")
                if detected_subject not in ALLOWED_QUIZ_SUBJECTS:
                    return f"""⚠️ Xin lỗi, hiện tại hệ thống chỉ hỗ trợ tạo đề cho **4 môn tự nhiên**.

🔍 **Phát hiện:** Bạn yêu cầu môn "{detected_subject}"

📚 **Các môn được hỗ trợ:**
✅ Toán
✅ Vật lý
✅ Hóa học
✅ Sinh học

💡 **Gợi ý:**
- "Tạo đề Toán về Hệ bất phương trình"
- "Tạo đề Vật lý về Động lực học"
- "Tạo đề Hóa học về Bảng tuần hoàn"
- "Tạo đề Sinh học về Quang hợp"

❓ Bạn có muốn tạo đề cho môn nào trong 4 môn trên không?"""
                
                # ========== VALID REQUEST - Proceed ==========
                print(f"   📚 Môn: {quiz_info['subject']}")
                print(f"   📖 Chủ đề: {quiz_info['topic']}")
                
                # Check if user specified difficulty in query
                user_difficulty = quiz_info.get("user_difficulty")
                
                if user_difficulty:
                    print(f"   🎯 Độ khó user chỉ định: {user_difficulty}")
                    use_student_difficulty = False
                else:
                    print(f"   🎯 Sử dụng độ khó từ profile")
                    use_student_difficulty = True
                
                # Generate quiz
                result = self.quiz_generator.generate_quiz(
                    subject=quiz_info["subject"],
                    topic=quiz_info["topic"],
                    difficulty=user_difficulty,
                    use_student_difficulty=use_student_difficulty
                )
                
                if result["success"]:
                    # Save to storage
                    try:
                        # Get student_id from profile
                        student_id = "unknown"
                        if self.quiz_generator.student_profile:
                            student_id = self.quiz_generator.student_profile.get("_id", "unknown")
                        
                        # Check if has answer_key
                        if not result.get("answer_key"):
                            print("   ⚠️ Thiếu answer_key!")
                            return "❌ Lỗi: Không thể tạo đề vì thiếu đáp án. Vui lòng thử lại."
                        
                        # Save to storage WITH answer_key
                        quiz_id = self.quiz_storage.save_quiz(
                            student_id=student_id,
                            content=result['quiz_markdown'],
                            answer_key=result['answer_key'],
                            subject=quiz_info["subject"],
                            topic=quiz_info["topic"],
                            difficulty=result["metadata"]["difficulty"]
                        )
                        
                        print(f"✅ Đã lưu vào database với ID: {quiz_id}")
                    except Exception as e:
                        print(f"⚠️ Không thể lưu quiz: {e}")
                    
                    # Return markdown directly
                    return f"""✅ Đã tạo xong đề kiểm tra!

{result['quiz_markdown']}

---

💡 **Lưu ý**: 
- Đề kiểm tra được tạo bởi AI, vui lòng kiểm tra kỹ trước khi sử dụng
- Bạn có thể yêu cầu tạo đề khác với độ khó hoặc số câu khác nhau
"""
                else:
                    return f"""❌ Không thể tạo đề kiểm tra: {result['error']}

💡 Vui lòng thử lại hoặc cung cấp thông tin rõ ràng hơn."""
            
            # Check if graph request
            if self._should_draw_graph(user_query):
                print("\n📊 Phát hiện yêu cầu vẽ đồ thị!")
                
                # Extract equation
                equation = self._extract_equation(user_query)
                
                if not equation:
                    return "⚠️ Không thể xác định hàm số cần vẽ. Vui lòng nhập rõ hơn (VD: 'vẽ đồ thị y = x**2')"
                
                print(f"   📝 Equation: y = {equation}")
                
                # Extract range
                x_min, x_max = extract_range_from_query(user_query)
                print(f"   📏 Range: [{x_min}, {x_max}]")
                
                # Generate graph
                result = self.graph_generator.generate_graph(equation, x_min, x_max)
                
                if result["success"]:
                    return f"""✅ Đã vẽ xong đồ thị!

📊 Thông tin:
- Hàm số: y = {equation}
- Khoảng giá trị: x ∈ [{x_min}, {x_max}]
- File: {result['file_path']}
- Kích thước: {result['file_size']/1024:.1f}KB

[IMAGE:{result['file_path']}]

💡 Bạn có muốn tôi giải thích gì về đồ thị này không?"""
                else:
                    return f"""❌ Không thể vẽ đồ thị: {result['error']}

💡 Gợi ý:
- Kiểm tra cú pháp hàm số (VD: x**2, sin(x), 2*x + 3)
- Đảm bảo hàm số hợp lệ trong khoảng [{x_min}, {x_max}]
- Thử lại với hàm số đơn giản hơn"""
            
            # Decide if should use search tool
            should_search = self._should_use_tool(user_query)
            
            if should_search:
                print("\n🔧 Quyết định: Sử dụng tool search_questions")
                
                # Use tool
                tool_result = search_questions_tool(
                    user_query,
                    self.intent_classifier,
                    self.retriever
                )
                
                # Generate final response with tool result
                messages = [
                    {
                        "role": "system",
                        "content": self._get_system_prompt(mode="search")
                    },
                    {
                        "role": "user",
                        "content": f"Câu hỏi của học sinh: {user_query}\n\nKết quả tìm kiếm:\n{tool_result}\n\nHãy trả lời câu hỏi dựa trên kết quả trên."
                    }
                ]
            else:
                print("\n💬 Quyết định: Trả lời trực tiếp (không cần search)")
                
                # Direct response without tool
                messages = [
                    {
                        "role": "system",
                        "content": self._get_system_prompt(mode="general")
                    },
                    {
                        "role": "user",
                        "content": user_query
                    }
                ]
            
            # Get LLM response
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"⚠️ Lỗi xử lý câu hỏi: {str(e)}"

# ================== RAG SYSTEM ==================
class ScienceQASystem:
    """Main RAG system for science Q&A"""
    
    def __init__(self):
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize components
        self.intent_classifier = IntentClassifier(self.client)
        self.retriever = QuestionRetriever(self.client, QDRANT_PATH, COLLECTION_NAME)
        self.agent = SimpleAgent(self.client, self.intent_classifier, self.retriever)
    
    def query(self, user_query: str) -> str:
        """Process user query through RAG system"""
        return self.agent.query(user_query)

# ================== DISPLAY HELPER ==================
def display_response(response: str):
    """Display response with image support"""
    
    # Check for image tag
    image_pattern = r'\[IMAGE:(.+?)\]'
    match = re.search(image_pattern, response)
    
    if match:
        img_path = match.group(1)
        
        # Remove image tag from text
        text = response.replace(match.group(0), '')
        print(text)
        
        # Try to open image
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', img_path], check=False)
            elif platform.system() == 'Linux':
                subprocess.run(['xdg-open', img_path], check=False)
            elif platform.system() == 'Windows':
                os.startfile(img_path)
            
            print(f"\n🖼️  Đã mở ảnh: {img_path}")
        except Exception as e:
            print(f"\n⚠️  Không thể mở ảnh tự động: {e}")
            print(f"   Vui lòng mở file: {img_path}")
    else:
        print(response)


# ================== MAIN CLI ==================
def main():
    print("=" * 70)
    print("HỆ THỐNG RAG - TRỢ LÝ HỌC TẬP MÔN TỰ NHIÊN")
    print("=" * 70)
    print("Môn học hỗ trợ: Toán, Lý, Hóa, Sinh")
    print("✨ Tính năng: Vẽ đồ thị + Tạo đề kiểm tra + Chấm điểm tự động")
    print("Gõ 'exit' hoặc 'quit' để thoát")
    print("=" * 70)
    
    # Initialize system
    print("\n🔧 Đang khởi tạo hệ thống...")
    try:
        rag_system = ScienceQASystem()
        print("✅ Hệ thống sẵn sàng!\n")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        return
    
    # Show examples
    print("💡 Ví dụ câu hỏi:")
    print("   - Định luật Newton là gì?")
    print("   - Vẽ đồ thị y = x**2")
    print("   - Vẽ đồ thị sin(x) từ -5 đến 5")
    print("   - Tạo đề kiểm tra Vật lý về Động lực học")
    print("   - Tạo đề Toán về Hệ bất phương trình")
    print("   - Hàm bậc hai có tính chất gì?\n")
    
    # Interactive loop
    while True:
        try:
            user_input = input("\n🎓 Học sinh: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'thoát']:
                print("\n👋 Tạm biệt! Chúc bạn học tốt!")
                break
            
            # Process query
            response = rag_system.query(user_input)
            
            print(f"\n🤖 Trợ lý:")
            display_response(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"\n⚠️ Lỗi: {e}")

if __name__ == "__main__":
    main()