"""
FastAPI application for Quiz Management System

Provides simple REST API for accessing quiz history
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, List
import sys
import os
import requests
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.quiz_storage import QuizStorage
from src.tools.submission_manager import SubmissionManager
from query import ScienceQASystem
from src.tools.session_manager import SessionManager
from src.tools.chat_history_manager import ChatHistoryManager

# ==================== EXTERNAL API CONFIG ====================
EXTERNAL_API_BASE_URL = os.getenv("EXTERNAL_API_BASE_URL", "https://v5bfv7qs-3001.asse.devtunnels.ms")

# ==================== HELPER FUNCTIONS ====================
def validate_student_id(student_id: str) -> Dict:
    """
    Validate student_id against external API
    
    Args:
        student_id: Student ID to validate
        
    Returns:
        {
            "is_valid": bool,
            "student_info": dict or None,
            "error": str or None
        }
    """
    try:
        # Call external API
        url = f"{EXTERNAL_API_BASE_URL}/api/public/rag/students"
        
        print(f"   🔍 Validating student_id: {student_id}")
        print(f"   🌐 Calling: {url}")
        
        response = requests.get(url, timeout=5)
        
        if response.status_code != 200:
            return {
                "is_valid": False,
                "student_info": None,
                "error": f"API returned status {response.status_code}"
            }
        
        data = response.json()
        
        if not data.get("success"):
            return {
                "is_valid": False,
                "student_info": None,
                "error": "API returned success=false"
            }
        
        # Find student in list
        students = data.get("data", {}).get("students", [])
        
        student = next((s for s in students if s["_id"] == student_id), None)
        
        if student:
            print(f"   ✅ Student found: {student['user_id']['full_name']}")
            return {
                "is_valid": True,
                "student_info": student,
                "error": None
            }
        else:
            print(f"   ❌ Student not found in list")
            return {
                "is_valid": False,
                "student_info": None,
                "error": f"Student ID {student_id} not found"
            }
        
    except requests.exceptions.Timeout:
        print(f"   ⚠️ API timeout")
        return {
            "is_valid": False,
            "student_info": None,
            "error": "External API timeout"
        }
    except requests.exceptions.RequestException as e:
        print(f"   ⚠️ API error: {e}")
        return {
            "is_valid": False,
            "student_info": None,
            "error": f"External API error: {str(e)}"
        }
    except Exception as e:
        print(f"   ⚠️ Validation error: {e}")
        return {
            "is_valid": False,
            "student_info": None,
            "error": f"Validation error: {str(e)}"
        }


# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Quiz Management API",
    description="API để quản lý đề kiểm tra trắc nghiệm",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins (production nên restrict)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize storage
storage = QuizStorage()
submission_manager = SubmissionManager()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
session_manager = SessionManager(openai_client=openai_client)
chat_history_manager = ChatHistoryManager()
print("✅ Session managers initialized")

# Initialize RAG system
try:
    rag_system = ScienceQASystem()
except Exception as e:
    print(f"⚠️  Warning: RAG system initialization failed: {e}")
    rag_system = None


# ==================== HEALTH CHECK ====================
@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Quiz Management API is running",
        "endpoints": {
            "latest": "/api/quiz/latest",
            "all": "/api/quiz/all",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health():
    """Detailed health check"""
    total = storage.count_total()
    return {
        "status": "healthy",
        "database": "connected",
        "total_quizzes": total
    }


# ==================== API 1: LATEST QUIZ ====================
@app.get("/api/quiz/latest")
def get_latest_quiz(
    student_id: Optional[str] = Query(None, description="Student ID to filter by")
) -> Dict:
    """
    Lấy bài kiểm tra mới nhất
    
    Args:
        student_id: Optional - Lọc theo student ID
        
    Returns:
        Bài kiểm tra mới nhất hoặc error
    """
    try:
        if student_id:
            # Get latest quiz for specific student
            quizzes = storage.get_student_quizzes(student_id, limit=1, offset=0)
            
            if not quizzes:
                return {
                    "success": False,
                    "message": f"Không tìm thấy đề kiểm tra cho student_id: {student_id}"
                }
            
            return {
                "success": True,
                "data": quizzes[0]
            }
        else:
            # Get latest quiz overall
            quizzes = storage.get_quizzes_by_filter(limit=1, offset=0)
            
            if not quizzes:
                return {
                    "success": False,
                    "message": "Chưa có đề kiểm tra nào trong hệ thống"
                }
            
            return {
                "success": True,
                "data": quizzes[0]
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ==================== API 2: ALL QUIZZES ====================
@app.get("/api/quiz/all")
def get_all_quizzes(
    student_id: Optional[str] = Query(None, description="Filter by student ID"),
    subject: Optional[str] = Query(None, description="Filter by subject (Toán, Vật lý, Hóa học, Sinh học)"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty (dễ, trung bình, khó)"),
    date_from: Optional[str] = Query(None, description="Filter from date (ISO format: 2025-01-01)"),
    date_to: Optional[str] = Query(None, description="Filter to date (ISO format: 2025-01-31)"),
    page: int = Query(1, ge=1, description="Page number (starts from 1)"),
    size: int = Query(20, ge=1, le=100, description="Items per page (max 100)")
) -> Dict:
    """
    Lấy tất cả bài kiểm tra với phân trang và filter
    
    Args:
        student_id: Optional - Lọc theo student
        subject: Optional - Lọc theo môn học
        difficulty: Optional - Lọc theo độ khó
        date_from: Optional - Lọc từ ngày
        date_to: Optional - Lọc đến ngày
        page: Page number (default: 1)
        size: Items per page (default: 20, max: 100)
        
    Returns:
        Paginated list of quizzes
    """
    try:
        # Calculate offset
        offset = (page - 1) * size
        
        # Get filtered quizzes
        quizzes = storage.get_quizzes_by_filter(
            student_id=student_id,
            subject=subject,
            difficulty=difficulty,
            date_from=date_from,
            date_to=date_to,
            limit=size,
            offset=offset
        )
        
        # Get total count with same filters
        # Note: Need to count with filters, not just count_total()
        # We'll get total by querying without limit
        all_filtered = storage.get_quizzes_by_filter(
            student_id=student_id,
            subject=subject,
            difficulty=difficulty,
            date_from=date_from,
            date_to=date_to,
            limit=999999,  # Large number to get all
            offset=0
        )
        total = len(all_filtered)
        
        # Calculate total pages
        total_pages = (total + size - 1) // size  # Ceiling division
        
        return {
            "success": True,
            "pagination": {
                "total": total,
                "page": page,
                "size": size,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            },
            "filters": {
                "student_id": student_id,
                "subject": subject,
                "difficulty": difficulty,
                "date_from": date_from,
                "date_to": date_to
            },
            "data": quizzes
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
# ==================== DAILY COUNT ====================
@app.get("/api/quiz/daily-count")
def get_daily_count(
    student_id: str = Query(..., description="Student ID (required)")
) -> Dict:
    """
    Đếm số lần học sinh làm bài theo từng ngày
    
    Args:
        student_id: Student ID (bắt buộc)
        
    Returns:
        Thống kê số bài theo ngày
    """
    try:
        # Get all quizzes of student
        all_quizzes = storage.get_student_quizzes(student_id, limit=9999, offset=0)
        
        # Group by date
        daily_stats = {}
        for quiz in all_quizzes:
            date = quiz["date"].split("T")[0]  # Extract YYYY-MM-DD
            
            if date not in daily_stats:
                daily_stats[date] = {
                    "date": date,
                    "count": 0,
                    "daily_counts": [],
                    "subjects": []
                }
            
            daily_stats[date]["count"] += 1
            daily_stats[date]["daily_counts"].append(quiz["daily_count"])
            daily_stats[date]["subjects"].append(quiz.get("subject"))
        
        # Convert to list and sort by date descending
        daily_list = sorted(
            daily_stats.values(), 
            key=lambda x: x["date"], 
            reverse=True
        )
        
        # Calculate summary
        from datetime import datetime
        today_date = datetime.now().strftime("%Y-%m-%d")
        today_count = daily_stats.get(today_date, {}).get("count", 0)
        
        return {
            "success": True,
            "student_id": student_id,
            "total_days": len(daily_list),
            "total_quizzes": len(all_quizzes),
            "today": {
                "date": today_date,
                "count": today_count
            },
            "daily_breakdown": daily_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/api/quiz/by-date")
def get_quizzes_by_date(
    student_id: str = Query(..., description="Student ID (required)"),
    date: str = Query(..., description="Date in YYYY-MM-DD format (e.g., 2025-01-10)")
) -> Dict:
    """
    Lấy tất cả bài kiểm tra của 1 ngày cụ thể
    
    Args:
        student_id: Student ID (bắt buộc)
        date: Ngày cần lấy (YYYY-MM-DD)
        
    Returns:
        Danh sách quiz của ngày đó
    """
    try:
        # Get all quizzes and filter by date
        all_quizzes = storage.get_student_quizzes(student_id, limit=9999, offset=0)
        
        quizzes_on_date = [
            q for q in all_quizzes 
            if q["date"].startswith(date)
        ]
        
        # Sort by daily_count
        quizzes_on_date.sort(key=lambda x: x["daily_count"])
        
        return {
            "success": True,
            "date": date,
            "student_id": student_id,
            "count": len(quizzes_on_date),
            "data": quizzes_on_date
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
# ==================== SESSION MANAGEMENT ENDPOINTS ====================

@app.post("/api/session/create")
def create_session(
    student_id: str = Query(..., description="Student ID (required)"),
    first_message: Optional[str] = Query(None, description="Optional first message to start conversation")
) -> Dict:
    """
    Create new chat session
    
    - Validates student_id against external API
    - If first_message provided: Create session + process message + get response
    - If no first_message: Create empty session with default name
    """
    try:
        # ========== VALIDATE STUDENT ID ==========
        validation = validate_student_id(student_id)
        
        if not validation["is_valid"]:
            raise HTTPException(
                status_code=404,
                detail=f"Student not found: {validation['error']}"
            )
        
        student_info = validation["student_info"]
        print(f"   ✅ Student validated: {student_info['user_id']['full_name']}")
        # =========================================
        
        # ========== CASE 1: WITH FIRST MESSAGE ==========
        if first_message:
            if not rag_system:
                raise HTTPException(
                    status_code=503,
                    detail="RAG system not initialized"
                )
            
            # Create session with LLM-generated name
            session_result = session_manager.create_session(
                student_id=student_id,
                first_message=first_message
            )
            
            if not session_result["success"]:
                raise HTTPException(
                    status_code=500,
                    detail=session_result.get("error", "Failed to create session")
                )
            
            session = session_result["session"]
            
            print(f"   ✨ Created session: {session['id']} - {session['name']}")
            
            # Process first message and get response
            response = rag_system.query(first_message, conversation_history=[])
            
            # Save messages to session
            try:
                # Save user message
                chat_history_manager.save_message(
                    session_id=session['id'],
                    role="user",
                    content=first_message
                )
                
                # Save assistant response
                chat_history_manager.save_message(
                    session_id=session['id'],
                    role="assistant",
                    content=response
                )
                
                # Update message count
                new_count = chat_history_manager.get_message_count(session['id'])
                session_manager.update_session(
                    session_id=session['id'],
                    message_count=new_count
                )
                
                print(f"   💾 Saved initial messages (total: {new_count})")
                
            except Exception as e:
                print(f"   ⚠️ Failed to save messages: {e}")
            
            return {
                "success": True,
                "session": session,
                "student_info": {
                    "id": student_info["_id"],
                    "name": student_info["user_id"]["full_name"],
                    "grade": student_info["grade_level"],
                    "class": student_info["current_class"]
                },
                "response": response,
                "has_first_message": True
            }
        
        # ========== CASE 2: EMPTY SESSION ==========
        else:
            # Create empty session with default name
            session_id = session_manager._generate_session_id(student_id)
            default_name = "Cuộc trò chuyện mới"
            
            from datetime import datetime
            now = datetime.now()
            
            conn = session_manager._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO chat_sessions (
                    id, student_id, name, first_message,
                    created_at, updated_at, message_count, is_archived
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                student_id,
                default_name,
                "",
                now.isoformat(),
                now.isoformat(),
                0,
                0
            ))
            
            conn.commit()
            conn.close()
            
            print(f"   ✨ Created empty session: {session_id}")
            
            return {
                "success": True,
                "session": {
                    "id": session_id,
                    "student_id": student_id,
                    "name": default_name,
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                    "message_count": 0
                },
                "student_info": {
                    "id": student_info["_id"],
                    "name": student_info["user_id"]["full_name"],
                    "grade": student_info["grade_level"],
                    "class": student_info["current_class"]
                },
                "response": None,
                "has_first_message": False
            }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session creation error: {str(e)}")

@app.get("/api/session/list")
def list_sessions(
    student_id: str = Query(..., description="Student ID (required)"),
    limit: int = Query(20, ge=1, le=100, description="Max sessions to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    include_archived: bool = Query(False, description="Include archived sessions")
) -> Dict:
    """
    List all sessions for a student
    
    Args:
        student_id: Student ID
        limit: Max sessions to return
        offset: Pagination offset
        include_archived: Include archived sessions
        
    Returns:
        List of sessions
    """
    try:
        sessions = session_manager.list_sessions(
            student_id=student_id,
            limit=limit,
            offset=offset,
            include_archived=include_archived
        )
        
        return {
            "success": True,
            "student_id": student_id,
            "count": len(sessions),
            "sessions": sessions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List sessions error: {str(e)}")


@app.get("/api/session/{session_id}")
def get_session(
    session_id: str,
    student_id: str = Query(..., description="Student ID for ownership verification")
) -> Dict:
    """
    Get session info
    
    Args:
        session_id: Session ID
        student_id: Student ID for verification
        
    Returns:
        Session info
    """
    try:
        # Verify ownership
        if not session_manager.verify_ownership(session_id, student_id):
            raise HTTPException(
                status_code=403,
                detail="Session not found or doesn't belong to you"
            )
        
        session = session_manager.get_session(session_id, student_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "success": True,
            "session": session
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get session error: {str(e)}")


@app.get("/api/session/{session_id}/history")
def get_session_history(
    session_id: str,
    student_id: str = Query(..., description="Student ID for ownership verification"),
    limit: Optional[int] = Query(None, description="Limit number of messages")
) -> Dict:
    """
    Get chat history for a session
    
    Args:
        session_id: Session ID
        student_id: Student ID for verification
        limit: Optional limit on number of messages
        
    Returns:
        Session info + chat history
    """
    try:
        # Verify ownership
        if not session_manager.verify_ownership(session_id, student_id):
            raise HTTPException(
                status_code=403,
                detail="Session not found or doesn't belong to you"
            )
        
        # Get session info
        session = session_manager.get_session(session_id, student_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get chat history
        messages = chat_history_manager.get_session_history(
            session_id=session_id,
            limit=limit
        )
        
        return {
            "success": True,
            "session": {
                "id": session['id'],
                "name": session['name'],
                "student_id": session['student_id'],
                "created_at": session['created_at'],
                "updated_at": session['updated_at'],
                "message_count": session['message_count']
            },
            "messages": messages
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get history error: {str(e)}")


@app.delete("/api/session/{session_id}")
def delete_session(
    session_id: str,
    student_id: str = Query(..., description="Student ID for ownership verification")
) -> Dict:
    """
    Delete a session and all its messages
    
    Args:
        session_id: Session ID to delete
        student_id: Student ID for verification
        
    Returns:
        Success message
    """
    try:
        # Delete session (will also delete messages via CASCADE)
        result = session_manager.delete_session(session_id, student_id)
        
        if not result["success"]:
            if "doesn't belong to you" in result.get("error", ""):
                raise HTTPException(status_code=403, detail=result["error"])
            else:
                raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "message": result["message"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete session error: {str(e)}")


@app.put("/api/session/{session_id}/rename")
def rename_session(
    session_id: str,
    student_id: str = Query(..., description="Student ID for ownership verification"),
    new_name: str = Query(..., description="New session name")
) -> Dict:
    """
    Rename a session
    
    Args:
        session_id: Session ID
        student_id: Student ID for verification
        new_name: New name for session
        
    Returns:
        Success message
    """
    try:
        # Verify ownership
        if not session_manager.verify_ownership(session_id, student_id):
            raise HTTPException(
                status_code=403,
                detail="Session not found or doesn't belong to you"
            )
        
        # Update session name
        session_manager.update_session(
            session_id=session_id,
            name=new_name
        )
        
        return {
            "success": True,
            "message": f"Session renamed to: {new_name}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rename session error: {str(e)}")


@app.post("/api/session/{session_id}/archive")
def archive_session(
    session_id: str,
    student_id: str = Query(..., description="Student ID for ownership verification")
) -> Dict:
    """
    Archive a session (soft delete)
    
    Args:
        session_id: Session ID
        student_id: Student ID for verification
        
    Returns:
        Success message
    """
    try:
        result = session_manager.archive_session(session_id, student_id)
        
        if not result["success"]:
            if "doesn't belong to you" in result.get("error", ""):
                raise HTTPException(status_code=403, detail=result["error"])
            else:
                raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "message": result["message"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Archive session error: {str(e)}")
    
# ==================== SUBMISSION ENDPOINTS ====================

@app.get("/api/quiz/current-status")
def get_current_quiz_status(
    student_id: str = Query(..., description="Student ID (required)")
) -> Dict:
    """
    Check if student has pending quiz
    
    Returns:
        Quiz info if pending, or null
    """
    try:
        pending_quiz = storage.get_latest_pending_quiz(student_id)
        
        if pending_quiz:
            return {
                "success": True,
                "has_pending": True,
                "quiz": {
                    "id": pending_quiz["id"],
                    "subject": pending_quiz.get("subject"),
                    "topic": pending_quiz.get("topic"),
                    "difficulty": pending_quiz.get("difficulty"),
                    "created_at": pending_quiz.get("date")
                }
            }
        else:
            return {
                "success": True,
                "has_pending": False,
                "quiz": None
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/api/submission/submit")
def submit_quiz(
    quiz_id: str = Query(..., description="Quiz ID"),
    student_id: str = Query(..., description="Student ID"),
    answers: str = Query(..., description="Student answers in format: 1-A,2-B,3-C,...")
) -> Dict:
    """
    Submit quiz and auto-grade
    
    Args:
        quiz_id: Quiz ID to submit
        student_id: Student ID
        answers: Format "1-A,2-B,3-C,4-D,5-A,6-B,7-C,8-D,9-A,10-B"
        
    Returns:
        Submission result with score
    """
    try:
        # 1. Check if quiz exists
        quiz = storage.get_quiz(quiz_id)
        if not quiz:
            raise HTTPException(status_code=404, detail=f"Quiz not found: {quiz_id}")
        
        # 2. Check if quiz belongs to student
        if quiz["student_id"] != student_id:
            raise HTTPException(status_code=403, detail="Quiz does not belong to this student")
        
        # 3. Check if quiz is pending
        if quiz.get("status") != "pending":
            raise HTTPException(status_code=400, detail="Quiz already submitted")
        
        # 4. Check if already submitted
        if submission_manager.check_quiz_submitted(quiz_id, student_id):
            raise HTTPException(status_code=400, detail="Quiz already submitted")
        
        # 5. Validate answers format
        if not answers or len(answers.split(',')) != 10:
            raise HTTPException(status_code=400, detail="Answers must have exactly 10 items (1-A,2-B,...)")
        
        # 6. Get answer key from quiz
        answer_key = quiz.get("answer_key")
        if not answer_key:
            raise HTTPException(status_code=500, detail="Quiz missing answer key")
        
        # 7. Submit and grade
        result = submission_manager.submit_quiz(
            quiz_id=quiz_id,
            student_id=student_id,
            student_answers=answers,
            answer_key=answer_key
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Submission failed"))
        
        # 8. Update quiz status to completed
        storage.update_quiz_status(quiz_id, "completed")
        
        return {
            "success": True,
            "message": "Đã nộp bài và chấm điểm thành công!",
            "submission_id": result["submission_id"],
            "score": result["score"],
            "total": result["total"],
            "percentage": result["percentage"],
            "daily_count": result["daily_count"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/api/submission/{submission_id}")
def get_submission(submission_id: str) -> Dict:
    """Get submission by ID (basic info)"""
    try:
        submission = submission_manager.get_submission(submission_id)
        
        if not submission:
            raise HTTPException(status_code=404, detail=f"Submission not found: {submission_id}")
        
        return {
            "success": True,
            "data": submission
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/api/submission/{submission_id}/result")
def get_submission_result(submission_id: str) -> Dict:
    """
    Get detailed submission result with correct/incorrect breakdown
    """
    try:
        # Get submission
        submission = submission_manager.get_submission(submission_id)
        
        if not submission:
            raise HTTPException(status_code=404, detail=f"Submission not found: {submission_id}")
        
        # Get quiz to get answer key
        quiz = storage.get_quiz(submission["quiz_id"])
        
        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found")
        
        # Get detailed result
        detailed = submission_manager.get_submission_with_details(
            submission_id,
            quiz["answer_key"]
        )
        
        return {
            "success": True,
            "submission_id": submission_id,
            "quiz_id": submission["quiz_id"],
            "student_id": submission["student_id"],
            "score": submission["score"],
            "total": 10.0,
            "percentage": (submission["score"] / 10.0) * 100,
            "correct_count": detailed["correct_count"],
            "incorrect_count": detailed["incorrect_count"],
            "submitted_at": submission["submitted_at"],
            "daily_count": submission["daily_count"],
            "details": detailed["details"],
            "quiz_info": {
                "subject": quiz.get("subject"),
                "topic": quiz.get("topic"),
                "difficulty": quiz.get("difficulty")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/api/submission/student/{student_id}")
def get_student_submissions(
    student_id: str,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
) -> Dict:
    """Get submission history for a student"""
    try:
        submissions = submission_manager.get_student_submissions(
            student_id,
            limit=limit,
            offset=offset
        )
        
        return {
            "success": True,
            "student_id": student_id,
            "count": len(submissions),
            "data": submissions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# ==================== BONUS: SINGLE QUIZ BY ID ====================
@app.get("/api/quiz/{quiz_id}")
def get_quiz_by_id(quiz_id: str) -> Dict:
    """
    Lấy chi tiết 1 bài kiểm tra theo ID
    
    Args:
        quiz_id: Quiz ID (e.g., quiz_20250110_001)
        
    Returns:
        Quiz details
    """
    try:
        quiz = storage.get_quiz(quiz_id)
        
        if not quiz:
            raise HTTPException(status_code=404, detail=f"Quiz not found: {quiz_id}")
        
        return {
            "success": True,
            "data": quiz
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ==================== BONUS: STATISTICS ====================
@app.get("/api/stats")
def get_statistics(
    student_id: Optional[str] = Query(None, description="Get stats for specific student")
) -> Dict:
    """
    Lấy thống kê
    
    Args:
        student_id: Optional - Stats for specific student
        
    Returns:
        Statistics data
    """
    try:
        stats = storage.get_stats(student_id=student_id)
        
        return {
            "success": True,
            "student_id": student_id,
            "data": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ==================== RAG QUERY ENDPOINT ====================
@app.post("/api/rag/query")
def rag_query(
    user_input: str = Query(..., description="User question or request"),
    session_id: str = Query(..., description="Session ID (required)"),
    student_id: Optional[str] = Query(None, description="Optional student ID for verification")
) -> Dict:
    """
    Query the RAG system within a session
    
    User must create a session first via POST /api/session/create
    
    Supports:
    - Answering questions about subjects
    - Creating quizzes
    - Drawing graphs
    - Submitting answers
    - General Q&A
    
    All interactions are saved to the session's chat history.
    
    Args:
        user_input: User's question or command
        session_id: Session ID (REQUIRED)
        student_id: Optional student ID for ownership verification
        
    Returns:
        RAG system response with session info
    """
    try:
        if not rag_system:
            raise HTTPException(
                status_code=503,
                detail="RAG system not initialized. Please check logs."
            )
        
        # ========== VALIDATE SESSION ==========
        # Get session info
        session = session_manager.get_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}"
            )
        
        # Verify ownership if student_id provided
        if student_id:
            if not session_manager.verify_ownership(session_id, student_id):
                raise HTTPException(
                    status_code=403,
                    detail="Session doesn't belong to you"
                )
        
        print(f"   📂 Using session: {session_id} - {session.get('name')}")
        
        # ========== LOAD CONVERSATION HISTORY ==========
        conversation_history = chat_history_manager.get_session_history(session_id)
        print(f"   📜 Loaded {len(conversation_history)} messages from history")
        
        # ========== PROCESS QUERY ==========
        response = rag_system.query(user_input, conversation_history)
        
        # ========== SAVE TO SESSION ==========
        try:
            # Save user message
            chat_history_manager.save_message(
                session_id=session_id,
                role="user",
                content=user_input
            )
            
            # Save assistant response
            chat_history_manager.save_message(
                session_id=session_id,
                role="assistant",
                content=response
            )
            
            # Update session metadata
            new_count = chat_history_manager.get_message_count(session_id)
            
            # ========== AUTO RENAME IF EMPTY SESSION ==========
            # Check if this is the first message in an empty session
            if session.get('first_message') == "" and new_count == 2:
                # This was an empty session, now has first message
                # Generate name using LLM based on user's first input
                new_name = session_manager._generate_session_name(user_input)
                
                # Update both message count AND name
                session_manager.update_session(
                    session_id=session_id,
                    message_count=new_count,
                    name=new_name
                )
                
                # Also update first_message field in database
                conn = session_manager._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE chat_sessions
                    SET first_message = ?
                    WHERE id = ?
                """, (user_input, session_id))
                conn.commit()
                conn.close()
                
                print(f"   🏷️  Auto-renamed empty session to: '{new_name}'")
            else:
                # Normal update - just update message count
                session_manager.update_session(
                    session_id=session_id,
                    message_count=new_count
                )
            # ================================================
            
            print(f"   💾 Saved messages to session (total: {new_count})")
            
        except Exception as e:
            print(f"   ⚠️ Failed to save to session: {e}")
            # Don't fail the request, just log
            
        except Exception as e:
            print(f"   ⚠️ Failed to save to session: {e}")
            # Don't fail the request, just log
        
        # ========== RETURN RESPONSE ==========
        return {
            "success": True,
            "user_input": user_input,
            "session": {
                "id": session['id'],
                "name": session.get('name'),
                "student_id": session.get('student_id'),
                "message_count": session.get('message_count', 0) + 2  # +2 for new messages
            },
            "response": response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query error: {str(e)}")
    
# ==================== RUN INFO ====================
if __name__ == "__main__":
    import uvicorn
    print("⚠️  Don't run this file directly!")
    print("👉 Use: python run_api.py")