"""
FastAPI application for Quiz Management System

Provides simple REST API for accessing quiz history
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, List
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.quiz_storage import QuizStorage
from src.tools.submission_manager import SubmissionManager


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


# ==================== RUN INFO ====================
if __name__ == "__main__":
    import uvicorn
    print("⚠️  Don't run this file directly!")
    print("👉 Use: python run_api.py")