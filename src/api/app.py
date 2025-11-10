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
                    "date_counts": [],
                    "subjects": []
                }
            
            daily_stats[date]["count"] += 1
            daily_stats[date]["date_counts"].append(quiz["date_count"])
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
        
        # Sort by date_count
        quizzes_on_date.sort(key=lambda x: x["date_count"])
        
        return {
            "success": True,
            "date": date,
            "student_id": student_id,
            "count": len(quizzes_on_date),
            "data": quizzes_on_date
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