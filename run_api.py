"""
Script to run the Quiz Management API

Usage:
    python run_api.py
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 STARTING QUIZ MANAGEMENT API")
    print("=" * 70)
    print()
    print("📍 API will be available at:")
    print("   • Main: http://localhost:8000")
    print("   • Docs: http://localhost:8000/docs")
    print("   • Health: http://localhost:8000/health")
    print()
    print("📚 Available endpoints:")
    print("   • GET /api/quiz/latest - Bài kiểm tra mới nhất")
    print("   • GET /api/quiz/all - Tất cả bài kiểm tra")
    print("   • GET /api/quiz/{quiz_id} - Chi tiết 1 bài")
    print("   • GET /api/quiz/daily-count - Thống kê theo ngày")
    print("   • GET /api/quiz/by-date - Lấy bài theo ngày cụ thể")
    print("   • GET /api/stats - Thống kê")
    print()
    print("⌨️  Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload khi code thay đổi
        log_level="info"
    )