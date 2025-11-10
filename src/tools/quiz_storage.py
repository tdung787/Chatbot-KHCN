"""
src/tools/quiz_storage.py

SQLite-based storage for quiz history
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json


class QuizStorage:
    """Manage quiz storage in SQLite database"""
    
    def __init__(self, db_path: str = "database/quiz_storage.db"):  # ← ĐỔI PATH
        self.db_path = Path(db_path)
        self._ensure_database()
    
    def _ensure_database(self):
        """Create database and tables if not exist"""
        # Create database directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create quizzes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quizzes (
                id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                date TEXT NOT NULL,
                date_count INTEGER NOT NULL,
                content TEXT NOT NULL,
                subject TEXT,
                topic TEXT,
                difficulty TEXT,
                num_questions INTEGER DEFAULT 10,
                time_limit INTEGER DEFAULT 15,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_student_id 
            ON quizzes(student_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_date 
            ON quizzes(date)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_subject 
            ON quizzes(subject)
        """)
        
        conn.commit()
        conn.close()
    
    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def _get_today_count(self, student_id: str) -> int:
        """Get count of quizzes created today by this student"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        cursor.execute("""
            SELECT COUNT(*) FROM quizzes
            WHERE student_id = ? AND date LIKE ?
        """, (student_id, f"{today}%"))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def save_quiz(
        self,
        student_id: str,
        content: str,
        subject: str = None,
        topic: str = None,
        difficulty: str = None,
        num_questions: int = 10,
        time_limit: int = 15
    ) -> str:
        """
        Save quiz to database
        
        Args:
            student_id: Student ID
            content: Quiz markdown content
            subject: Subject name (optional)
            topic: Topic name (optional)
            difficulty: Difficulty level (optional)
            num_questions: Number of questions (default: 10)
            time_limit: Time limit in minutes (default: 15)
            
        Returns:
            Quiz ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get today's count for this student
        date_count = self._get_today_count(student_id) + 1
        
        # Generate ID: quiz_YYYYMMDD_XXX
        now = datetime.now()
        today = now.strftime("%Y%m%d")
        quiz_id = f"quiz_{today}_{date_count:03d}"
        
        # Insert quiz
        cursor.execute("""
            INSERT INTO quizzes (
                id, student_id, date, date_count, content,
                subject, topic, difficulty, num_questions, time_limit
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            quiz_id,
            student_id,
            now.isoformat(),
            date_count,
            content,
            subject,
            topic,
            difficulty,
            num_questions,
            time_limit
        ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Đã lưu đề thi: {quiz_id}")
        return quiz_id
    
    def get_quiz(self, quiz_id: str) -> Optional[Dict]:
        """Get quiz by ID"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row  # Return dict-like rows
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM quizzes WHERE id = ?
        """, (quiz_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_student_quizzes(
        self, 
        student_id: str, 
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict]:
        """Get recent quizzes by student (paginated)"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM quizzes
            WHERE student_id = ?
            ORDER BY date DESC
            LIMIT ? OFFSET ?
        """, (student_id, limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_today_quizzes(self, student_id: str) -> List[Dict]:
        """Get all quizzes created today by student"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        cursor.execute("""
            SELECT * FROM quizzes
            WHERE student_id = ? AND date LIKE ?
            ORDER BY date_count ASC
        """, (student_id, f"{today}%"))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_quizzes_by_filter(
        self,
        student_id: str = None,
        subject: str = None,
        difficulty: str = None,
        date_from: str = None,
        date_to: str = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """Get quizzes with filters (for API endpoints)"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build dynamic query
        query = "SELECT * FROM quizzes WHERE 1=1"
        params = []
        
        if student_id:
            query += " AND student_id = ?"
            params.append(student_id)
        
        if subject:
            query += " AND subject = ?"
            params.append(subject)
        
        if difficulty:
            query += " AND difficulty = ?"
            params.append(difficulty)
        
        if date_from:
            query += " AND date >= ?"
            params.append(date_from)
        
        if date_to:
            query += " AND date <= ?"
            params.append(date_to)
        
        query += " ORDER BY date DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_stats(self, student_id: str = None) -> Dict:
        """Get statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Total quizzes
        if student_id:
            cursor.execute("""
                SELECT COUNT(*) FROM quizzes WHERE student_id = ?
            """, (student_id,))
        else:
            cursor.execute("SELECT COUNT(*) FROM quizzes")
        
        total = cursor.fetchone()[0]
        
        # By subject
        if student_id:
            cursor.execute("""
                SELECT subject, COUNT(*) as count
                FROM quizzes
                WHERE student_id = ? AND subject IS NOT NULL
                GROUP BY subject
            """, (student_id,))
        else:
            cursor.execute("""
                SELECT subject, COUNT(*) as count
                FROM quizzes
                WHERE subject IS NOT NULL
                GROUP BY subject
            """)
        
        by_subject = {row[0]: row[1] for row in cursor.fetchall()}
        
        # By difficulty
        if student_id:
            cursor.execute("""
                SELECT difficulty, COUNT(*) as count
                FROM quizzes
                WHERE student_id = ? AND difficulty IS NOT NULL
                GROUP BY difficulty
            """, (student_id,))
        else:
            cursor.execute("""
                SELECT difficulty, COUNT(*) as count
                FROM quizzes
                WHERE difficulty IS NOT NULL
                GROUP BY difficulty
            """)
        
        by_difficulty = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            "total_quizzes": total,
            "by_subject": by_subject,
            "by_difficulty": by_difficulty
        }
    
    def delete_quiz(self, quiz_id: str) -> bool:
        """Delete quiz by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM quizzes WHERE id = ?", (quiz_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted
    
    def count_total(self, student_id: str = None) -> int:
        """Count total quizzes"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if student_id:
            cursor.execute("""
                SELECT COUNT(*) FROM quizzes WHERE student_id = ?
            """, (student_id,))
        else:
            cursor.execute("SELECT COUNT(*) FROM quizzes")
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count