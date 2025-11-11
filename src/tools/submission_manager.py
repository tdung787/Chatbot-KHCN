"""
tools/submission_manager.py

Quản lý việc nộp bài và chấm điểm
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class SubmissionManager:
    """Manage quiz submissions and grading"""
    
    def __init__(self, db_path: str = "database/quiz_storage.db"):
        self.db_path = Path(db_path)
        self._ensure_table()
    
    def _ensure_table(self):
        """Create submissions table if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id TEXT PRIMARY KEY,
                quiz_id TEXT NOT NULL,
                student_id TEXT NOT NULL,
                student_answers TEXT NOT NULL,
                score REAL NOT NULL,
                daily_count INTEGER NOT NULL,
                submitted_at TEXT NOT NULL,
                FOREIGN KEY (quiz_id) REFERENCES quizzes(id)
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_submissions_student 
            ON submissions(student_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_submissions_quiz 
            ON submissions(quiz_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_submissions_date 
            ON submissions(submitted_at)
        """)
        
        conn.commit()
        conn.close()
    
    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def _get_today_submission_count(self, student_id: str) -> int:
        """Count submissions today"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        cursor.execute("""
            SELECT COUNT(*) FROM submissions
            WHERE student_id = ? AND submitted_at LIKE ?
        """, (student_id, f"{today}%"))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def grade_submission(self, quiz_id: str, student_answers: str, answer_key: str) -> float:
        """
        Grade submission by comparing with answer key
        
        Args:
            quiz_id: Quiz ID
            student_answers: "1-A,2-B,3-C,..."
            answer_key: "1-A,2-B,3-D,..." (correct answers)
            
        Returns:
            Score (0.0 - 10.0)
        """
        try:
            # Parse answer key
            correct_dict = {}
            for item in answer_key.split(','):
                parts = item.strip().split('-')
                if len(parts) == 2:
                    num, ans = parts
                    correct_dict[int(num)] = ans.strip().upper()
            
            # Parse student answers
            student_dict = {}
            for item in student_answers.split(','):
                parts = item.strip().split('-')
                if len(parts) == 2:
                    num, ans = parts
                    student_dict[int(num)] = ans.strip().upper()
            
            # Compare and count correct answers
            correct_count = 0
            for i in range(1, 11):  # 10 questions
                if student_dict.get(i) == correct_dict.get(i):
                    correct_count += 1
            
            # Score: 1 point per question
            score = float(correct_count)
            
            return score
            
        except Exception as e:
            print(f"⚠️ Grading error: {e}")
            return 0.0
    
    def submit_quiz(
        self,
        quiz_id: str,
        student_id: str,
        student_answers: str,
        answer_key: str
    ) -> Dict:
        """
        Submit quiz and auto-grade
        
        Args:
            quiz_id: Quiz ID
            student_id: Student ID
            student_answers: "1-A,2-B,3-C,..."
            answer_key: Correct answers from quiz
            
        Returns:
            Submission result with score
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 1. Calculate daily_count
            daily_count = self._get_today_submission_count(student_id) + 1
            
            # 2. Generate submission_id
            now = datetime.now()
            today = now.strftime("%Y%m%d")
            submission_id = f"sub_{today}_{daily_count:03d}"
            
            # 3. Grade submission
            score = self.grade_submission(quiz_id, student_answers, answer_key)
            
            # 4. Save to database
            cursor.execute("""
                INSERT INTO submissions (
                    id, quiz_id, student_id, student_answers,
                    score, daily_count, submitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                submission_id,
                quiz_id,
                student_id,
                student_answers,
                score,
                daily_count,
                now.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Đã lưu bài nộp: {submission_id} - Điểm: {score}/10")
            
            return {
                "success": True,
                "submission_id": submission_id,
                "score": score,
                "total": 10.0,
                "percentage": (score / 10.0) * 100,
                "daily_count": daily_count,
                "submitted_at": now.isoformat()
            }
            
        except Exception as e:
            print(f"⚠️ Submit error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_submission(self, submission_id: str) -> Optional[Dict]:
        """Get submission by ID"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM submissions WHERE id = ?
        """, (submission_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_submission_with_details(self, submission_id: str, answer_key: str) -> Optional[Dict]:
        """
        Get submission with detailed comparison
        
        Args:
            submission_id: Submission ID
            answer_key: Correct answers from quiz
            
        Returns:
            Submission with details array showing correct/incorrect per question
        """
        submission = self.get_submission(submission_id)
        if not submission:
            return None
        
        # Parse answers
        correct_dict = {}
        for item in answer_key.split(','):
            parts = item.strip().split('-')
            if len(parts) == 2:
                num, ans = parts
                correct_dict[int(num)] = ans.strip().upper()
        
        student_dict = {}
        for item in submission['student_answers'].split(','):
            parts = item.strip().split('-')
            if len(parts) == 2:
                num, ans = parts
                student_dict[int(num)] = ans.strip().upper()
        
        # Build details
        details = []
        for i in range(1, 11):
            correct_ans = correct_dict.get(i, "?")
            student_ans = student_dict.get(i, "?")
            is_correct = (student_ans == correct_ans)
            
            details.append({
                "question_number": i,
                "correct_answer": correct_ans,
                "student_answer": student_ans,
                "is_correct": is_correct,
                "points": 1.0 if is_correct else 0.0
            })
        
        submission['details'] = details
        submission['correct_count'] = sum(1 for d in details if d['is_correct'])
        submission['incorrect_count'] = sum(1 for d in details if not d['is_correct'])
        
        return submission
    
    def get_student_submissions(
        self,
        student_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict]:
        """Get submissions by student"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM submissions
            WHERE student_id = ?
            ORDER BY submitted_at DESC
            LIMIT ? OFFSET ?
        """, (student_id, limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def check_quiz_submitted(self, quiz_id: str, student_id: str) -> bool:
        """Check if student already submitted this quiz"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM submissions
            WHERE quiz_id = ? AND student_id = ?
        """, (quiz_id, student_id))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0