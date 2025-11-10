from src.utils.assign_answers import assign_answers_with_ai

def main():
    # ====== ĐỊNH NGHĨA INPUT / OUTPUT ======
    summary_json = "data/output/page_comparison_summary.json"          # file JSON từ bước filter
    questions_folder = "data/input/truncated_text"                       # thư mục câu hỏi
    answers_folder = "data/input/normalized_answers"                   # thư mục đáp án
    output_folder = "data/output/assigned_answers"                          # nơi lưu file sau khi gán

    # ====== GỌI HÀM GÁN ĐÁP ÁN BẰNG AI ======
    assign_answers_with_ai(
        summary_json=summary_json,
        questions_folder=questions_folder,
        answers_folder=answers_folder,
        output_folder=output_folder
    )

if __name__ == "__main__":
    main()
