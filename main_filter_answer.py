from src.utils.filter_answer import filter_pages_with_answers

def main():
    # Đường dẫn đến 3 thư mục
    questions_folder = "data/input/cleaned_text"
    answers_folder = "data/input/normalized_answers"
    output_folder = "data/output/"

    filter_pages_with_answers(questions_folder, answers_folder, output_folder)

if __name__ == "__main__":
    main()
