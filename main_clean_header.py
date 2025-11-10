from src.utils.clean_headers_footers import remove_headers_and_footers

if __name__ == "__main__":
    input_dir = "data/input/text"
    output_dir = "data/output/cleaned_text"

    remove_headers_and_footers(input_dir, output_dir)
