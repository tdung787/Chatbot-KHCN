from src.utils.text_normalizer import VietnameseTextNormalizer

def main():
    normalizer = VietnameseTextNormalizer(use_llm=True)
    
    # Chọn file input
    input_folder = "data/input/answers"
    output_json = "data/output/answers_normalized.json"
    output_txt_folder = "data/output/normalized_answers" 
    
    # Chuẩn hóa
    normalizer.normalize_folder(
        input_folder=input_folder,
        output_json=output_json,
        output_txt_folder=output_txt_folder, 
        method='hybrid' #method: 'regex' | 'llm' | 'hybrid'
    )

if __name__ == "__main__":
    main()