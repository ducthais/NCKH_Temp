"""Batch document parsing script"""
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.sectioner import split_sections
from src.nlp.extractor import extract_entities


def parse_cv_file(cv_path):
    """Parse single CV and extract structured data"""
    with open(cv_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    sections = split_sections(text)
    entities = extract_entities(text, sections)
    
    return {
        'filename': cv_path.name,
        'raw_text': text,
        'sections': sections,
        'entities': entities
    }


def main():
    """Parse all CVs in batch and save as JSONL"""
    print("[*] Bắt đầu parse batch CVs...")
    
    cv_dir = Path('data/raw/cv')
    output_file = Path('data/interim/cvs.jsonl')
    
    if not cv_dir.exists():
        print(f"[!] Không tìm thấy thư mục CV: {cv_dir}")
        return
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    parsed_count = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for cv_file in sorted(cv_dir.glob('*.txt')):
            try:
                print(f"  Parsing: {cv_file.name}")
                cv_data = parse_cv_file(cv_file)
                out_f.write(json.dumps(cv_data, ensure_ascii=False) + '\n')
                parsed_count += 1
            except Exception as e:
                print(f"  [!] Lỗi parse {cv_file.name}: {e}")
    
    print(f"[✓] Đã parse {parsed_count} CVs → {output_file}")


if __name__ == "__main__":
    main()
