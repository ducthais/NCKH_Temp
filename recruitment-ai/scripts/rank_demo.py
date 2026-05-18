"""Ranking demonstration script"""
import json
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.sectioner import split_sections
from src.nlp.extractor import extract_entities


def calculate_match_score(cv_data, jd_text):
    """Calculate match score between CV and JD"""
    # Extract skills from both
    cv_skills = set(cv_data.get('entities', {}).get('skills_raw', []))
    
    jd_sections = split_sections(jd_text)
    jd_skills_raw = jd_sections.get('SKILLS', jd_text).lower().split(',')
    jd_skills = {s.strip() for s in jd_skills_raw if s.strip()}
    
    # Calculate overlap
    if not jd_skills:
        return 0.0
    
    overlap = len(cv_skills & jd_skills)
    score = overlap / len(jd_skills)
    
    return score


def load_jd(jd_path):
    """Load Job Description from file"""
    with open(jd_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_parsed_cvs(jsonl_path):
    """Load parsed CVs from JSONL file"""
    cvs = []
    if jsonl_path.exists():
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    cvs.append(json.loads(line))
                except:
                    pass
    return cvs


def main():
    """Rank CVs for a given JD"""
    parser = argparse.ArgumentParser(description='Rank CVs for a Job Description')
    parser.add_argument('--jd', default='data/raw/jd/backend.txt',
                       help='Path to Job Description file')
    parser.add_argument('--top', type=int, default=10,
                       help='Show top N results')
    
    args = parser.parse_args()
    
    jd_path = Path(args.jd)
    if not jd_path.exists():
        print(f"[!] Không tìm thấy JD: {jd_path}")
        return
    
    # Load JD
    print(f"[*] Đọc JD: {jd_path.name}")
    jd_text = load_jd(jd_path)
    print(f"    Độ dài: {len(jd_text)} ký tự")
    
    # Load parsed CVs
    jsonl_path = Path('data/interim/cvs.jsonl')
    if not jsonl_path.exists():
        print(f"[!] Chưa parse CVs. Chạy trước: python scripts/parse_batch.py")
        return
    
    print(f"[*] Đọc CVs từ: {jsonl_path}")
    cvs = load_parsed_cvs(jsonl_path)
    print(f"    Tổng CVs: {len(cvs)}")
    
    # Rank CVs
    print(f"[*] Đang rank CVs...")
    results = []
    for cv in cvs:
        score = calculate_match_score(cv, jd_text)
        results.append({
            'filename': cv['filename'],
            'score': score,
            'skills': cv.get('entities', {}).get('skills_raw', [])
        })
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Display results
    print(f"\n[✓] Top {min(args.top, len(results))} matches:")
    print("-" * 70)
    print(f"{'Rank':<5} {'Filename':<25} {'Score':<10} {'Skills'}")
    print("-" * 70)
    
    for idx, result in enumerate(results[:args.top], 1):
        skills_str = ', '.join(result['skills'][:3])
        if len(result['skills']) > 3:
            skills_str += f" +{len(result['skills']) - 3}"
        print(f"{idx:<5} {result['filename']:<25} {result['score']:.3f}       {skills_str}")
    
    print("-" * 70)


if __name__ == "__main__":
    main()
