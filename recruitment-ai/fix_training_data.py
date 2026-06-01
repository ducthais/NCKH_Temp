"""
Script phat hien va sua loi annotation trong train.jsonl
- Kiem tra tung entity duoc gan nhan
- Dung heuristic de phat hien nhan sai
- Tu dong sua thanh 'O'
- Luu backup va ghi file da sua
"""

import json
import re
import os
import shutil
from collections import defaultdict, Counter
from copy import deepcopy

FILE_PATH = 'data/annotated/train.jsonl'
BACKUP_PATH = 'data/annotated/train.jsonl.backup'

# ============================================================
# HEURISTIC RULES FOR EACH ENTITY TYPE
# ============================================================

VALID_LANGUAGES = {
    'english', 'vietnamese', 'japanese', 'chinese', 'korean', 'french',
    'german', 'spanish', 'italian', 'russian', 'portuguese', 'thai',
    'hindi', 'arabic', 'malay', 'indonesian', 'dutch', 'swedish',
    'polish', 'turkish', 'czech', 'danish', 'finnish', 'greek',
    'hebrew', 'hungarian', 'norwegian', 'romanian', 'mandarin',
    'cantonese', 'tagalog', 'filipino', 'bengali', 'urdu', 'persian',
    'tieng viet', 'tieng anh', 'tieng nhat', 'tieng trung', 'tieng han',
    'tieng phap', 'tieng duc',
    # Common abbreviations
    'ielts', 'toeic', 'toefl', 'jlpt', 'topik', 'hsk', 'delf', 'dalf',
    'n1', 'n2', 'n3', 'n4', 'n5',
}

VALID_DEGREE_KEYWORDS = {
    'bachelor', 'master', 'phd', 'doctor', 'associate', 'diploma',
    'b.s.', 'b.a.', 'm.s.', 'm.a.', 'b.sc', 'm.sc', 'ph.d',
    'b.eng', 'm.eng', 'mba', 'bs', 'ba', 'ms', 'ma',
    'cu nhan', 'thac si', 'tien si', 'ky su', 'cao dang',
    'dai hoc', 'trung cap',
    'of', 'science', 'arts', 'engineering', 'technology',
    'information', 'computer', 'software', 'business',
    'administration', 'management', 'marketing', 'nursing',
}

COMMON_SKILLS = {
    'python', 'java', 'javascript', 'c++', 'c#', 'c', 'go', 'rust',
    'typescript', 'ruby', 'php', 'swift', 'kotlin', 'scala', 'r',
    'sql', 'nosql', 'html', 'css', 'react', 'angular', 'vue',
    'node.js', 'nodejs', 'express', 'django', 'flask', 'spring',
    'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'git', 'linux',
    'windows', 'macos', 'mysql', 'postgresql', 'mongodb', 'redis',
    'elasticsearch', 'kafka', 'rabbitmq', 'nginx', 'apache',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas',
    'numpy', 'matplotlib', 'power bi', 'tableau', 'excel',
    'machine learning', 'deep learning', 'nlp', 'computer vision',
    'data analysis', 'data science', 'artificial intelligence',
    'agile', 'scrum', 'jira', 'confluence', 'ci/cd', 'devops',
    'rest', 'api', 'graphql', 'microservices', 'oop',
}


def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_data(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def extract_entities(tokens, tags):
    """Extract entity spans: list of (start_idx, end_idx, entity_type, text)"""
    entities = []
    current_start = None
    current_type = None

    for i, tag in enumerate(tags):
        if tag.startswith('B-'):
            if current_type is not None:
                entities.append((current_start, i - 1, current_type,
                                 ' '.join(tokens[current_start:i])))
            current_start = i
            current_type = tag[2:]
        elif tag.startswith('I-') and current_type:
            continue  # part of current entity
        else:
            if current_type is not None:
                entities.append((current_start, i - 1, current_type,
                                 ' '.join(tokens[current_start:i])))
                current_start = None
                current_type = None

    if current_type is not None:
        entities.append((current_start, len(tags) - 1, current_type,
                         ' '.join(tokens[current_start:])))

    return entities


def is_valid_language(text):
    """Check if text is a valid language name or proficiency level"""
    text_lower = text.lower().strip()

    # Check exact match
    if text_lower in VALID_LANGUAGES:
        return True

    # Check if any valid language is contained
    for lang in VALID_LANGUAGES:
        if lang in text_lower or text_lower in lang:
            return True

    # Language proficiency patterns
    if re.search(r'(ielts|toeic|toefl|jlpt|topik|hsk|delf)\s*[\d.]+', text_lower):
        return True

    return False


def is_valid_gpa(text):
    """Check if text looks like a GPA value"""
    text = text.strip()
    # Patterns: 3.5, 3.5/4.0, 8.0/10, 3.51, etc.
    if re.search(r'^\d+\.?\d*\s*/\s*\d+\.?\d*$', text):
        return True
    if re.search(r'^\d+\.\d+$', text):
        return True
    # Single digit that could be GPA
    if re.search(r'^\d+\.?\d*$', text):
        try:
            val = float(text)
            if 0 <= val <= 10:
                return True
        except ValueError:
            pass
    return False


def is_valid_url(text):
    """Check if text looks like a URL"""
    text = text.strip().lower()
    url_patterns = [
        r'https?://', r'www\.', r'\.com', r'\.org', r'\.net', r'\.io',
        r'\.edu', r'\.vn', r'github', r'linkedin', r'gitlab', r'bitbucket',
        r'\.dev', r'\.me', r'\.co', r'facebook\.com', r'@',
    ]
    for pattern in url_patterns:
        if re.search(pattern, text):
            return True
    return False


def is_valid_certificate(text):
    """Check if text looks like a certificate/certification name"""
    text_lower = text.lower().strip()
    cert_keywords = [
        'certificate', 'certification', 'certified', 'license', 'diploma',
        'ielts', 'toeic', 'toefl', 'jlpt', 'topik', 'hsk',
        'aws', 'azure', 'google', 'cisco', 'comptia', 'oracle',
        'pmp', 'scrum', 'agile', 'itil',
        'epic', 'analyst', 'developer', 'engineer', 'professional',
        'associate', 'practitioner', 'architect', 'specialist',
        'giai', 'nhat', 'nhi', 'ba', 'gioi', 'kha', 'xuat sac',
        'award', 'prize', 'honor', 'scholarship',
        'chung chi', 'chung nhan', 'bang', 'giai thuong',
    ]
    for kw in cert_keywords:
        if kw in text_lower:
            return True

    # Very short text that's not a known cert keyword is suspicious
    if len(text_lower) <= 2 and text_lower not in {'a+', 'a', 'b', 'c'}:
        return False

    return False


def is_valid_major(text):
    """Check if text looks like an academic major/field of study"""
    text_lower = text.lower().strip()
    major_keywords = [
        'computer', 'science', 'engineering', 'technology', 'information',
        'software', 'business', 'administration', 'management', 'marketing',
        'finance', 'accounting', 'economics', 'mathematics', 'physics',
        'chemistry', 'biology', 'medicine', 'nursing', 'pharmacy',
        'law', 'education', 'psychology', 'sociology', 'history',
        'literature', 'philosophy', 'art', 'design', 'architecture',
        'electrical', 'mechanical', 'civil', 'industrial', 'data',
        'artificial', 'intelligence', 'cybersecurity', 'network',
        'multimedia', 'communication', 'journalism', 'tourism',
        'hospitality', 'agriculture', 'environmental',
        'cntt', 'cong nghe', 'thong tin', 'ky thuat', 'phan mem',
        'dien', 'dien tu', 'co khi', 'xay dung', 'kinh te',
        'quan tri', 'ke toan', 'tai chinh', 'ngan hang',
        'luat', 'giao duc', 'y', 'duoc', 'nong nghiep',
        'truyen thong', 'bao chi', 'du lich', 'khach san',
        'moi truong', 'sinh hoc', 'hoa hoc', 'toan', 'vat ly',
    ]
    for kw in major_keywords:
        if kw in text_lower:
            return True
    return False


def is_valid_company(text):
    """Check if text could be a company name"""
    text_lower = text.lower().strip()

    # Too short single punctuation or common words
    if text_lower in {',', '.', '-', '|', ':', ';', 'and', 'or', 'the', 'a', 'an', 'for', 'of', 'in', 'at', 'to'}:
        return False

    # Starts with a verb or description - likely not a company
    action_starts = [
        'dieu phoi', 'quan ly', 'ho tro', 'thiet ke', 'phat trien',
        'leader trong', 'for bao',
    ]
    for start in action_starts:
        if text_lower.startswith(start):
            return False

    return True


def is_valid_job_title(text):
    """Check if text could be a job title"""
    text_lower = text.lower().strip()

    # Punctuation-only or very short meaningless text
    if text_lower in {',', '.', '-', '|', ':', ';', 'and', 'or', 'the', 'manage', ', manage'}:
        return False

    # Starts with comma
    if text_lower.startswith(','):
        return False

    return True


def is_valid_person(text):
    """Check if text looks like a person name"""
    text = text.strip()
    # Person names should have at least 2 characters, mostly alpha
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < 2:
        return False
    # Should not be all lowercase common words
    if text.lower() in {'the', 'and', 'or', 'for', 'in', 'at', 'to', 'a', 'an'}:
        return False
    return True


def is_valid_location(text):
    """Check if text looks like a location"""
    text = text.strip()
    if len(text) <= 1 and not text.isalpha():
        return False
    if text in {',', '.', '-', '|', ':', ';'}:
        return False
    return True


def is_punctuation_only(text):
    """Check if text is only punctuation/whitespace"""
    return all(c in ',.;:|-/\\()[]{}!?@#$%^&*+=<>~` \t\n\r' for c in text)


def validate_entity(start, end, entity_type, text, tokens, tags):
    """
    Returns (is_valid, reason) for an entity.
    """
    text_stripped = text.strip()

    # Universal checks
    if is_punctuation_only(text_stripped):
        return False, f"Punctuation-only text: '{text_stripped}'"

    if len(text_stripped) == 0:
        return False, "Empty entity text"

    # Single character that's not meaningful
    if len(text_stripped) == 1 and text_stripped in {',', '.', '-', ':', ';', '|', '(', ')'}:
        return False, f"Single punctuation: '{text_stripped}'"

    # Type-specific validation
    if entity_type == 'LANGUAGE':
        if not is_valid_language(text_stripped):
            return False, f"Not a valid language: '{text_stripped}'"

    elif entity_type == 'GPA':
        if not is_valid_gpa(text_stripped):
            return False, f"Not a valid GPA: '{text_stripped}'"

    elif entity_type == 'URL':
        if not is_valid_url(text_stripped):
            return False, f"Not a valid URL: '{text_stripped}'"

    elif entity_type == 'CERTIFICATE':
        if not is_valid_certificate(text_stripped):
            return False, f"Not a valid certificate: '{text_stripped}'"

    elif entity_type == 'MAJOR':
        if not is_valid_major(text_stripped):
            return False, f"Not a valid major: '{text_stripped}'"

    elif entity_type == 'COMPANY':
        if not is_valid_company(text_stripped):
            return False, f"Not a valid company: '{text_stripped}'"

    elif entity_type == 'JOB_TITLE':
        if not is_valid_job_title(text_stripped):
            return False, f"Not a valid job title: '{text_stripped}'"

    elif entity_type == 'PERSON':
        if not is_valid_person(text_stripped):
            return False, f"Not a valid person name: '{text_stripped}'"

    elif entity_type == 'LOCATION':
        if not is_valid_location(text_stripped):
            return False, f"Not a valid location: '{text_stripped}'"

    return True, "OK"


def fix_entity_tags(tags, start, end):
    """Set tags from start to end to 'O'"""
    for i in range(start, end + 1):
        tags[i] = 'O'


def main():
    print("=" * 70)
    print("  TIM VA SUA LOI ANNOTATION TRONG train.jsonl")
    print("=" * 70)

    if not os.path.exists(FILE_PATH):
        print(f"[LOI] File khong ton tai: {FILE_PATH}")
        return

    # Backup
    shutil.copy2(FILE_PATH, BACKUP_PATH)
    print(f"\n  -> Da tao backup: {BACKUP_PATH}")

    data = load_data(FILE_PATH)
    print(f"  -> Da load {len(data)} mau")

    # Track fixes
    fix_log = []
    fix_count_by_type = Counter()
    total_fixes = 0

    for idx, item in enumerate(data):
        tokens = item['tokens']
        tags = item['ner_tags']
        item_id = item.get('id', f'line_{idx+1}')

        entities = extract_entities(tokens, tags)

        for start, end, entity_type, text in entities:
            is_valid, reason = validate_entity(start, end, entity_type, text, tokens, tags)

            if not is_valid:
                fix_log.append({
                    'sample_id': item_id,
                    'entity_type': entity_type,
                    'text': text,
                    'reason': reason,
                    'position': f"tokens[{start}:{end+1}]",
                })
                fix_entity_tags(tags, start, end)
                fix_count_by_type[entity_type] += 1
                total_fixes += 1

    # Print report
    print(f"\n{'~'*70}")
    print(f"  KET QUA: Tim thay {total_fixes} entity bi gan nhan sai")
    print(f"{'~'*70}")

    print(f"\n  Phan bo loi theo loai thuc the:")
    print(f"  {'Loai':<18} {'So loi':>10}")
    print(f"  {'~'*18} {'~'*10}")
    for etype, count in fix_count_by_type.most_common():
        print(f"  {etype:<18} {count:>10}")
    print(f"  {'~'*18} {'~'*10}")
    print(f"  {'TONG':<18} {total_fixes:>10}")

    # Print detailed log (first 50)
    print(f"\n{'~'*70}")
    print(f"  CHI TIET CAC LOI DA SUA (hien thi toi da 80 mau)")
    print(f"{'~'*70}")
    for i, fix in enumerate(fix_log[:80]):
        print(f"  [{i+1:>3}] ID={fix['sample_id']:<12} "
              f"Type={fix['entity_type']:<15} "
              f"Text='{fix['text'][:50]}' "
              f"-> {fix['reason']}")

    if len(fix_log) > 80:
        print(f"  ... va {len(fix_log) - 80} loi khac")

    # Save fixed data
    save_data(data, FILE_PATH)
    print(f"\n  -> Da luu file da sua: {FILE_PATH}")
    print(f"  -> Backup goc: {BACKUP_PATH}")
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
