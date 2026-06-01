"""
Script tu dong gan lai nhan LANGUAGE va bo sung MAJOR trong train.jsonl
- Tim cac token la ten ngon ngu (English, Vietnamese, ...) va gan nhan LANGUAGE
- Tim cac token la nganh hoc (Computer Science, ...) gan nhan MAJOR khi nam trong context giao duc
- Uu tien khong ghi de len cac nhan entity khac da co
"""

import json
import re
import os
from collections import Counter
from copy import deepcopy

FILE_PATH = 'data/annotated/train.jsonl'

# ============================================================
# LANGUAGE DETECTION
# ============================================================

# Map of language names (lowercase) to display names
LANGUAGE_NAMES = {
    'english': 'English',
    'vietnamese': 'Vietnamese',
    'japanese': 'Japanese',
    'chinese': 'Chinese',
    'korean': 'Korean',
    'french': 'French',
    'german': 'German',
    'spanish': 'Spanish',
    'italian': 'Italian',
    'russian': 'Russian',
    'portuguese': 'Portuguese',
    'thai': 'Thai',
    'hindi': 'Hindi',
    'arabic': 'Arabic',
    'mandarin': 'Mandarin',
    'cantonese': 'Cantonese',
}

# Vietnamese language names - multi-token
VIET_LANGUAGE_PAIRS = {
    # (token1_lower, token2_lower) -> True
    ('tiếng', 'anh'): True,
    ('tiếng', 'việt'): True,
    ('tiếng', 'nhật'): True,
    ('tiếng', 'trung'): True,
    ('tiếng', 'hàn'): True,
    ('tiếng', 'pháp'): True,
    ('tiếng', 'đức'): True,
    ('tiếng', 'tây'): True,  # Tiếng Tây Ban Nha
    ('tiếng', 'ý'): True,
    ('tiếng', 'nga'): True,
}

# Language proficiency test tokens - these should also be LANGUAGE
LANG_PROFICIENCY_SINGLE = {
    'ielts', 'toeic', 'toefl', 'jlpt', 'topik', 'hsk',
    'delf', 'dalf', 'tcf',
}

# Context words that suggest language usage
LANGUAGE_CONTEXT_BEFORE = {
    'language', 'languages', 'proficiency', 'foreign', 'ngoại', 'ngữ',
    'ngôn', 'fluent', 'native', 'intermediate', 'advanced', 'basic',
    'beginner', 'conversational', 'proficient',
}

LANGUAGE_CONTEXT_AFTER = {
    'proficiency', 'fluent', 'native', 'intermediate', 'advanced',
    'basic', 'conversational', 'level', 'speaking', 'writing',
    'reading', 'listening', 'communication',
}

# ============================================================
# MAJOR DETECTION
# ============================================================

# Single-word major identifiers
MAJOR_SINGLE_TOKENS = {
    'marketing', 'accounting', 'finance', 'economics', 'nursing',
    'pharmacy', 'medicine', 'architecture', 'journalism',
    'tourism', 'hospitality', 'law', 'psychology',
}

# Multi-token major patterns (tuples of lowercase tokens)
MAJOR_MULTI_PATTERNS = [
    # English
    ('computer', 'science'),
    ('information', 'technology'),
    ('software', 'engineering'),
    ('electrical', 'engineering'),
    ('mechanical', 'engineering'),
    ('civil', 'engineering'),
    ('industrial', 'engineering'),
    ('chemical', 'engineering'),
    ('data', 'science'),
    ('data', 'analytics'),
    ('artificial', 'intelligence'),
    ('business', 'administration'),
    ('business', 'management'),
    ('international', 'business'),
    ('public', 'relations'),
    ('graphic', 'design'),
    ('interior', 'design'),
    ('digital', 'marketing'),
    ('content', 'marketing'),
    ('environmental', 'science'),
    ('political', 'science'),
    ('computer', 'engineering'),
    ('network', 'engineering'),
    ('cybersecurity', 'engineering'),
    ('multimedia', 'communication'),
    ('international', 'relations'),
    
    # Vietnamese
    ('công', 'nghệ', 'thông', 'tin'),
    ('kỹ', 'thuật', 'phần', 'mềm'),
    ('khoa', 'học', 'máy', 'tính'),
    ('quản', 'trị', 'kinh', 'doanh'),
    ('kế', 'toán'),
    ('tài', 'chính'),
    ('ngân', 'hàng'),
    ('luật', 'thương', 'mại'),
    ('luật', 'kinh', 'tế'),
    ('sư', 'phạm'),
    ('truyền', 'thông'),
    ('báo', 'chí'),
    ('du', 'lịch'),
    ('kiến', 'trúc'),
    ('y', 'khoa'),
    ('dược', 'học'),
    ('sinh', 'học'),
    ('hóa', 'học'),
    ('vật', 'lý'),
    ('toán', 'học'),
    ('ngôn', 'ngữ'),
    ('ngoại', 'ngữ'),
    ('kỹ', 'thuật'),
    ('điện', 'tử'),
    ('cơ', 'khí'),
    ('xây', 'dựng'),
    ('môi', 'trường'),
]

# Context keywords that suggest a major field is nearby
MAJOR_CONTEXT_BEFORE = {
    'major', 'major:', 'ngành', 'ngành:', 'chuyên', 'chuyên:', 
    'specialization', 'specialization:', 'field', 'concentration',
    'degree', 'b.s.', 'b.a.', 'm.s.', 'm.a.', 'bachelor', 'master',
}

EDUCATION_SECTION_KEYWORDS = {
    'education', 'học', 'vấn', 'coursework', 'academic',
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


def is_already_tagged(tags, idx):
    """Check if a token at idx is already part of an entity"""
    return tags[idx] != 'O'


def tag_single_token(tags, idx, entity_type):
    """Tag a single token as B-entity_type"""
    if not is_already_tagged(tags, idx):
        tags[idx] = f'B-{entity_type}'
        return True
    return False


def tag_multi_tokens(tags, start_idx, length, entity_type):
    """Tag multiple consecutive tokens as B-/I- entity_type"""
    # Check none are already tagged
    for i in range(start_idx, start_idx + length):
        if is_already_tagged(tags, i):
            return False
    
    tags[start_idx] = f'B-{entity_type}'
    for i in range(start_idx + 1, start_idx + length):
        tags[i] = f'I-{entity_type}'
    return True


def retag_soft_skill_to_language(tokens, tags, idx):
    """If a token is tagged as SOFT_SKILL but is actually a language name, retag it"""
    token_lower = tokens[idx].lower()
    if token_lower in LANGUAGE_NAMES:
        if tags[idx] == f'B-SOFT_SKILL':
            # Check if this is a standalone entity or start of multi-token
            # Only retag if the entity is just the language name
            end = idx + 1
            while end < len(tags) and tags[end] == 'I-SOFT_SKILL':
                end += 1
            
            entity_text = ' '.join(tokens[idx:end]).lower()
            # If the entity is just the language name, retag
            if entity_text == token_lower:
                tags[idx] = 'B-LANGUAGE'
                return True
    return False


def add_language_labels(tokens, tags):
    """Find and tag LANGUAGE entities in the token list"""
    changes = 0
    n = len(tokens)
    
    for i in range(n):
        token_lower = tokens[i].lower().strip()
        
        # 1. Check for Vietnamese language pairs: "Tiếng Anh", "Tiếng Việt", etc.
        if i + 1 < n:
            pair = (tokens[i].lower(), tokens[i+1].lower())
            if pair in VIET_LANGUAGE_PAIRS:
                if tag_multi_tokens(tags, i, 2, 'LANGUAGE'):
                    changes += 1
                    continue
        
        # 2. Check for single English language names
        if token_lower in LANGUAGE_NAMES:
            # First try to retag from SOFT_SKILL
            if tags[i].startswith('B-SOFT_SKILL'):
                if retag_soft_skill_to_language(tokens, tags, i):
                    changes += 1
                    continue
            # Otherwise tag if currently O
            if tag_single_token(tags, i, 'LANGUAGE'):
                changes += 1
                continue
        
        # 3. Check for language proficiency tests
        if token_lower in LANG_PROFICIENCY_SINGLE:
            # Look for score after it (e.g., "IELTS 6.0", "TOEIC 750")
            if not is_already_tagged(tags, i):
                tags[i] = 'B-LANGUAGE'
                # Check if next token is a score
                if i + 1 < n and re.match(r'^\d+\.?\d*$', tokens[i+1]):
                    if not is_already_tagged(tags, i + 1):
                        tags[i+1] = 'I-LANGUAGE'
                changes += 1
                continue
    
    return changes


def add_major_labels(tokens, tags):
    """Find and tag MAJOR entities in context of education sections"""
    changes = 0
    n = len(tokens)
    
    # Strategy: look for major context keywords and tag the following field name
    for i in range(n):
        token_lower = tokens[i].lower().strip().rstrip(':')
        
        # Check if this token is a major context keyword
        if token_lower in {'major', 'ngành', 'chuyên ngành'}:
            # Skip colon if present
            j = i + 1
            if j < n and tokens[j] in {':', '|', '-'}:
                j += 1
            
            if j >= n:
                continue
                
            # Try multi-token major patterns first
            matched = False
            for pattern in MAJOR_MULTI_PATTERNS:
                plen = len(pattern)
                if j + plen <= n:
                    candidate = tuple(tokens[k].lower() for k in range(j, j + plen))
                    if candidate == pattern:
                        if tag_multi_tokens(tags, j, plen, 'MAJOR'):
                            changes += 1
                            matched = True
                            break
            
            if matched:
                continue
            
            # Try to tag remaining tokens until a stop word or punctuation
            if not matched and j < n and not is_already_tagged(tags, j):
                # Collect tokens that look like a major name
                major_tokens = []
                k = j
                while k < n:
                    t = tokens[k]
                    # Stop at common delimiters
                    if t in {';', '|', '\n', '.', 'GPA', 'gpa', 'Expected', 'expected'}:
                        break
                    if t.lower() in {'gpa', 'expected', 'graduation', 'coursework', 'courses'}:
                        break
                    # Stop if already tagged
                    if is_already_tagged(tags, k):
                        break
                    major_tokens.append(k)
                    k += 1
                
                # Only tag if we found reasonable major tokens (1-6 words)
                if 1 <= len(major_tokens) <= 6:
                    # Trim trailing punctuation
                    while major_tokens and tokens[major_tokens[-1]] in {',', ';', '.', ':', '|', '-'}:
                        major_tokens.pop()
                    
                    if major_tokens:
                        tags[major_tokens[0]] = 'B-MAJOR'
                        for mi in major_tokens[1:]:
                            tags[mi] = 'I-MAJOR'
                        changes += 1
    
    # Also look for degree patterns: "B.S. Software Engineering"
    for i in range(n):
        token_lower = tokens[i].lower()
        if token_lower in {'b.s.', 'b.a.', 'm.s.', 'm.a.', 'b.sc', 'm.sc', 'b.eng', 'm.eng'}:
            # The following tokens might be the major
            j = i + 1
            if j < n and not is_already_tagged(tags, j):
                major_tokens = []
                k = j
                while k < n and len(major_tokens) < 5:
                    t = tokens[k]
                    if t in {'.', ';', '|', ',', 'GPA', 'gpa', 'Expected'}:
                        break
                    if t.lower() in {'gpa', 'expected', 'graduation'}:
                        break
                    if is_already_tagged(tags, k):
                        break
                    major_tokens.append(k)
                    k += 1
                
                # Trim trailing punctuation
                while major_tokens and tokens[major_tokens[-1]] in {',', ';', '.', ':', '|', '-'}:
                    major_tokens.pop()
                
                if 1 <= len(major_tokens) <= 5:
                    # Check if these are already tagged as DEGREE
                    already_degree = any(tags[mi].endswith('DEGREE') for mi in major_tokens if is_already_tagged(tags, mi))
                    if not already_degree:
                        all_clear = all(not is_already_tagged(tags, mi) for mi in major_tokens)
                        if all_clear:
                            tags[major_tokens[0]] = 'B-MAJOR'
                            for mi in major_tokens[1:]:
                                tags[mi] = 'I-MAJOR'
                            changes += 1
    
    return changes


def main():
    print("=" * 70)
    print("  GAN LAI NHAN LANGUAGE VA BO SUNG MAJOR")
    print("=" * 70)

    data = load_data(FILE_PATH)
    print(f"  Da load {len(data)} mau")

    total_lang_changes = 0
    total_major_changes = 0

    for item in data:
        tokens = item['tokens']
        tags = item['ner_tags']

        lang_changes = add_language_labels(tokens, tags)
        major_changes = add_major_labels(tokens, tags)

        total_lang_changes += lang_changes
        total_major_changes += major_changes

    print(f"\n  Ket qua:")
    print(f"    LANGUAGE entities da them : {total_lang_changes}")
    print(f"    MAJOR entities da them    : {total_major_changes}")
    print(f"    Tong                      : {total_lang_changes + total_major_changes}")

    # Count final distribution
    lang_count = 0
    major_count = 0
    all_types = Counter()
    for item in data:
        for tag in item['ner_tags']:
            if tag.startswith('B-'):
                etype = tag[2:]
                all_types[etype] += 1
                if etype == 'LANGUAGE':
                    lang_count += 1
                elif etype == 'MAJOR':
                    major_count += 1

    print(f"\n  Phan bo entity sau khi sua:")
    for etype, count in all_types.most_common():
        print(f"    {etype:<18}: {count}")

    print(f"\n  Tong so loai entity: {len(all_types)}")

    # Save
    save_data(data, FILE_PATH)
    print(f"\n  Da luu file: {FILE_PATH}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
