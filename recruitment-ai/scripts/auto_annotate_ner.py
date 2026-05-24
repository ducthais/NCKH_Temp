"""
auto_annotate_ner.py — Gán nhãn BIO tự động bằng Weak Supervision

Nhãn được hỗ trợ:
  SKILL       — từ SpaCy EntityRuler (skills.csv)
  DEGREE      — heuristic theo keyword
  JOB_TITLE   — heuristic theo keyword
  DURATION    — regex năm và tháng/năm
  GPA         — regex điểm GPA
  UNIVERSITY  — heuristic theo keyword "đại học", "cao đẳng", "university", v.v.

Kết quả xuất ra JSONL chuẩn để đưa vào train_ner.py.
"""
import os
import re
import json
from pathlib import Path
from src.nlp.extractor import NLP, DEGREE_KEYWORDS, JOB_TITLE_HINTS

# ------------------------------------------------------------------ patterns
YEAR_RANGE_RE = re.compile(
    r"((?:19|20)\d{2})\s*[-–/]\s*((?:19|20)\d{2}|present|now|hiện\s*tại|nay)",
    re.I,
)
MONTH_YEAR_RE = re.compile(
    r"(?:t\d{1,2}|0[1-9]|1[0-2])\s*/\s*(?:20\d{2})",
    re.I,
)
GPA_RE = re.compile(r"\b(gpa|điểm\s*gpa)\s*[:：]?\s*(\d+[\.,]\d+(?:\s*/\s*\d+(?:[\.,]\d+)?)?)", re.I)

UNIVERSITY_HINTS = [
    "university", "college", "institute", "academy", "polytechnic",
    "đại học", "cao đẳng", "học viện", "trường đh", "trường đại học", "trường cao đẳng",
]

# Bổ sung vào JOB_TITLE_HINTS
EXTRA_JOB_HINTS = [
    "manager", "director", "executive", "specialist", "consultant",
    "officer", "coordinator", "leader", "head", "assistant", "associate",
    "nhân viên", "trưởng", "phó", "quản lý", "giám sát", "chuyên viên",
    "cộng tác viên", "thực tập sinh",
]

ALL_JOB_HINTS = JOB_TITLE_HINTS + EXTRA_JOB_HINTS


def _find_duration_spans(tokens: list[str]) -> dict[int, tuple[str, int]]:
    """Trả về {start_idx: (B/I-tag, end_idx)} cho các DURATION."""
    text = " ".join(tokens)
    spans = {}

    for m in YEAR_RANGE_RE.finditer(text):
        # Xác định index token
        start_char = m.start()
        end_char = m.end()
        char_count = 0
        start_tok = end_tok = None
        for i, tok in enumerate(tokens):
            if char_count == start_char and start_tok is None:
                start_tok = i
            if char_count >= end_char:
                end_tok = i
                break
            char_count += len(tok) + 1
        if start_tok is not None and end_tok is None:
            end_tok = len(tokens) - 1
        if start_tok is not None:
            for j in range(start_tok, end_tok + 1):
                spans[j] = ("B-DURATION" if j == start_tok else "I-DURATION", end_tok)

    for m in MONTH_YEAR_RE.finditer(text):
        start_char = m.start()
        end_char = m.end()
        char_count = 0
        start_tok = end_tok = None
        for i, tok in enumerate(tokens):
            if char_count == start_char and start_tok is None:
                start_tok = i
            if char_count >= end_char:
                end_tok = i
                break
            char_count += len(tok) + 1
        if start_tok is not None and end_tok is None:
            end_tok = len(tokens) - 1
        if start_tok is not None:
            for j in range(start_tok, end_tok + 1):
                if j not in spans:
                    spans[j] = ("B-DURATION" if j == start_tok else "I-DURATION", end_tok)

    return spans


def _find_gpa_spans(tokens: list[str]) -> set[int]:
    """Trả về set index của token GPA."""
    text = " ".join(tokens)
    gpa_idxs = set()
    for m in GPA_RE.finditer(text):
        start_char = m.start()
        end_char = m.end()
        char_count = 0
        in_span = False
        for i, tok in enumerate(tokens):
            if char_count >= start_char and char_count < end_char:
                gpa_idxs.add(i)
                in_span = True
            elif in_span:
                break
            char_count += len(tok) + 1
    return gpa_idxs


def _is_university_line(line_lower: str) -> bool:
    return any(hint in line_lower for hint in UNIVERSITY_HINTS)


def auto_annotate_cvs(
    input_dir: str = "data/annotated/txt_for_labeling",
    output_file: str = "data/annotated/train_auto.jsonl",
):
    """
    Dùng Weak Supervision để tự động gán nhãn BIO cho văn bản CV.
    Kết quả ghi ra JSONL chuẩn HuggingFace.
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Thư mục {input_path} không tồn tại!")
        return

    txt_files = list(input_path.glob("*.txt"))
    if not txt_files:
        print(f"Không có file .txt nào trong {input_path}")
        return

    print(f"Auto-Annotating {len(txt_files)} CV bằng Weak Supervision...")

    dataset = []

    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        doc = NLP(text)

        tokens: list[str] = []
        ner_tags: list[str] = []

        # Map entity spans từ SpaCy
        skill_ents = {ent.start: ent for ent in doc.ents if ent.label_ == "SKILL"}

        # Pre-compute DURATION & GPA spans trên token list SpaCy
        raw_tokens = [t.text for t in doc if t.text.strip()]

        i = 0
        doc_tokens_list = [t for t in doc if t.text.strip()]
        duration_spans = _find_duration_spans(raw_tokens)
        gpa_spans = _find_gpa_spans(raw_tokens)

        j = 0  # index trong doc_tokens_list
        while j < len(doc_tokens_list):
            token = doc_tokens_list[j]
            spacy_idx = token.i  # index trong doc SpaCy

            # --- SKILL từ EntityRuler ---
            if spacy_idx in skill_ents:
                ent = skill_ents[spacy_idx]
                ner_tags.append("B-SKILL")
                tokens.append(token.text)
                # các token còn lại trong entity
                for k in range(1, len(ent)):
                    next_tok_idx = j + k
                    if next_tok_idx < len(doc_tokens_list):
                        tokens.append(doc_tokens_list[next_tok_idx].text)
                        ner_tags.append("I-SKILL")
                j += len(ent)
                continue

            tok_text = token.text
            tok_lower = tok_text.lower()

            # --- DURATION ---
            if j in duration_spans:
                tag, _ = duration_spans[j]
                ner_tags.append(tag)
                tokens.append(tok_text)
                j += 1
                continue

            # --- GPA ---
            if j in gpa_spans:
                ner_tags.append("B-GPA")
                tokens.append(tok_text)
                j += 1
                continue

            # --- DEGREE ---
            if tok_lower in DEGREE_KEYWORDS:
                ner_tags.append("B-DEGREE")
                tokens.append(tok_text)
                j += 1
                continue

            # --- JOB_TITLE (single token heuristic) ---
            if tok_lower in ALL_JOB_HINTS:
                ner_tags.append("B-JOB_TITLE")
                tokens.append(tok_text)
                j += 1
                continue

            # --- UNIVERSITY (line-level heuristic) ---
            # Kiểm tra dòng hiện tại có chứa hint trường không
            sent_text = token.sent.text.lower() if token.sent else tok_lower
            if _is_university_line(sent_text) and tok_text[0].isupper():
                # Gán B cho token đầu tiên của line có chứa university hint
                if j == 0 or doc_tokens_list[j - 1].sent != token.sent:
                    ner_tags.append("B-UNIVERSITY")
                else:
                    ner_tags.append("I-UNIVERSITY")
                tokens.append(tok_text)
                j += 1
                continue

            # --- O ---
            ner_tags.append("O")
            tokens.append(tok_text)
            j += 1

        if tokens:
            dataset.append({
                "id": file_path.stem,
                "tokens": tokens,
                "ner_tags": ner_tags,
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Đã tạo {len(dataset)} mẫu → {output_path.absolute()}")
    print("   Chạy: python -m src.training.train_ner để huấn luyện.")


if __name__ == "__main__":
    auto_annotate_cvs()

