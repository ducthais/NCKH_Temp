from __future__ import annotations
import json
from pathlib import Path
import fitz  # PyMuPDF

def should_run_ocr(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 80:
        return True
    alpha_ratio = sum(ch.isalpha() for ch in stripped) / max(len(stripped), 1)
    return alpha_ratio < 0.2

def parse_pdf_native(pdf_path: str | Path) -> dict:
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    pages = []
    full_text_parts = []

    for i, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks", sort=True)
        text_blocks = []
        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            if block_type == 0 and text and text.strip():
                text_blocks.append({
                    "bbox": [x0, y0, x1, y1],
                    "text": text.strip()
                })

        page_text = "\n".join(block["text"] for block in text_blocks)
        pages.append({
            "page_num": i,
            "blocks": text_blocks,
            "text": page_text
        })
        full_text_parts.append(page_text)

    raw_text = "\n\n".join(full_text_parts)
    return {
        "candidate_id": pdf_path.stem,
        "source_path": str(pdf_path),
        "pages": pages,
        "raw_text": raw_text,
        "needs_ocr": should_run_ocr(raw_text),
    }

def parse_folder(input_dir="data/raw/cv", output_file="data/interim/cv_native.jsonl"):
    input_dir = Path(input_dir)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for pdf_file in sorted(input_dir.glob("*.pdf")):
            record = parse_pdf_native(pdf_file)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parse_folder()
