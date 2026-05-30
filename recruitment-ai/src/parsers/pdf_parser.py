from __future__ import annotations
import json
from pathlib import Path
import fitz

def should_run_ocr(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 80:
        return True
        
    # Nếu PDF chứa nhiều text dạng "cid:xx", chứng tỏ font bị lỗi mapping
    if "cid:" in text.lower():
        return True
        
    alpha_ratio = sum(ch.isalpha() for ch in stripped) / max(len(stripped), 1)
    if alpha_ratio < 0.3:
        return True
        
    # Nhận diện font rác: Ký tự không phải ASCII và cũng không phải chữ Tiếng Việt
    non_ascii = [c for c in stripped if ord(c) > 127]
    vietnamese_chars = set("áàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ")
    corrupted_non_ascii = [c for c in non_ascii if c not in vietnamese_chars]
    
    if len(non_ascii) > 0 and len(corrupted_non_ascii) / len(non_ascii) > 0.4:
        return True
        
    # Nhận diện font rác: Quá nhiều ký tự lạ / hộp vuông
    if text.count('') > 5 or text.count('') > 5:
        return True
        
    # Nhận diện font rác: Chữ hoa bất thường giữa từ (vd: simeVelensDeerien, eA, CpencY)
    import re
    if len(re.findall(r'[a-z][A-Z]', stripped)) >= 3:
        return True

    return False

def parse_pdf_native(pdf_path: str | Path) -> dict:
    pdf_path = Path(pdf_path)
    pages = []
    full_text_parts = []

    doc = fitz.open(pdf_path)
    # Chỉ quét tối đa 3 trang đầu để tránh quét nhầm Phụ lục / Bằng cấp (Certificates)
    max_pages = min(len(doc), 3)
    for i in range(max_pages):
        page = doc[i]
        try:
            text = page.get_text("text", sort=True) or ""
        except Exception:
            text = page.get_text("text") or ""
            
        text_blocks = [{"bbox": [0, 0, 0, 0], "text": text.strip()}] if text.strip() else []

        pages.append({
            "page_num": i,
            "blocks": text_blocks,
            "text": text
        })
        full_text_parts.append(text)

    doc.close()

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
