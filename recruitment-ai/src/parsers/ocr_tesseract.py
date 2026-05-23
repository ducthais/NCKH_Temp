from __future__ import annotations
from pathlib import Path
import pdfplumber
import pytesseract

TESS_CONFIG = r"--oem 1 --psm 6"

def ocr_pdf(pdf_path: str | Path, lang: str = "vie+eng") -> dict:
    pdf_path = Path(pdf_path)
    pages = []
    raw_text_parts = []

    with pdfplumber.open(pdf_path) as doc:
        for i, page in enumerate(doc.pages, start=1):
            # Render page to PIL Image at approx 144 DPI (equivalent to zoom 2.0 from 72 DPI)
            img = page.to_image(resolution=144).original
            text = pytesseract.image_to_string(img, lang=lang, config=TESS_CONFIG)
            pages.append({"page_num": i, "text": text})
            raw_text_parts.append(text)

    return {
        "candidate_id": pdf_path.stem,
        "source_path": str(pdf_path),
        "ocr_used": True,
        "pages": pages,
        "raw_text": "\n\n".join(raw_text_parts),
    }
