from __future__ import annotations
import io
from pathlib import Path
import fitz
import pytesseract
from PIL import Image

TESS_CONFIG = r"--oem 1 --psm 6"

def render_page_to_pil(page: fitz.Page, zoom: float = 2.0) -> Image.Image:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))

def ocr_pdf(pdf_path: str | Path, lang: str = "vie+eng") -> dict:
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    pages = []
    raw_text_parts = []

    for i, page in enumerate(doc, start=1):
        img = render_page_to_pil(page)
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
