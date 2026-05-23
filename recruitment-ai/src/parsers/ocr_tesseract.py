from __future__ import annotations
from pathlib import Path
# pyrefly: ignore [missing-import]
import pdfplumber
import pytesseract
import os

# Đường dẫn mặc định của Tesseract trên Windows nếu cài qua Winget
tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tess_path):
    pytesseract.pytesseract.tesseract_cmd = tess_path
# Chỉ định thư mục chứa dữ liệu ngôn ngữ (tessdata)
tessdata_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "tessdata"))
os.environ["TESSDATA_PREFIX"] = tessdata_dir
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

def ocr_image(image_path: str | Path, lang: str = "vie+eng") -> dict:
    from PIL import Image
    image_path = Path(image_path)
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang, config=TESS_CONFIG)
    
    return {
        "candidate_id": image_path.stem,
        "source_path": str(image_path),
        "ocr_used": True,
        "pages": [{"page_num": 1, "text": text}],
        "raw_text": text,
    }
