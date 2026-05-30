from __future__ import annotations
from pathlib import Path
# pyrefly: ignore [missing-import]
import pdfplumber
import pytesseract
import os
from PIL import Image, ImageEnhance

# Đường dẫn mặc định của Tesseract trên Windows nếu cài qua Winget
tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tess_path):
    pytesseract.pytesseract.tesseract_cmd = tess_path
# Chỉ định thư mục chứa dữ liệu ngôn ngữ (tessdata)
tessdata_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "tessdata"))
os.environ["TESSDATA_PREFIX"] = tessdata_dir
TESS_CONFIG = '--oem 3 --psm 3 -c preserve_interword_spaces=1'

def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    # 1. Convert to grayscale
    img_gray = img.convert("L")
    
    # 2. Resize/Upscale if the image is small
    w, h = img_gray.size
    target_width = 2400
    if w < target_width:
        scale = target_width / w
        new_size = (int(w * scale), int(h * scale))
        img_gray = img_gray.resize(new_size, Image.Resampling.LANCZOS)
        
    # 3. Enhance Contrast & Sharpness
    img_gray = ImageEnhance.Contrast(img_gray).enhance(2.0)
    img_gray = ImageEnhance.Sharpness(img_gray).enhance(1.5)
    
    return img_gray

def ocr_pdf(pdf_path: str | Path, lang: str = "vie+eng") -> dict:
    pdf_path = Path(pdf_path)
    pages = []
    raw_text_parts = []

    import fitz
    from PIL import Image
    import io

    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(4.0, 4.0) # approx 300 DPI (better for OCR)
    
    max_pages = min(len(doc), 3)

    for i in range(max_pages):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        img_preprocessed = preprocess_image_for_ocr(img)
        
        text = pytesseract.image_to_string(img_preprocessed, lang=lang, config=TESS_CONFIG)
        pages.append({"page_num": i + 1, "text": text})
        raw_text_parts.append(text)

    doc.close()

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
    img_preprocessed = preprocess_image_for_ocr(img)
    text = pytesseract.image_to_string(img_preprocessed, lang=lang, config=TESS_CONFIG)
    
    return {
        "candidate_id": image_path.stem,
        "source_path": str(image_path),
        "ocr_used": True,
        "pages": [{"page_num": 1, "text": text}],
        "raw_text": text,
    }
