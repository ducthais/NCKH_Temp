import os
from pathlib import Path
from src.parsers.pdf_parser import parse_pdf_native
from src.parsers.ocr_tesseract import ocr_pdf, ocr_image

def extract_all_cvs(input_dir="data/raw/cv", output_dir="data/annotated/txt_for_labeling"):
    """
    Quét toàn bộ file PDF và file ảnh trong thư mục input_dir, bóc tách text 
    và lưu thành file .txt vào thư mục output_dir để dùng cho Label Studio.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Tạo thư mục output nếu chưa có
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Thư mục {input_path} không tồn tại. Vui lòng tạo và thả file PDF/Ảnh vào.")
        return

    # Quét tất cả file định dạng pdf, png, jpg, jpeg
    files_to_process = []
    for ext in ("*.pdf", "*.png", "*.jpg", "*.jpeg"):
        files_to_process.extend(input_path.glob(ext))
        
    if not files_to_process:
        print(f"Không tìm thấy file PDF hay Ảnh nào trong {input_path}.")
        return

    print(f"Tìm thấy {len(files_to_process)} file. Đang tiến hành bóc tách...")
    
    success_count = 0
    for file_path in files_to_process:
        try:
            ext = file_path.suffix.lower()
            if ext == ".pdf":
                # 1. Parse bằng pdfplumber (text native)
                parsed = parse_pdf_native(file_path)
                
                # 2. Nếu file là dạng ảnh quét (cần OCR)
                if parsed.get("needs_ocr", False):
                    print(f"[{file_path.name}] PDF cần OCR. Đang xử lý bằng Tesseract...")
                    parsed = ocr_pdf(file_path)
            else:
                # Ảnh (png, jpg, jpeg) -> Chạy OCR trực tiếp
                print(f"[{file_path.name}] Là file Ảnh. Đang OCR bằng Tesseract...")
                parsed = ocr_image(file_path)
                
            raw_text = parsed.get("raw_text", "").strip()
            
            # 3. Lưu ra file txt
            if raw_text:
                txt_filename = file_path.stem + ".txt"
                txt_filepath = output_path / txt_filename
                
                with open(txt_filepath, "w", encoding="utf-8") as f:
                    f.write(raw_text)
                
                success_count += 1
                print(f"✅ Đã trích xuất: {txt_filename}")
            else:
                print(f"⚠️ Cảnh báo: File {file_path.name} không có chữ nào được trích xuất.")
                
        except Exception as e:
            print(f"❌ Lỗi khi xử lý file {file_path.name}: {e}")

    print("-" * 40)
    print(f"Hoàn tất! Đã trích xuất thành công {success_count}/{len(files_to_process)} file.")
    print(f"Các file .txt được lưu tại: {output_path.absolute()}")

if __name__ == "__main__":
    extract_all_cvs()
