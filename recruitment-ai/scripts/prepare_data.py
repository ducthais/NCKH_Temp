"""Data preparation script - Sample data generator"""
from pathlib import Path


def create_sample_data():
    """Create sample CV and JD files for demo"""
    # Create sample CVs
    cv_dir = Path('data/raw/cv')
    cv_dir.mkdir(parents=True, exist_ok=True)
    
    sample_cv1 = """Lê Văn A
Email: leva@example.com
Phone: +84912345678

KỸ NĂNG
Python, SQL, Docker, Machine Learning

KINH NGHIỆM
Senior Backend Developer tại TechCorp 2021-2024
- Phát triển API với FastAPI
- Xử lý dữ liệu lớn với Python

HỌC VẤN
Cử nhân Khoa học Máy tính, ĐH Bách Khoa 2020
"""
    
    sample_cv2 = """Nguyễn Thị B
Email: thib@example.com
Phone: +84987654321

KỸ NĂNG
JavaScript, React, SQL, Docker

KINH NGHIỆM
Frontend Developer tại StartupXYZ 2022-2024
- Phát triển giao diện với React
- Tối ưu hóa hiệu suất website

HỌC VẤN
Cử nhân Công nghệ Thông tin, ĐH FPT 2021
"""
    
    with open(cv_dir / 'cv_a.txt', 'w', encoding='utf-8') as f:
        f.write(sample_cv1)
    
    with open(cv_dir / 'cv_b.txt', 'w', encoding='utf-8') as f:
        f.write(sample_cv2)
    
    # Create sample JDs
    jd_dir = Path('data/raw/jd')
    jd_dir.mkdir(parents=True, exist_ok=True)
    
    sample_jd = """Backend Developer - Python/FastAPI
    
Tôi cần tìm kiếm:
- 3+ năm kinh nghiệm với Python
- Thành thạo FastAPI và SQL
- Có kinh nghiệm Docker
- Hiểu biết về Machine Learning là lợi thế

Mô tả công việc:
- Xây dựng APIs cho hệ thống phức tạp
- Optimize database queries
- Làm việc trong team
"""
    
    with open(jd_dir / 'backend.txt', 'w', encoding='utf-8') as f:
        f.write(sample_jd)
    
    print("✓ Tạo sample data thành công")
    print(f"  CVs: {list(cv_dir.glob('*.txt'))}")
    print(f"  JDs: {list(jd_dir.glob('*.txt'))}")


def main():
    """Prepare and preprocess data"""
    print("[*] Chuẩn bị dữ liệu mẫu...")
    create_sample_data()
    print("[✓] Hoàn tất")


if __name__ == "__main__":
    main()
