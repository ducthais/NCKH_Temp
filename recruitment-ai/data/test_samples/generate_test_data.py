#!/usr/bin/env python3
"""
Script tạo bộ mẫu thử cho hệ thống Recruitment-AI.
Tạo 5 CV PDF + 1 file JD text.

Mức độ phù hợp:
  CV1: Nguyễn Văn An    - Rất phù hợp (Senior Backend, full-stack Python/Java)
  CV2: Trần Thị Bình    - Phù hợp (Mid-level Full-Stack JS/React)
  CV3: Lê Hoàng Cường   - Trung bình (Junior, ít kinh nghiệm thực tế)
  CV4: Phạm Minh Đức    - Ít phù hợp (Marketing, sai ngành)
  CV5: Vũ Ngọc Linh     - Khá phù hợp (DevOps/Cloud + Backend)
"""

import os
from fpdf import FPDF

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# JOB DESCRIPTION
# ============================================================
JD_TEXT = """
VỊ TRÍ: BACKEND DEVELOPER (PYTHON/JAVA)

MÔ TẢ CÔNG VIỆC:
- Thiết kế và phát triển các API RESTful sử dụng Python (Django/FastAPI) hoặc Java (Spring Boot)
- Xây dựng và tối ưu hóa cơ sở dữ liệu PostgreSQL, MongoDB
- Triển khai ứng dụng trên môi trường cloud (AWS/GCP) sử dụng Docker và Kubernetes
- Tham gia code review, viết unit test và tích hợp CI/CD pipeline
- Phối hợp với team Frontend để tích hợp API
- Nghiên cứu và áp dụng các công nghệ mới phù hợp

YÊU CẦU:
- Tốt nghiệp Đại học ngành Công nghệ Thông tin hoặc tương đương
- Ít nhất 2 năm kinh nghiệm phát triển Backend
- Thành thạo Python hoặc Java, có kinh nghiệm với framework Django/FastAPI hoặc Spring Boot
- Kiến thức vững về SQL (PostgreSQL, MySQL) và NoSQL (MongoDB, Redis)
- Kinh nghiệm sử dụng Docker, Git, CI/CD
- Hiểu biết về RESTful API design, microservices architecture
- Kỹ năng giao tiếp, làm việc nhóm tốt
- Tiếng Anh đọc hiểu tài liệu kỹ thuật

ƯU TIÊN:
- Kinh nghiệm với Kubernetes, AWS
- Kiến thức về Machine Learning, NLP
- Có kinh nghiệm với Elasticsearch, message queue (Kafka/RabbitMQ)
- Đã từng tham gia dự án quy mô lớn
""".strip()


# ============================================================
# CV DATA
# ============================================================

CVS = [
    # ---- CV1: Rất phù hợp ----
    {
        "filename": "CV_Nguyen_Van_An.pdf",
        "content": [
            ("NGUYỄN VĂN AN", 18, True),
            ("Backend Developer", 14, True),
            ("Email: nguyenvan.an@gmail.com | SĐT: +84 912 345 678", 10, False),
            ("", 10, False),
            ("MỤC TIÊU NGHỀ NGHIỆP", 13, True),
            ("Lập trình viên Backend với 4 năm kinh nghiệm phát triển hệ thống web sử dụng Python và Java. Mong muốn phát triển sự nghiệp trong môi trường công nghệ hiện đại, áp dụng kiến trúc microservices và cloud computing.", 10, False),
            ("", 10, False),
            ("HỌC VẤN", 13, True),
            ("Cử nhân Công nghệ Thông tin - Đại học Bách Khoa TP.HCM", 10, False),
            ("2016 - 2020 | GPA: 3.5/4.0", 10, False),
            ("", 10, False),
            ("KINH NGHIỆM LÀM VIỆC", 13, True),
            ("Backend Developer | Công ty ABC Technology", 11, True),
            ("Tháng 3/2022 - Hiện tại", 10, False),
            ("- Phát triển RESTful API bằng Python FastAPI phục vụ hệ thống e-commerce với 500K users", 10, False),
            ("- Thiết kế và tối ưu database PostgreSQL, giảm 40% query time", 10, False),
            ("- Triển khai CI/CD pipeline sử dụng GitHub Actions và Docker", 10, False),
            ("- Xây dựng hệ thống caching với Redis, tăng throughput 3x", 10, False),
            ("- Deploy ứng dụng lên AWS ECS với Kubernetes", 10, False),
            ("", 10, False),
            ("Junior Developer | Công ty XYZ Software", 11, True),
            ("06/2020 - 02/2022", 10, False),
            ("- Phát triển backend API bằng Java Spring Boot cho hệ thống quản lý nhân sự", 10, False),
            ("- Xây dựng module authentication sử dụng JWT và Spring Security", 10, False),
            ("- Viết unit test với JUnit, đạt 85% code coverage", 10, False),
            ("- Sử dụng MongoDB cho module báo cáo và analytics", 10, False),
            ("- Tham gia code review và mentoring 2 thực tập sinh", 10, False),
            ("", 10, False),
            ("KỸ NĂNG", 13, True),
            ("Ngôn ngữ: Python, Java, JavaScript, SQL", 10, False),
            ("Framework: FastAPI, Django, Spring Boot, Spring Security", 10, False),
            ("Database: PostgreSQL, MongoDB, Redis, MySQL", 10, False),
            ("DevOps: Docker, Kubernetes, AWS (EC2, S3, Lambda), GitHub Actions, CI/CD", 10, False),
            ("Khác: Git, RESTful API, Microservices, Elasticsearch, OOP, Design Patterns", 10, False),
            ("", 10, False),
            ("DỰ ÁN", 13, True),
            ("E-commerce Platform (2023)", 11, True),
            ("- Hệ thống thương mại điện tử microservices phục vụ 500K người dùng", 10, False),
            ("- Tech stack: FastAPI, PostgreSQL, Redis, Docker, Kubernetes, AWS", 10, False),
            ("- Vai trò: Lead backend, thiết kế kiến trúc và tối ưu performance", 10, False),
            ("", 10, False),
            ("CHỨNG CHỈ", 13, True),
            ("- AWS Certified Solutions Architect – Associate (2023)", 10, False),
            ("- TOEIC 850", 10, False),
            ("", 10, False),
            ("NGOẠI NGỮ", 13, True),
            ("Tiếng Anh: Đọc hiểu tài liệu kỹ thuật tốt, giao tiếp cơ bản", 10, False),
        ]
    },

    # ---- CV2: Phù hợp ----
    {
        "filename": "CV_Tran_Thi_Binh.pdf",
        "content": [
            ("TRẦN THỊ BÌNH", 18, True),
            ("Full-Stack Developer", 14, True),
            ("Email: tranthib@outlook.com | SĐT: 0987 654 321", 10, False),
            ("", 10, False),
            ("MỤC TIÊU NGHỀ NGHIỆP", 13, True),
            ("Full-Stack Developer với 3 năm kinh nghiệm chuyên về JavaScript ecosystem. Tìm kiếm cơ hội phát triển kỹ năng backend và đóng góp vào các dự án có tác động lớn.", 10, False),
            ("", 10, False),
            ("HỌC VẤN", 13, True),
            ("Kỹ sư Phần mềm - Đại học FPT", 10, False),
            ("2017 - 2021 | GPA: 3.2/4.0", 10, False),
            ("", 10, False),
            ("KINH NGHIỆM LÀM VIỆC", 13, True),
            ("Full-Stack Developer | Startup VietTech", 11, True),
            ("01/2022 - Hiện tại", 10, False),
            ("- Phát triển ứng dụng web sử dụng React (Frontend) và Node.js Express (Backend)", 10, False),
            ("- Thiết kế REST API cho hệ thống quản lý đơn hàng", 10, False),
            ("- Sử dụng MongoDB và PostgreSQL làm database chính", 10, False),
            ("- Deploy ứng dụng trên Docker containers, CI/CD với GitLab CI", 10, False),
            ("- Tích hợp thanh toán qua VNPay và Momo", 10, False),
            ("", 10, False),
            ("Frontend Developer Intern | Công ty DEF", 11, True),
            ("06/2021 - 12/2021", 10, False),
            ("- Phát triển giao diện web responsive bằng React và TypeScript", 10, False),
            ("- Tích hợp RESTful API từ backend team", 10, False),
            ("- Sử dụng Git để quản lý phiên bản code", 10, False),
            ("", 10, False),
            ("KỸ NĂNG", 13, True),
            ("Ngôn ngữ: JavaScript, TypeScript, Python, SQL", 10, False),
            ("Frontend: React, Next.js, HTML, CSS, Tailwind", 10, False),
            ("Backend: Node.js, Express.js, FastAPI (cơ bản)", 10, False),
            ("Database: MongoDB, PostgreSQL, Redis", 10, False),
            ("Tools: Docker, Git, GitLab CI/CD, Postman, Linux", 10, False),
            ("", 10, False),
            ("DỰ ÁN", 13, True),
            ("Order Management System (2023)", 11, True),
            ("- Hệ thống quản lý đơn hàng cho chuỗi cửa hàng bán lẻ", 10, False),
            ("- Tech stack: React, Node.js, Express, MongoDB, Docker", 10, False),
            ("- Vai trò: Full-stack developer, phụ trách API và UI", 10, False),
            ("", 10, False),
            ("NGOẠI NGỮ", 13, True),
            ("Tiếng Anh: IELTS 6.5", 10, False),
        ]
    },

    # ---- CV3: Trung bình (Junior, ít kinh nghiệm) ----
    {
        "filename": "CV_Le_Hoang_Cuong.pdf",
        "content": [
            ("LÊ HOÀNG CƯỜNG", 18, True),
            ("Sinh viên CNTT", 14, True),
            ("Email: lehoangcuong99@gmail.com | SĐT: 0909 111 222", 10, False),
            ("", 10, False),
            ("MỤC TIÊU NGHỀ NGHIỆP", 13, True),
            ("Sinh viên năm cuối ngành CNTT mong muốn tìm vị trí thực tập hoặc junior developer để phát triển kỹ năng lập trình thực tế.", 10, False),
            ("", 10, False),
            ("HỌC VẤN", 13, True),
            ("Cử nhân Công nghệ Thông tin - Đại học Khoa học Tự nhiên TP.HCM", 10, False),
            ("2020 - 2024 | GPA: 3.0/4.0", 10, False),
            ("", 10, False),
            ("KINH NGHIỆM LÀM VIỆC", 13, True),
            ("Thực tập sinh Backend | Công ty GHI Solutions", 11, True),
            ("06/2023 - 09/2023", 10, False),
            ("- Hỗ trợ phát triển API bằng Python Flask cho module quản lý người dùng", 10, False),
            ("- Viết query SQL trên MySQL để tạo báo cáo", 10, False),
            ("- Học và áp dụng Git để làm việc nhóm", 10, False),
            ("", 10, False),
            ("KỸ NĂNG", 13, True),
            ("Ngôn ngữ: Python (cơ bản), Java (cơ bản), SQL", 10, False),
            ("Framework: Flask (cơ bản)", 10, False),
            ("Database: MySQL, SQLite", 10, False),
            ("Tools: Git, VS Code", 10, False),
            ("Khác: OOP, HTML, CSS", 10, False),
            ("", 10, False),
            ("DỰ ÁN HỌC TẬP", 13, True),
            ("Website Quản lý Thư viện (Đồ án môn học)", 11, True),
            ("- Xây dựng web app quản lý thư viện bằng Flask + MySQL", 10, False),
            ("- Chức năng: CRUD sách, quản lý mượn/trả, tìm kiếm", 10, False),
            ("- Làm việc nhóm 4 người, vai trò: backend developer", 10, False),
            ("", 10, False),
            ("NGOẠI NGỮ", 13, True),
            ("Tiếng Anh: TOEIC 600", 10, False),
        ]
    },

    # ---- CV4: Không phù hợp (Marketing) ----
    {
        "filename": "CV_Pham_Minh_Duc.pdf",
        "content": [
            ("PHẠM MINH ĐỨC", 18, True),
            ("Digital Marketing Specialist", 14, True),
            ("Email: phamminhduc.mkt@gmail.com | SĐT: 0933 444 555", 10, False),
            ("", 10, False),
            ("MỤC TIÊU NGHỀ NGHIỆP", 13, True),
            ("Chuyên viên Digital Marketing với 3 năm kinh nghiệm trong lĩnh vực performance marketing và content marketing. Tìm kiếm cơ hội tại doanh nghiệp công nghệ để phát triển chiến lược marketing đa kênh.", 10, False),
            ("", 10, False),
            ("HỌC VẤN", 13, True),
            ("Cử nhân Marketing - Đại học Kinh tế TP.HCM", 10, False),
            ("2017 - 2021", 10, False),
            ("", 10, False),
            ("KINH NGHIỆM LÀM VIỆC", 13, True),
            ("Digital Marketing Executive | Công ty MNO Commerce", 11, True),
            ("03/2022 - Hiện tại", 10, False),
            ("- Quản lý chiến dịch quảng cáo Facebook Ads và Google Ads với ngân sách 500 triệu/tháng", 10, False),
            ("- Tối ưu hóa SEO cho website thương mại điện tử, tăng organic traffic 150%", 10, False),
            ("- Xây dựng nội dung marketing cho các kênh social media (Facebook, TikTok, Instagram)", 10, False),
            ("- Phân tích dữ liệu chiến dịch bằng Google Analytics và báo cáo hiệu quả ROI", 10, False),
            ("", 10, False),
            ("Marketing Intern | Công ty PQR", 11, True),
            ("01/2021 - 02/2022", 10, False),
            ("- Hỗ trợ viết content cho fanpage và website", 10, False),
            ("- Quản lý lịch đăng bài trên các kênh social media", 10, False),
            ("- Hỗ trợ tổ chức sự kiện và livestream bán hàng", 10, False),
            ("", 10, False),
            ("KỸ NĂNG", 13, True),
            ("Marketing: Facebook Ads, Google Ads, TikTok Ads, SEO, Content Marketing", 10, False),
            ("Analytics: Google Analytics, Power BI", 10, False),
            ("Design: Canva, Adobe Photoshop (cơ bản)", 10, False),
            ("Khác: Microsoft Office, Copywriting, Email Marketing, Community Management", 10, False),
            ("", 10, False),
            ("DỰ ÁN", 13, True),
            ("Chiến dịch Tết 2024 - MNO Commerce", 11, True),
            ("- Lên kế hoạch và triển khai chiến dịch marketing Tết trên đa kênh", 10, False),
            ("- Kết quả: Tăng 200% doanh thu so với cùng kỳ, đạt 50K đơn hàng", 10, False),
            ("", 10, False),
            ("NGOẠI NGỮ", 13, True),
            ("Tiếng Anh: IELTS 7.0", 10, False),
        ]
    },

    # ---- CV5: Khá phù hợp (DevOps + Backend) ----
    {
        "filename": "CV_Vu_Ngoc_Linh.pdf",
        "content": [
            ("VŨ NGỌC LINH", 18, True),
            ("DevOps Engineer / Backend Developer", 14, True),
            ("Email: vungoclinh.dev@gmail.com | SĐT: +84 978 123 456", 10, False),
            ("", 10, False),
            ("MỤC TIÊU NGHỀ NGHIỆP", 13, True),
            ("DevOps Engineer với 3 năm kinh nghiệm trong việc xây dựng và vận hành hạ tầng cloud. Có nền tảng backend development với Python. Mong muốn kết hợp cả hai kỹ năng trong vai trò Backend/DevOps.", 10, False),
            ("", 10, False),
            ("HỌC VẤN", 13, True),
            ("Kỹ sư Mạng máy tính và Truyền thông dữ liệu - Đại học Bách Khoa Hà Nội", 10, False),
            ("2016 - 2021 | GPA: 3.3/4.0", 10, False),
            ("", 10, False),
            ("KINH NGHIỆM LÀM VIỆC", 13, True),
            ("DevOps Engineer | Công ty STU Cloud", 11, True),
            ("01/2023 - Hiện tại", 10, False),
            ("- Quản lý hạ tầng Kubernetes cluster (20+ nodes) trên AWS EKS", 10, False),
            ("- Xây dựng CI/CD pipeline hoàn chỉnh với Jenkins và GitHub Actions", 10, False),
            ("- Triển khai monitoring stack: Prometheus, Grafana, ELK Stack", 10, False),
            ("- Viết Terraform scripts để Infrastructure as Code", 10, False),
            ("- Tối ưu chi phí cloud, giảm 30% bill AWS hàng tháng", 10, False),
            ("", 10, False),
            ("Backend Developer | Công ty VWX Tech", 11, True),
            ("03/2021 - 12/2022", 10, False),
            ("- Phát triển REST API bằng Python Django cho hệ thống quản lý tài liệu", 10, False),
            ("- Thiết kế database PostgreSQL, viết migration scripts", 10, False),
            ("- Dockerize toàn bộ ứng dụng và setup Docker Compose cho development", 10, False),
            ("- Sử dụng Redis cho caching và Celery cho task queue", 10, False),
            ("- Deploy ứng dụng lên Google Cloud Platform (GCP)", 10, False),
            ("", 10, False),
            ("KỸ NĂNG", 13, True),
            ("Ngôn ngữ: Python, Bash, Golang (cơ bản), SQL", 10, False),
            ("Backend: Django, FastAPI, REST API", 10, False),
            ("Database: PostgreSQL, Redis, MongoDB", 10, False),
            ("DevOps: Docker, Kubernetes, AWS (EKS, EC2, S3, Lambda), GCP, Terraform", 10, False),
            ("CI/CD: Jenkins, GitHub Actions, GitLab CI", 10, False),
            ("Monitoring: Prometheus, Grafana, Elasticsearch", 10, False),
            ("Khác: Git, Linux, Nginx, Microservices, Load Balancing, Message Queue", 10, False),
            ("", 10, False),
            ("DỰ ÁN", 13, True),
            ("Cloud Migration Project (2023)", 11, True),
            ("- Di chuyển hệ thống legacy từ on-premise lên AWS", 10, False),
            ("- Thiết kế kiến trúc microservices trên Kubernetes", 10, False),
            ("- Giảm downtime xuống 99.99% uptime", 10, False),
            ("", 10, False),
            ("CHỨNG CHỈ", 13, True),
            ("- AWS Certified DevOps Engineer – Professional (2024)", 10, False),
            ("- Certified Kubernetes Administrator (CKA) (2023)", 10, False),
            ("", 10, False),
            ("NGOẠI NGỮ", 13, True),
            ("Tiếng Anh: TOEIC 780", 10, False),
        ]
    },
]


def create_pdf(data: dict):
    """Tạo PDF từ data dict."""
    from fpdf.enums import XPos, YPos
    pdf = FPDF()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_top_margin(15)
    pdf.add_page()
    
    # Add Unicode font
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path)
        pdf.add_font("DejaVu", "B", bold_font_path)
        font_name = "DejaVu"
    else:
        font_name = "Helvetica"
    
    page_w = pdf.w - pdf.l_margin - pdf.r_margin  # effective width
    
    for text, size, bold in data["content"]:
        if not text:
            pdf.ln(3)
            continue
        
        # Always reset x to left margin
        pdf.set_x(pdf.l_margin)
        
        # Section headers
        if bold and size == 13:
            pdf.set_font(font_name, "B", 11)
            pdf.set_fill_color(230, 235, 245)
            pdf.cell(page_w, 7, f"  {text}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
            pdf.ln(2)
        # Name (big title)
        elif bold and size >= 16:
            pdf.set_font(font_name, "B", 14)
            pdf.set_text_color(0, 51, 102)
            pdf.cell(page_w, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            pdf.set_text_color(0, 0, 0)
        # Subtitle
        elif bold and size == 14:
            pdf.set_font(font_name, "B", 11)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(page_w, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            pdf.set_text_color(0, 0, 0)
        # Sub-headings (company names etc)
        elif bold and size == 11:
            pdf.set_font(font_name, "B", 10)
            pdf.cell(page_w, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        # Normal text
        else:
            pdf.set_font(font_name, "", 9)
            pdf.multi_cell(page_w, 5, text)
    
    filepath = os.path.join(OUTPUT_DIR, data["filename"])
    pdf.output(filepath)
    print(f"  ✓ Created: {filepath}")


def main():
    print("=" * 60)
    print("TẠO BỘ MẪU THỬ CHO RECRUITMENT-AI")
    print("=" * 60)
    
    # Save JD
    jd_path = os.path.join(OUTPUT_DIR, "JD_Backend_Developer.txt")
    with open(jd_path, "w", encoding="utf-8") as f:
        f.write(JD_TEXT)
    print(f"\n✓ JD saved: {jd_path}")
    
    # Create CVs
    print("\nĐang tạo CV PDF...")
    for cv_data in CVS:
        create_pdf(cv_data)
    
    print(f"\n{'=' * 60}")
    print(f"HOÀN TẤT! Tạo {len(CVS)} CV + 1 JD")
    print(f"Thư mục: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    print()
    print("HƯỚNG DẪN SỬ DỤNG:")
    print("1. Mở Streamlit app → Quản lý đợt tuyển dụng")
    print("2. Tạo đợt mới, paste nội dung từ JD_Backend_Developer.txt")
    print("3. Vào Phân tích CV → Upload 5 file CV PDF")
    print("4. Bấm 'Bắt đầu Phân tích' và kiểm tra kết quả")
    print()
    print("KỲ VỌNG KẾT QUẢ:")
    print("  #1 Nguyễn Văn An   → Cao nhất (Senior Backend Python/Java)")
    print("  #2 Vũ Ngọc Linh   → Khá cao (DevOps + Backend Python)")
    print("  #3 Trần Thị Bình  → Trung bình khá (Full-Stack JS)")
    print("  #4 Lê Hoàng Cường → Thấp (Junior, ít kinh nghiệm)")
    print("  #5 Phạm Minh Đức  → Thấp nhất (Marketing, sai ngành)")


if __name__ == "__main__":
    main()
