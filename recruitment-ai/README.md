# 🎯 AI Recruitment System — NCKH

> **Hệ thống Tuyển dụng Thông minh** ứng dụng NLP và Machine Learning để tự động phân tích CV và xếp hạng ứng viên.

## 📖 Giới thiệu

Đây là đề tài Nghiên cứu Khoa học (NCKH) xây dựng hệ thống hỗ trợ tuyển dụng sử dụng AI, bao gồm:

- **OCR & PDF Parsing** — Trích xuất văn bản từ CV dạng PDF và ảnh
- **NER (Named Entity Recognition)** — Nhận diện thực thể (kỹ năng, trường học, kinh nghiệm) bằng PhoBERT fine-tuned
- **Hybrid Matching** — Kết hợp BM25 (lexical) + Semantic Embedding để xếp hạng ứng viên
- **XAI Dashboard** — Giải thích kết quả xếp hạng trực quan (Radar chart, Skill gap, Decision matrix)

## 🗂️ Cấu trúc Project

```
recruitment-ai/
├── data/
│   ├── annotated/          # train.jsonl — dữ liệu huấn luyện NER (BIO format)
│   ├── dictionaries/       # skills.csv  — từ điển kỹ năng (200+ entries)
│   ├── raw/                # CV và JD gốc
│   ├── interim/            # File tạm khi xử lý
│   └── ranking_dataset/    # pairs.csv   — nhãn relevance cho LTR
├── src/
│   ├── parsers/            # PDF parser + OCR Tesseract
│   ├── nlp/                # Sectioner, NER Extractor, Normalizer, Embedder
│   ├── matching/           # BM25 Retriever + Hybrid Scorer
│   ├── training/           # train_ner.py (PhoBERT), train_ltr.py (XGBoost)
│   ├── evaluation/         # Metrics: F1/NDCG@10/CER
│   ├── store/              # FAISS + PostgreSQL
│   └── app/                # Streamlit Dashboard
├── scripts/
│   ├── auto_annotate_ner.py    # Weak supervision annotation
│   ├── extract_cvs_to_txt.py   # Batch PDF→TXT
│   └── rank_demo.py            # Demo xếp hạng CLI
├── experiments/
│   ├── phobert-ner-final/  # Model NER đã huấn luyện
│   └── xgboost_ltr.json    # Model LTR đã huấn luyện
├── notebooks/
│   └── eda.ipynb           # Phân tích dữ liệu khám phá
└── tests/                  # Unit & Integration tests
```

## ⚡ Cài đặt

### 1. Tạo virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 2. Cài dependencies

```bash
pip install -r requirements.txt
```

### 3. Cài Tesseract OCR (cho CV dạng ảnh)

Download: https://github.com/UB-Mannheim/tesseract/wiki

Đảm bảo thư mục `tessdata/` có file ngôn ngữ `vie.traineddata`.

### 4. (Tùy chọn) Cài Vietnamese tokenizer

```bash
pip install underthesea
```

### 5. Khởi động PostgreSQL (tùy chọn)

```bash
docker-compose up -d
```

## 🚀 Chạy ứng dụng

```bash
python -m streamlit run src/app/streamlit_app.py
```

Mở trình duyệt tại: http://localhost:8501

## 🔧 Pipeline Huấn luyện

### Bước 1: Chuẩn bị dữ liệu NER

```bash
# Chuyển CV PDF sang TXT
python scripts/extract_cvs_to_txt.py

# Hoặc dùng weak supervision để gán nhãn tự động
python scripts/auto_annotate_ner.py
```

Dữ liệu gán nhãn thủ công đặt tại: `data/annotated/train.jsonl`

### Bước 2: Huấn luyện NER model (xlm-roberta-base)

```bash
python -m src.training.train_ner
```

Model sẽ được lưu tại: `experiments/phobert-ner-final/`

### Bước 3: Huấn luyện Learning-to-Rank

Tạo file nhãn relevance tại `data/ranking_dataset/pairs.csv`:

| Cột | Mô tả |
|-----|-------|
| `jd_id` | ID của Job Description |
| `candidate_id` | ID của ứng viên |
| `bm25_score` | Điểm BM25 |
| `semantic_score` | Điểm cosine similarity |
| `skill_jaccard` | Jaccard overlap kỹ năng |
| `years_score` | Điểm kinh nghiệm (0-1) |
| `degree_match` | Bằng cấp phù hợp (0/1) |
| `relevance_label` | Nhãn liên quan (0=không, 1=thấp, 2=trung bình, 3=cao) |

```bash
python -m src.training.train_ltr
```

### Bước 4: Đánh giá

```bash
python -m src.evaluation.evaluate \
  --gold-entities data/labeled/entities/gold.csv \
  --pred-entities data/labeled/entities/pred.csv \
  --pairs data/ranking_dataset/pairs.csv
```

## 📊 Công thức Scoring

$$\text{score} = 0.30 \times \text{BM25}_{norm} + 0.40 \times \text{Semantic} + 0.20 \times \text{Skill\_Jaccard} + 0.10 \times \text{YearsExp}_{norm}$$

| Thành phần | Trọng số | Mô tả |
|-----------|---------|-------|
| BM25 | 30% | Lexical matching JD–CV |
| Semantic | 40% | Cosine similarity embedding |
| Skill Jaccard | 20% | Tỷ lệ kỹ năng khớp |
| Experience | 10% | Số năm kinh nghiệm |

## 🏷️ NER Entity Types

| Label | Ví dụ |
|-------|-------|
| `SKILL` | Python, React, Docker |
| `JOB_TITLE` | Backend Developer, Data Scientist |
| `UNIVERSITY` | Đại học Bách Khoa, SGU |
| `DEGREE` | Kỹ sư, Cử nhân, Thạc sĩ |
| `GPA` | 3.5/4.0 |
| `DURATION` | 2022-2025, T6/2024-T10/2024 |
| `COMPANY` | Google, VNG Corporation |
| `LOCATION` | Hà Nội, TP.HCM |
| `PROJECT_NAME` | SGU-Student-RAG System |
| `SOFT_SKILL` | Teamwork, Communication |
| `CERTIFICATE` | IELTS 6.5, AWS Certified |

## 🧪 Chạy Tests

```bash
pytest tests/ -v
```

## 📚 Công nghệ Sử dụng

| Lớp | Công nghệ |
|-----|----------|
| NER | xlm-roberta-base (fine-tuned), SpaCy EntityRuler |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 |
| Lexical | BM25Okapi (rank_bm25) + underthesea |
| Ranking | XGBoost LTR (rank:ndcg) |
| UI | Streamlit + Plotly |
| Storage | FAISS (vector) + PostgreSQL (metadata) |
| OCR | Tesseract + pdfplumber |

## 📄 License

MIT License — NCKH Project
