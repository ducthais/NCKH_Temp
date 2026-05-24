"""
train_ltr.py — Huấn luyện mô hình Learning-to-Rank (XGBoost ranker)

Dữ liệu thực: data/ranking_dataset/pairs.csv
Cột yêu cầu: jd_id, candidate_id, bm25_score, semantic_score, skill_jaccard,
              years_score, degree_match (0/1), relevance_label (0-3)

Nếu chưa có file pairs.csv, script sẽ tạo file mẫu để bạn điền nhãn.
"""
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import xgboost as xgb
from pathlib import Path

PAIRS_CSV = Path("data/ranking_dataset/pairs.csv")
SAMPLE_CSV = Path("data/ranking_dataset/pairs_sample.csv")
MODEL_OUT  = Path("experiments/xgboost_ltr.json")

FEATURE_COLS = ["bm25_score", "semantic_score", "skill_jaccard", "years_score", "degree_match"]
LABEL_COL    = "relevance_label"
GROUP_COL    = "jd_id"


def create_sample_pairs_file():
    """Tạo file CSV mẫu để người dùng điền nhãn relevance."""
    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    sample = pd.DataFrame([
        {"jd_id": "jd_001", "candidate_id": "cv_001", "bm25_score": 12.5, "semantic_score": 0.72,
         "skill_jaccard": 0.60, "years_score": 0.8, "degree_match": 1, "relevance_label": 3},
        {"jd_id": "jd_001", "candidate_id": "cv_002", "bm25_score": 8.3,  "semantic_score": 0.55,
         "skill_jaccard": 0.40, "years_score": 0.4, "degree_match": 0, "relevance_label": 1},
        {"jd_id": "jd_001", "candidate_id": "cv_003", "bm25_score": 4.1,  "semantic_score": 0.38,
         "skill_jaccard": 0.20, "years_score": 0.2, "degree_match": 0, "relevance_label": 0},
    ])
    sample.to_csv(SAMPLE_CSV, index=False, encoding="utf-8")
    print(f"📄 Đã tạo file mẫu tại: {SAMPLE_CSV.absolute()}")
    print("   Hãy điền thêm dữ liệu vào file này, rồi copy thành pairs.csv để train.")


def load_real_data(csv_path: Path):
    """Load dữ liệu ranking thực từ CSV."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    missing = [c for c in FEATURE_COLS + [LABEL_COL, GROUP_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"File CSV thiếu cột: {missing}")

    # Sắp xếp theo jd_id để group liên tục
    df = df.sort_values(GROUP_COL).reset_index(drop=True)

    X = df[FEATURE_COLS].values.astype(float)
    y = df[LABEL_COL].values.astype(float)

    # Tính group sizes: số CV per JD query
    group_sizes = df.groupby(GROUP_COL, sort=False).size().tolist()

    return X, y, group_sizes, df


def create_mock_features(num_samples: int = 200):
    """
    Fallback: dữ liệu mock khi chưa có pairs.csv.
    Các features: [bm25_score, semantic_score, skill_jaccard, years_score, degree_match]
    """
    np.random.seed(42)
    X = np.random.rand(num_samples, 5)
    X[:, 0] *= 30   # BM25 scale (0-30)
    X[:, 3] *= 1.0  # years_score đã normalize 0-1
    X[:, 4] = np.random.randint(0, 2, num_samples)  # degree_match binary

    y_cont = (
        (X[:, 0] / 30) * 1.2 +
        X[:, 1] * 1.0 +
        X[:, 2] * 0.8 +
        X[:, 3] * 0.4 +
        X[:, 4] * 0.6 +
        np.random.normal(0, 0.15, num_samples)
    )
    y = np.clip(np.round(y_cont * 3 / (1.2 + 1.0 + 0.8 + 0.4 + 0.6)), 0, 3).astype(int)
    docs_per_query = 10
    group = [docs_per_query] * (num_samples // docs_per_query)
    return X, y, group


def train_ltr_model(model_out_path: str = str(MODEL_OUT)):
    out_path = Path(model_out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Chọn nguồn dữ liệu ---
    if PAIRS_CSV.exists():
        print(f"✅ Dùng dữ liệu thực: {PAIRS_CSV}")
        X, y, group, df = load_real_data(PAIRS_CSV)
        num_queries = len(set(df[GROUP_COL]))
        print(f"   {len(df)} cặp (JD, CV) | {num_queries} query JD")
    else:
        print("⚠️  Không tìm thấy data/ranking_dataset/pairs.csv — dùng mock data để demo.")
        print("   Để train với dữ liệu thực, hãy tạo pairs.csv theo hướng dẫn trong README.\n")
        create_sample_pairs_file()
        X, y, group = create_mock_features(200)

    # --- Train XGBoost LTR ---
    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_COLS)
    dtrain.set_group(group)

    params = {
        "objective":    "rank:ndcg",
        "eval_metric":  "ndcg@10",
        "learning_rate": 0.05,
        "max_depth":    4,
        "min_child_weight": 2,
        "subsample":    0.8,
        "tree_method":  "hist",
        "seed":         42,
    }

    print("🚀 Training XGBoost Ranker (rank:ndcg)...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        verbose_eval=10,
    )

    model.save_model(str(out_path))
    print(f"\n✅ Model đã được lưu tại: {out_path.absolute()}")

    # In feature importance
    scores = model.get_score(importance_type="gain")
    print("\n📊 Feature Importance (gain):")
    for feat, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"   {feat}: {score:.4f}")

    return model


if __name__ == "__main__":
    train_ltr_model()

