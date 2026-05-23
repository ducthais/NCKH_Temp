import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_mock_features(num_samples=1000):
    """
    Mock feature generation for Learning-to-Rank.
    In a real scenario, these features would be extracted from (JD, CV) pairs:
    [bm25_score, dense_dot_product, skill_jaccard, years_diff, is_degree_match]
    """
    np.random.seed(42)
    # Features: BM25 (0-50), Dense (0-1), Jaccard (0-1), YearsExp (0-20), DegreeMatch (0/1)
    X = np.random.rand(num_samples, 5)
    X[:, 0] *= 50  # BM25 scale
    X[:, 3] *= 20  # Years scale
    X[:, 4] = np.random.randint(0, 2, num_samples) # Binary
    
    # Generate mock scores (0 to 3) somewhat correlated with features to make the model learn something
    y_continuous = (X[:, 0]/50)*1.5 + (X[:, 1])*1.0 + (X[:, 2])*0.8 + (X[:, 4])*0.5 + np.random.normal(0, 0.2, num_samples)
    y = np.clip(np.round(y_continuous), 0, 3).astype(int)
    
    # Group sizes (for ranking, we need queries and how many documents per query)
    # Let's say 20 documents per query
    group = [20] * (num_samples // 20)
    
    return X, y, group

def train_ltr_model(model_out_path="experiments/xgboost_ltr.json"):
    print("Generating mock features...")
    X, y, group = create_mock_features(1000)
    
    # Use standard train test split (in LTR, we should split by query groups, but for simplicity here)
    # We will use rank:ndcg objective
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(group)
    
    params = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg',
        'learning_rate': 0.1,
        'max_depth': 4,
        'tree_method': 'hist'
    }
    
    print("Training XGBoost Ranker...")
    model = xgb.train(params, dtrain, num_boost_round=50)
    
    out_path = Path(model_out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_path))
    print(f"Model saved to {model_out_path}")

if __name__ == "__main__":
    train_ltr_model()
