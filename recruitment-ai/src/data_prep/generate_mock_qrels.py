import json
import random
from pathlib import Path

def generate_mock_qrels(output_path="data/ranking_dataset/qrels_mock.jsonl", num_jds=5, cvs_per_jd=20):
    """
    Generate mock qrels (query relevance scores) for Learning-to-Rank training.
    Format: {"jd_id": "jd_1", "cv_id": "cv_5", "score": 2} # 0-3 relevance score
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for jd_idx in range(1, num_jds + 1):
            jd_id = f"jd_{jd_idx:03d}"
            
            # For each JD, we mock 'cvs_per_jd' judgments
            for cv_idx in range(1, cvs_per_jd + 1):
                cv_id = f"cv_{jd_idx}_{cv_idx:03d}"
                
                # Distribution of scores: 0 (Irrelevant - 60%), 1 (Slightly relevant - 20%), 
                # 2 (Relevant - 15%), 3 (Highly relevant - 5%)
                rand_val = random.random()
                if rand_val < 0.60:
                    score = 0
                elif rand_val < 0.80:
                    score = 1
                elif rand_val < 0.95:
                    score = 2
                else:
                    score = 3
                
                record = {
                    "jd_id": jd_id,
                    "cv_id": cv_id,
                    "score": score
                }
                f.write(json.dumps(record) + "\n")
                
    print(f"Generated mock qrels at: {output_path}")

if __name__ == "__main__":
    generate_mock_qrels()
