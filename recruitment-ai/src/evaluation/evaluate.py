# src/evaluation/evaluate.py
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, ndcg_score
from torchmetrics.text import CharErrorRate

def eval_entities(gold_csv: str, pred_csv: str):
    gold = pd.read_csv(gold_csv)
    pred = pd.read_csv(pred_csv)

    gold["key"] = gold["doc_id"].astype(str) + "|" + gold["label"] + "|" + gold["value_norm"]
    pred["key"] = pred["doc_id"].astype(str) + "|" + pred["label"] + "|" + pred["value_norm"]

    universe = sorted(set(gold["key"]) | set(pred["key"]))
    y_true = [1 if k in set(gold["key"]) else 0 for k in universe]
    y_pred = [1 if k in set(pred["key"]) else 0 for k in universe]

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"entity_precision": p, "entity_recall": r, "entity_f1": f1}

def eval_ranking(pairs_csv: str):
    df = pd.read_csv(pairs_csv)  # columns: jd_id, candidate_id, y_true, y_score
    ndcgs = []
    for _, g in df.groupby("jd_id"):
        ndcgs.append(ndcg_score([g["y_true"].tolist()], [g["y_score"].tolist()], k=10))
    return {"ndcg@10": float(np.mean(ndcgs)) if ndcgs else 0.0}

def eval_ocr(ocr_csv: str):
    df = pd.read_csv(ocr_csv)  # columns: gold, pred
    cer = CharErrorRate()(df["pred"].tolist(), df["gold"].tolist()).item()
    return {"ocr_cer": cer}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-entities", default="data/labeled/entities/gold.csv")
    parser.add_argument("--pred-entities", default="data/labeled/entities/pred.csv")
    parser.add_argument("--pairs", default="data/labeled/pairs/pairs.csv")
    parser.add_argument("--ocr", default="data/labeled/ocr/ocr_eval.csv")
    args = parser.parse_args()

    metrics = {}
    metrics.update(eval_entities(args.gold_entities, args.pred_entities))
    metrics.update(eval_ranking(args.pairs))
    metrics.update(eval_ocr(args.ocr))

    print(pd.Series(metrics).to_string())

if __name__ == "__main__":
    main()
