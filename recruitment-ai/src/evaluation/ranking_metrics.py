import numpy as np

def precision_at_k(r: list[int], k: int) -> float:
    """
    Calculate Precision@K.
    r: Relevance scores (e.g., [1, 0, 1, ...])
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def dcg_at_k(r: list[int], k: int) -> float:
    """
    Calculate DCG@K.
    """
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r: list[int], k: int) -> float:
    """
    Calculate nDCG@K.
    """
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

def mean_reciprocal_rank(rs: list[list[int]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    rs: List of relevance score lists for multiple queries.
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])
