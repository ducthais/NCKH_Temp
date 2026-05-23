import string
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self):
        self.corpus = []
        self.doc_ids = []
        self.bm25 = None

    def _tokenize(self, text: str) -> list[str]:
        """Basic whitespace and punctuation tokenizer. Can be replaced with Underthesea/Pyvi for Vietnamese."""
        text = text.lower()
        for p in string.punctuation:
            text = text.replace(p, " ")
        return text.split()

    def fit(self, documents: list[dict]):
        """
        Fit BM25 model.
        documents: list of dict with 'id' and 'text'
        """
        self.corpus = []
        self.doc_ids = []
        tokenized_corpus = []

        for doc in documents:
            self.doc_ids.append(doc["id"])
            self.corpus.append(doc["text"])
            tokenized_corpus.append(self._tokenize(doc["text"]))

        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Retrieve top_k documents for a query.
        """
        if not self.bm25:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "doc_id": self.doc_ids[idx],
                "score": float(scores[idx]),
                "text": self.corpus[idx]
            })
        return results
