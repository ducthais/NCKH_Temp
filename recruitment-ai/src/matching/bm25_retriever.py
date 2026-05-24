import string
# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi

# Thử dùng underthesea để tách từ tiếng Việt tốt hơn
try:
    # pyrefly: ignore [missing-import]
    from underthesea import word_tokenize as vi_tokenize
    _VI_TOKENIZER = True
except ImportError:
    _VI_TOKENIZER = False

class BM25Retriever:
    def __init__(self):
        self.corpus = []
        self.doc_ids = []
        self.bm25 = None

    def _tokenize(self, text: str) -> list[str]:
        """
        Tách từ thông minh:
        - Nếu có underthesea: dùng Vietnamese word segmentation
        - Fallback: whitespace + punctuation tokenizer
        Cài underthesea: pip install underthesea
        """
        text = text.lower()
        if _VI_TOKENIZER:
            try:
                # underthesea trả về chuỗi với dấu _ giữa các từ ghép
                tokens = vi_tokenize(text, format="text").split()
                return [t for t in tokens if t and t not in string.punctuation]
            except Exception:
                pass
        # Fallback: basic whitespace + punctuation split
        for p in string.punctuation:
            text = text.replace(p, " ")
        return [t for t in text.split() if t]

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
