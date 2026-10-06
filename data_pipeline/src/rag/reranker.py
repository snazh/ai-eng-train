from typing import Any

from fastembed.rerank.cross_encoder import TextCrossEncoder


class ReRanker:
    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        self.model = model
        self.encoder = TextCrossEncoder(model_name=self.model)

    def rank(self, query: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:

        texts = [doc["chunk_text"] for doc in documents]
        scores = self.encoder.rerank(query, texts)
        ranked_docs = [
            {**doc, "rerank_score": score} for doc, score in zip(documents, scores, strict=True)
        ]
        sorted_ranked_docs = sorted(ranked_docs, key=lambda doc: doc["rerank_score"], reverse=True)

        return sorted_ranked_docs
