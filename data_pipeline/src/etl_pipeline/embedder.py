from fastembed import TextEmbedding

from data_pipeline.src.etl_pipeline.chunker import Chunk


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        # Fast amd light ONNX-модель
        self.model = TextEmbedding(model_name=model_name)

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        texts = [ch.text for ch in chunks]
        embeddings = list(self.model.embed(texts))

        for chunk, emb in zip(chunks, embeddings, strict=True):
            chunk.embedding = emb.tolist()

        return chunks

    def embed_query(self, query: str) -> list[float]:
        if not query or not query.strip():
            return []

        embeddings = list(self.model.query_embed([query]))
        return embeddings[0].tolist()
