import dataclasses
from typing import Any, Dict, List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, PointStruct, VectorParams

from data_pipeline.src.etl_pipeline.chunker import Chunk

import uuid
NAMESPACE_CHUNK = uuid.NAMESPACE_DNS
class VectorRepository:
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "pdf_chunks", vector_size: int = 384):
        self.collection_name = collection_name
        
        
        self.client = QdrantClient(host=host, port=port)

        
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        # set index for payload by filename
        self.client.create_payload_index(field_name="file_name", field_schema=PayloadSchemaType.KEYWORD, collection_name=self.collection_name)
    def upsert_chunks(self, chunks: List[Chunk], source: str)-> None:

        # delete old chunks
        self.client.delete(collection_name=self.collection_name, points_selector=Filter(must=[FieldCondition(key="file_name", match=MatchValue(value=source))]))

        points = []
        for i, chunk in enumerate(chunks):

            if chunk.embedding is None:
                continue


            chunk_id = uuid.uuid5(NAMESPACE_CHUNK, f"{source}:{i}")

            start, end = chunk.pages
            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=chunk.embedding,
                    payload={
                        "chunk_title": chunk.title,
                        "chunk_text": chunk.text,
                        "chunk_start_page": start,
                        "chunk_end_page": end,
                        **dataclasses.asdict(chunk.elements[0].metadata)
                        
                    }
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)



    def search(self, query_vector: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )
        return [
            {
                "score": hit.score,
                "chunk_title": hit.payload.get("chunk_title"),
                "chunk_text": hit.payload.get("chunk_text"),
                "chunk_start_page": hit.payload.get("chunk_start_page"),
                "chunk_end_page": hit.payload.get("chunk_end_page"),
                "metadata": {k: v for k, v in hit.payload.items() if k not in ("chunk_title", "chunk_text", "chunk_start_page", "chunk_end_page")}
            }
            for hit in results.points
        ]