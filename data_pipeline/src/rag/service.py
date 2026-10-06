from typing import List, Dict, Any  
from data_pipeline.src.etl_pipeline.embedder import Embedder
from data_pipeline.src.etl_pipeline.vector_db import VectorRepository
from data_pipeline.src.llm.base import BaseLLM
from data_pipeline.src.rag.reranker import ReRanker




class RagService:
    def __init__(self, embedder: Embedder, repository: VectorRepository, llm: BaseLLM, reranker: ReRanker):
        self.embedder = embedder
        self.repository = repository
        self.llm = llm
        self.reranker = reranker

    def retrieve_context(self, query: str, candidates: int = 20, top_k: int = 2)-> List[Dict[str, Any]]:
        query_vector = self.embedder.embed_query(query)
        if not query_vector:
            return []


        # search in vector DB
        documents = self.repository.search(query_vector=query_vector, limit=candidates)

        reranked_documents = self.reranker.rank(query, documents)
        return reranked_documents[:top_k]

    

    def build_prompt(self, query: str, context_docs: List[Dict[str, Any]])->str:
        context_blocks = []
        for i, doc in enumerate(context_docs, 1):
            title = doc.get("chunk_title") or "Untitled Section"
            context_blocks.append(f"--- Document Chunk {i} [{title}, p. {doc['chunk_start_page']}-{doc['chunk_end_page']}] ---\n{doc['chunk_text']}")

        context_str = "\n\n".join(context_blocks)

        prompt = f"""Answer the question using ONLY the context snippets below.

Rules:
- Cite the page after each statement, like this: [p. 4]
- If the context does not contain the answer, reply exactly: "Insufficient information in the provided context."
- Do not use any knowledge outside the context.

Context:
{context_str}

Question: {query}

Answer:"""

        
        return prompt

    def ask_llm(self, prompt:str) -> str:
        return self.llm.answer(prompt)