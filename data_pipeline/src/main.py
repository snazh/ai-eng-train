from data_pipeline.src.etl_pipeline.chunker import Chunker
from data_pipeline.src.etl_pipeline.embedder import Embedder
from data_pipeline.src.etl_pipeline.file_service import get_raw_files
from data_pipeline.src.etl_pipeline.parser import PdfParser
from data_pipeline.src.etl_pipeline.vector_db import VectorRepository
from data_pipeline.src.llm.llama_cpp import LLamaCppLLM
from data_pipeline.src.rag.reranker import ReRanker
from data_pipeline.src.rag.service import RagService
from data_pipeline.src.config import config

def main():
    print("Starting...")
    # --- PHASE 1: INGESTION (PDF -> Qdrant) ---
    pdf_path = get_raw_files()[0]



    llm = LLamaCppLLM(base_url=config.LLAMA_BASE_URL)
    if not llm.available():
        raise Exception("llama cpp health is false")
    parser = PdfParser(pdf_path)
    elements = parser.extract_clean_text()

    chunker = Chunker(max_chars=3000)
    chunks = chunker.create_chunks(elements)

    embedder = Embedder()
    chunks = embedder.embed_chunks(chunks)



    repo = VectorRepository()
    repo.upsert_chunks(chunks, source=pdf_path.stem)
    print(f"Successfully indexed {len(chunks)} chunks into Qdrant.")

    # --- PHASE 2: RETRIEVAL & PROMPT GENERATION ---

    reranker = ReRanker()
    
    rag = RagService(embedder=embedder, repository=repo, llm=llm, reranker=reranker)
    
    query = "What is the primary mechanism of Scaled Dot-Product Attention?"
    retrieved_docs = rag.retrieve_context(query=query, candidates=20, top_k=2)
    prompt = rag.build_prompt(query=query, context_docs=retrieved_docs)
    answer = rag.ask_llm(prompt=prompt)
    print("\n=== Generated LLM Prompt ===")
    print(prompt)
    print("\n=== Generated LLM Answer ===")
    print(answer)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:

        print("Pipeline failed")
        print(e)