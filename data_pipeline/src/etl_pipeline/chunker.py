


from typing import Any, Dict, List
from dataclasses import dataclass, field

from data_pipeline.src.etl_pipeline.parser import Element

@dataclass
class Chunk:
    title: str | None = None
    elements: List[Element] = field(default_factory=list)
    embedding: List[float] | None = None
    
    def add_elements(self, el: Element) -> None:
        self.elements.append(el)

    def get_chars_volume(self)->int:
        return sum([len(el.text) for el in self.elements])

    @property
    def text(self) -> str:
        return "\n".join(el.text for el in self.elements)
    @property
    def pages(self) -> tuple:
        pages = [el.page for el in self.elements if el.page is not None]
        if not pages:
            return None, None
        return min(pages), max(pages)
class Chunker:
    def __init__(self, max_chars: int = 10000):
        self.max_chars = max_chars


    def create_chunks(self, elements: List[Element]) -> List[Chunk]:

        if not elements:
            return []

        chunks: List[Chunk] = []
        current_chunk = Chunk()
        current_title: str | None = None
        for el in elements:
            is_title = (el.category == "Title")
            
            if is_title:
                current_title = el.text

            is_full = (current_chunk.get_chars_volume() + len(el.text) > self.max_chars)

            if (is_title or is_full) and current_chunk.elements:
                chunks.append(current_chunk)
                current_chunk = Chunk(title=current_title)

            if current_chunk.title is None and current_title is not None:
                current_chunk.title = current_title

            current_chunk.add_elements(el)

        # Save last chunk
        if current_chunk.elements:
            chunks.append(current_chunk)

        return chunks