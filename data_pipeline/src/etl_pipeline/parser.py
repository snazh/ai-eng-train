import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pdfplumber
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_from_json, elements_to_json

from data_pipeline.src.constants import PARSE_PARAMS
from data_pipeline.src.etl_pipeline.file_service import (
    get_file_hash,
    get_processed_file,
)

EXCLUDED_CATEGORIES = {
    "Header",
    "Footer",
    "Footnote",
    "PageBreak",
    "UncategorizedText",
    "FigureCaption",
}


@dataclass
class MetaData:
    title: str
    author: str
    file_name: str
    creation_date: date


@dataclass
class Element:
    category: str
    text: str
    page: int
    metadata: MetaData


class PdfParser:
    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.is_file():
            raise FileNotFoundError(f"File not found: {self.pdf_path.resolve()}")
        self.elements: list[dict] = None
        self.metadata: MetaData = None

    def set_file_metadata(self) -> None:

        with pdfplumber.open(self.pdf_path) as pdf:
            meta = pdf.metadata or {}
            self.metadata = MetaData(
                title=meta.get("Title", ""),
                author=meta.get("Author", ""),
                file_name=self.pdf_path.stem,
                creation_date=meta.get("CreationDate", ""),
            )

        if not self.metadata.title:
            for el in self.elements:
                # fitst title -> article's main title
                if el.category == "Title":
                    self.metadata.title = el.text
                    break

    def remove_references(self) -> None:
        # remove reference list
        ref_start_index = None
        ref_end_index = len(self.elements)

        for i, el in enumerate(self.elements):
            if el.text.strip().lower() == "references":
                ref_start_index = i
                break

        if ref_start_index is not None:
            for i in range(ref_start_index + 1, len(self.elements)):
                if self.elements[i].category == "Title":
                    ref_end_index = i
                    break

            self.elements = self.elements[:ref_start_index] + self.elements[ref_end_index:]

    def remove_above_abstract(self) -> None:
        # remove everything above abstract
        abst_start_index = None

        for i, el in enumerate(self.elements):
            if el.text.strip().lower().startswith("abstract"):
                abst_start_index = i
                el.category = "Title"
                break

        if abst_start_index is not None:
            self.elements = self.elements[abst_start_index:]

    def remove_colontitles(self) -> None:
        # count title occurences
        hashmap = {}
        for el in self.elements:
            text = el.text.strip()

            if el.category == "Title":
                hashmap[text] = hashmap.get(text, 0) + 1
        # remove repetetive titles (colontitles)
        self.elements = [
            el for el in self.elements if el.category != "Title" or hashmap[el.text.strip()] <= 2
        ]

    def find_missing_titles(self) -> None:
        for el in self.elements:
            is_title = bool(re.search(r"^\d+(\.\d+)*\.\s+[A-Z]", el.text.strip()))
            if is_title:
                el.category = "Title"

    def extract_elements(self):
        pdf_file_hash = get_file_hash(self.pdf_path)

        cache_path = get_processed_file(name=self.pdf_path.stem, file_hash=pdf_file_hash)

        # retrieve cached elements
        if Path.exists(cache_path):
            elements = elements_from_json(filename=str(cache_path))
            return elements

        elements = partition_pdf(filename=str(self.pdf_path), **PARSE_PARAMS)
        # cache elements
        elements_to_json(elements, filename=str(cache_path))
        return elements

    def extract_clean_text(self) -> list[Element]:
        """
        Use ML-model
        Filters garbage (Header, Footer, Footnote).
        """

        self.elements = self.extract_elements()

        self.set_file_metadata()
        self.remove_above_abstract()
        self.remove_references()
        self.find_missing_titles()
        self.remove_colontitles()

        structured_blocks = []

        for el in self.elements:
            category = getattr(el, "category", "UncategorizedText")
            if category not in EXCLUDED_CATEGORIES:
                text = " ".join(str(el).split())
                page = el.metadata.page_number
                if text:
                    structured_blocks.append(Element(category, text, page, self.metadata))

        return structured_blocks
