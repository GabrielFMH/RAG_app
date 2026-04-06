import os
import tempfile
from typing import List, Tuple

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFProcessor:
    """Handles PDF parsing, text extraction, and chunking."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file using PyMuPDF."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()

        return text

    def extract_text_from_bytes(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes (for uploaded files)."""
        text = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()

        return text

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using RecursiveCharacterTextSplitter."""
        if not text.strip():
            raise ValueError("Cannot chunk empty text")

        chunks = self.text_splitter.split_text(text)
        return chunks

    def process_pdf(self, pdf_path: str) -> Tuple[str, List[str]]:
        """Full pipeline: extract text and chunk it.
        
        Returns:
            Tuple of (full_text, chunks)
        """
        full_text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(full_text)
        return full_text, chunks

    def process_pdf_bytes(self, pdf_bytes: bytes) -> Tuple[str, List[str]]:
        """Full pipeline for uploaded PDF bytes.
        
        Returns:
            Tuple of (full_text, chunks)
        """
        full_text = self.extract_text_from_bytes(pdf_bytes)
        chunks = self.chunk_text(full_text)
        return full_text, chunks

    def get_pdf_metadata(self, pdf_path: str) -> dict:
        """Extract metadata from PDF."""
        with fitz.open(pdf_path) as doc:
            metadata = doc.metadata
            page_count = doc.page_count

        return {
            "metadata": metadata,
            "page_count": page_count,
        }
