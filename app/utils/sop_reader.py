import os
from pathlib import Path
from typing import List

from ..rag import rag_store

BASE_DIR = Path(__file__).resolve().parent.parent / "sop_dummy"


async def list_sop_files() -> List[str]:
    """Return a list of SOP filenames (just the file name, no path)."""
    if not BASE_DIR.exists():
        return []
    return [p.name for p in BASE_DIR.iterdir() if p.is_file()]


async def read_sop(file_name: str) -> str:
    """Read a SOP file (txt or pdf) and return its plain‑text content.

    The function also stores the extracted text in the global RAGStore under the
    category ``sop`` so that later AI calls can automatically use it as context.
    """
    # Prevent path traversal – only allow files that reside in the SOP folder
    safe_name = os.path.basename(file_name)
    file_path = BASE_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"SOP file '{file_name}' not found")

    if file_path.suffix.lower() == ".txt":
        text = file_path.read_text(encoding="utf-8")
    elif file_path.suffix.lower() == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError:
            # Fitz not available – return a simple placeholder so the endpoint still works
            return "[PDF content could not be extracted – PyMuPDF not installed]"
        doc = fitz.open(str(file_path))
        # Simple extraction: concatenate text from all pages
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        # If extraction yields empty string, fall back to placeholder
        if not text.strip():
            return "PDF content could not be extracted"

    else:
        raise ValueError("Unsupported SOP file type; only .txt and .pdf are allowed")

    # Store extracted text in RAG for later use
    await rag_store.add(category="sop", data={"filename": safe_name, "text": text})
    return text
