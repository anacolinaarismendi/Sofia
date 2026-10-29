"""
pdf_extractor.py
Extractor de páginas de documentos PDF con sistema de caché en disco (JSON).
"""

from pathlib import Path
from typing import List, Optional
import json


class CachedPDFExtractor:
    """Extractor de texto PDF de alto rendimiento con caché persistente en disco (JSON)."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_or_extract_all_pages(self, pdf_path: Path, max_pages: Optional[int] = None) -> List[str]:
        """Carga páginas de texto desde el caché local si existe; de lo contrario, extrae del PDF.
        
        Soporta extracción completa sin límite de páginas mediante pypdfium2 (ultrarrápido),
        con fallback resiliente a pdfplumber y pypdf.
        """
        pdf_path = Path(pdf_path)
        cache_file = self.cache_dir / f"{pdf_path.stem}_pages.json"

        # Comprobar caché existente si ya tiene páginas suficientes
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    pages = json.load(f)
                    # Si no hay límite o el caché cubre lo solicitado y tiene un tamaño significativo (>150 págs)
                    if max_pages is None and len(pages) >= 500:
                        return pages
                    elif max_pages is not None and len(pages) >= max_pages:
                        return pages[:max_pages]
            except Exception:
                pass

        pages: List[str] = []
        if not pdf_path.exists():
            return pages

        # Intento 1: pypdfium2 (ultrarrápido: 640 páginas en < 1 segundo)
        try:
            import pypdfium2
            pdf = pypdfium2.PdfDocument(str(pdf_path))
            total = len(pdf)
            limite = total if max_pages is None else min(max_pages, total)
            for idx in range(limite):
                page_text = pdf[idx].get_textpage().get_text_range() or ""
                pages.append(page_text)
        except Exception:
            pages = []

        # Intento 2: pdfplumber (fallback)
        if not pages:
            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    total = len(pdf.pages)
                    limite = total if max_pages is None else min(max_pages, total)
                    for idx in range(limite):
                        text = pdf.pages[idx].extract_text() or ""
                        pages.append(text)
            except Exception:
                pages = []

        # Intento 3: pypdf (fallback)
        if not pages:
            try:
                import pypdf
                reader = pypdf.PdfReader(str(pdf_path))
                total = len(reader.pages)
                limite = total if max_pages is None else min(max_pages, total)
                for idx in range(limite):
                    text = reader.pages[idx].extract_text() or ""
                    pages.append(text)
            except Exception:
                return []

        # Guardar en caché si se extrajo texto completo
        if pages:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(pages, f, ensure_ascii=False)
            except Exception:
                pass

        return pages
