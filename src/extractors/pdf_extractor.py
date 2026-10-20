"""
pdf_extractor.py
Extractor de páginas de documentos PDF con sistema de caché en disco (JSON).
"""

from pathlib import Path
from typing import List, Optional
import json


class CachedPDFExtractor:
    """Extractor de texto PDF que cachea los resultados para evitar relecturas costosas."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_or_extract_all_pages(self, pdf_path: Path, max_pages: int = 150) -> List[str]:
        """Carga páginas de texto desde el caché local si existe; de lo contrario, extrae del PDF."""
        pdf_path = Path(pdf_path)
        cache_file = self.cache_dir / f"{pdf_path.stem}_pages.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    pages = json.load(f)
                    return pages[:max_pages]
            except Exception:
                pass

        pages: List[str] = []
        if not pdf_path.exists():
            return pages

        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for idx, page in enumerate(pdf.pages):
                    if idx >= max_pages:
                        break
                    text = page.extract_text() or ""
                    pages.append(text)
        except Exception:
            try:
                import pypdf
                reader = pypdf.PdfReader(str(pdf_path))
                for idx, page in enumerate(reader.pages):
                    if idx >= max_pages:
                        break
                    text = page.extract_text() or ""
                    pages.append(text)
            except Exception:
                return []

        # Guardar en caché si se extrajo texto
        if pages:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(pages, f, ensure_ascii=False)
            except Exception:
                pass

        return pages
