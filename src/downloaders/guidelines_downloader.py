"""
guidelines_downloader.py
Gestor de verificación y descarga de guías clínicas oficiales y literatura médica.
"""

from pathlib import Path
from typing import Dict, List, Optional


class GuidelinesDownloader:
    """Verifica y gestiona los documentos clínicos originales en data/raw."""

    GUIAS_REQUERIDAS: Dict[str, Dict[str, str]] = {
        "CANMAT_BIPOLAR_2018": {
            "nombre_archivo": "BDI-20-97.pdf",
            "descripcion": "CANMAT and ISBD 2018 guidelines for the management of patients with bipolar disorder",
            "referencia": "Bipolar Disord. 2018;20:97–170."
        },
        "JUFE_PSICOFARMACOLOGIA": {
            "nombre_archivo": "Psicofarmacologia practica - G. Jufe.pdf",
            "descripcion": "Psicofarmacología Práctica - Dra. Gabriela Jufe (3ª Edición)",
            "referencia": "Editorial Polemos, Buenos Aires."
        }
    }

    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or Path("data/raw")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def check_status(self) -> Dict[str, bool]:
        """Comprueba el estado de disponibilidad de cada guía requerida."""
        status = {}
        for key, info in self.GUIAS_REQUERIDAS.items():
            path_esperado = self.raw_dir / info["nombre_archivo"]
            status[key] = path_esperado.exists()
        return status

    def get_missing_guidelines(self) -> List[str]:
        """Devuelve la lista de guías clínicas no presentes en data/raw."""
        return [k for k, exists in self.check_status().items() if not exists]
