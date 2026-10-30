"""Motor analítico de detección, estratificación de riesgo y desprescripción de Cascadas Farmacológicas.
Implementa:
- Detección reactiva de pares de fármacos en regímenes de polifarmacia (Cascadas Activas).
- Evaluación preventiva de síntomas emergentes (Riesgo Inminente de Cascada).
- Extracción heurística de fármacos desde texto clínico o recetas médicas libres.
- Cuantificación de Carga Anticolinérgica (Anticholinergic Cognitive Burden - ACB).
- Algoritmo de desprescripción razonada según criterios STOPP/START v3 y consensos clínicos.
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple
import unicodedata

from src.cascades.rules import (
    TABLA_CARGA_ANTICOLINERGICA_ACB,
    CASCADAS_CLINICAS_EXPANDIDAS,
    PAREJAS_CASCADA_POLIFARMACIA
)

logger = logging.getLogger(__name__)


def normalizar_texto(texto: str) -> str:
    """Normaliza texto eliminando acentos, caracteres especiales y convirtiendo a minúsculas."""
    if not texto:
        return ""
    texto = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8")
    return texto.lower().strip()


def extraer_nombre_base_farmaco(cadena: str) -> str:
    """Limpia dosis, posologías y formas farmacéuticas para aislar el nombre del principio activo o marca."""
    if not cadena:
        return ""
    # Quitar dosis tipo 5mg, 10 mg/día, 0.5 mcg, etc.
    cadena = re.sub(r"\b\d+(?:[\.,]\d+)?\s*(?:mg|g|mcg|ml|ug|ui|h|hrs|horas)\b", " ", cadena, flags=re.IGNORECASE)
    # Quitar caracteres no alfanuméricos
    cadena = re.sub(r"[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]", " ", cadena)
    norm = normalizar_texto(cadena)
    stopwords_receta = {
        "cada", "al", "dia", "manana", "tarde", "noche", "comp", "comprimido", "comprimidos",
        "capsula", "capsulas", "gotas", "ampolla", "ampollas", "retard", "xr", "lp", "cr",
        "de", "con", "en", "el", "la", "los", "las", "un", "una", "por", "sobre", "c", "v", "o"
    }
    tokens = [t for t in norm.split() if len(t) > 1 and t not in stopwords_receta]
    return " ".join(tokens)


@dataclass
class CascadaActiva:
    """Representa una cascada de prescripción confirmada entre dos fármacos presentes en el régimen."""
    id: str
    farmaco_causal: str
    farmaco_cascada: str
    nivel: str
    titulo: str
    mecanismo: str
    riesgo_iatrogenico: str
    protocolo_desprescripcion: List[str]
    doc_id: str


@dataclass
class RiesgoLatente:
    """Representa un efecto adverso potencial de un fármaco que podría desencadenar una cascada."""
    farmaco: str
    sintoma: str
    nivel: str
    titulo: str
    error_tipico: str
    protocolo: List[str]
    doc_id: str


@dataclass
class CargaAnticolinergica:
    """Evaluación de la carga anticolinérgica acumulada (Escala ACB)."""
    score_total: int
    estratificacion_riesgo: str
    nivel_alerta: str
    descripcion_clinica: str
    farmacos_contribuyentes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ResultadoAnalisisRegimen:
    """Resultado consolidado del análisis integral de un régimen de polifarmacia."""
    farmacos_analizados: List[str]
    cascadas_activas: List[CascadaActiva]
    riesgos_latentes: List[RiesgoLatente]
    carga_anticolinergica: CargaAnticolinergica
    resumen_ejecutivo: str
    recomendaciones_prioritarias: List[str]


class PrescriptionCascadeDetector:
    """Detector clínico inteligente de cascadas de prescripción y carga anticolinérgica."""

    def __init__(self):
        self.tabla_acb = TABLA_CARGA_ANTICOLINERGICA_ACB
        self.cascadas_clinicas = CASCADAS_CLINICAS_EXPANDIDAS
        self.parejas_cascada = PAREJAS_CASCADA_POLIFARMACIA
        self._construir_indice_vocabulario()

    def _construir_indice_vocabulario(self):
        """Compila un vocabulario de búsqueda para extracción NER heurística en texto libre."""
        self.vocabulario_farmacos: Dict[str, str] = {}
        for entrada in self.parejas_cascada:
            for alias in entrada["causal_alias"]:
                self.vocabulario_farmacos[normalizar_texto(alias)] = entrada["farmaco_causal_nombre"]
            for alias in entrada["cascada_alias"]:
                self.vocabulario_farmacos[normalizar_texto(alias)] = entrada["farmaco_cascada_nombre"]

        for farmaco_label in self.cascadas_clinicas.keys():
            # Extraer nombre antes del paréntesis: ej. 'Haloperidol (Antipsicótico Típico)' -> 'haloperidol'
            nombre_base = farmaco_label.split("(")[0].strip()
            self.vocabulario_farmacos[normalizar_texto(nombre_base)] = farmaco_label

        for k in self.tabla_acb.keys():
            self.vocabulario_farmacos[normalizar_texto(k)] = k.capitalize()

    def normalizar_farmaco(self, nombre: str) -> str:
        """Devuelve el nombre canónico normalizado del fármaco."""
        return extraer_nombre_base_farmaco(nombre)

    def calcular_carga_anticolinergica(self, farmacos: List[str]) -> CargaAnticolinergica:
        """Calcula el score total de la Escala de Carga Cognitiva Anticolinérgica (ACB)."""
        score_total = 0
        contribuyentes = []
        farmacos_vistos: Set[str] = set()

        for f_raw in farmacos:
            f_norm = extraer_nombre_base_farmaco(f_raw)
            if not f_norm or f_norm in farmacos_vistos:
                continue
            farmacos_vistos.add(f_norm)

            # Búsqueda por subcadena o clave exacta
            item_match = None
            for k, info in self.tabla_acb.items():
                if k in f_norm or f_norm in k:
                    item_match = (k, info)
                    break

            if item_match:
                k, info = item_match
                sc = info["score"]
                score_total += sc
                if sc > 0:
                    contribuyentes.append({
                        "farmaco": f_raw,
                        "identificado_como": k.capitalize(),
                        "score": sc,
                        "grupo": info.get("grupo", "Fármaco con actividad anticolinérgica"),
                        "riesgo": info.get("riesgo", "")
                    })

        # Estratificación clínica según STOPP/START v3 y consenso ACB
        if score_total >= 3:
            estratificacion = "Alto / Crítico (ACB ≥ 3)"
            nivel_alerta = "CRÍTICO"
            desc = (
                "⚠️ ALERTA DE CARGA ANTICOLINÉRGICA ELEVADA (Score ≥ 3): Riesgo sustancialmente aumentado "
                "de deterioro cognitivo acelerado, síndrome confusional agudo (delirio), caídas con fractura, "
                "estreñimiento severo con riesgo de íleo paralítico y aumento de mortalidad global en adultos mayores."
            )
        elif score_total >= 1:
            estratificacion = "Moderado (ACB 1-2)"
            nivel_alerta = "ADVERTENCIA"
            desc = (
                "Precaución por carga anticolinérgica acumulativa: puede generar sequedad de boca, visión borrosa, "
                "ligera lentitud de procesamiento cognitivo y caídas posturales."
            )
        else:
            estratificacion = "Bajo / Seguro (ACB 0)"
            nivel_alerta = "SEGURO"
            desc = "Régimen sin carga anticolinérgica clínicamente relevante identificada."

        return CargaAnticolinergica(
            score_total=score_total,
            estratificacion_riesgo=estratificacion,
            nivel_alerta=nivel_alerta,
            descripcion_clinica=desc,
            farmacos_contribuyentes=contribuyentes
        )

    def detectar_cascadas_activas(self, farmacos: List[str]) -> List[CascadaActiva]:
        """Detecta combinaciones iatrogénicas activas (Fármaco A causa efecto que Fármaco B pretende tratar)."""
        cascadas_encontradas: List[CascadaActiva] = []
        nombres_normalizados = [extraer_nombre_base_farmaco(f) for f in farmacos if f.strip()]

        for pareja in self.parejas_cascada:
            idx_causal = None
            idx_cascada = None

            for i, f_norm in enumerate(nombres_normalizados):
                if any(alias in f_norm or f_norm in alias for alias in pareja["causal_alias"]):
                    idx_causal = i
                    break

            for j, f_norm in enumerate(nombres_normalizados):
                if j != idx_causal and any(alias in f_norm or f_norm in alias for alias in pareja["cascada_alias"]):
                    idx_cascada = j
                    break

            if idx_causal is not None and idx_cascada is not None:
                cascada = CascadaActiva(
                    id=pareja["id"],
                    farmaco_causal=farmacos[idx_causal],
                    farmaco_cascada=farmacos[idx_cascada],
                    nivel=pareja["nivel"],
                    titulo=pareja["titulo"],
                    mecanismo=pareja["mecanismo"],
                    riesgo_iatrogenico=pareja["riesgo_iatrogenico"],
                    protocolo_desprescripcion=pareja["protocolo_desprescripcion"],
                    doc_id=pareja.get("doc_id", "CASC-PROT-003")
                )
                cascadas_encontradas.append(cascada)

        return cascadas_encontradas

    def detectar_riesgo_por_sintoma(self, farmaco: str, sintoma: str) -> Optional[Dict[str, Any]]:
        """Evalúa si un síntoma emergente corresponde a un efecto secundario catalogado con riesgo de cascada."""
        f_norm = normalizar_texto(farmaco)
        s_norm = normalizar_texto(sintoma)

        # Buscar en el catálogo jerárquico
        for f_label, sintomas_dict in self.cascadas_clinicas.items():
            f_label_norm = normalizar_texto(f_label)
            if f_norm in f_label_norm or f_label_norm in f_norm:
                for s_label, regla in sintomas_dict.items():
                    s_label_norm = normalizar_texto(s_label)
                    if s_norm in s_label_norm or s_label_norm in s_norm:
                        return regla

        return None

    def extraer_farmacos_de_texto(self, texto: str) -> List[str]:
        """Extrae nombres de psicofármacos y marcas comerciales desde un texto clínico libre."""
        if not texto:
            return []

        encontrados: Set[str] = set()
        texto_norm = normalizar_texto(texto)

        for alias, canonic in self.vocabulario_farmacos.items():
            # Buscar coincidencia con límites de palabra
            patron = rf"\b{re.escape(alias)}\b"
            if re.search(patron, texto_norm):
                encontrados.add(canonic.split("(")[0].strip())

        return sorted(list(encontrados))

    def analizar_regimen(self, farmacos: List[str], texto_clinico: Optional[str] = None) -> ResultadoAnalisisRegimen:
        """Realiza el análisis integral de un régimen farmacológico."""
        todos_farmacos = list(farmacos)
        if texto_clinico:
            farmacos_extraidos = self.extraer_farmacos_de_texto(texto_clinico)
            for f in farmacos_extraidos:
                if f not in todos_farmacos:
                    todos_farmacos.append(f)

        cascadas_activas = self.detectar_cascadas_activas(todos_farmacos)
        carga_acb = self.calcular_carga_anticolinergica(todos_farmacos)

        # Buscar riesgos latentes para los fármacos prescritos
        riesgos_latentes: List[RiesgoLatente] = []
        for f in todos_farmacos:
            f_norm = normalizar_texto(f)
            for f_label, sint_dict in self.cascadas_clinicas.items():
                if f_norm in normalizar_texto(f_label):
                    for s_label, r in sint_dict.items():
                        riesgos_latentes.append(RiesgoLatente(
                            farmaco=f,
                            sintoma=s_label,
                            nivel=r.get("nivel", "ADVERTENCIA"),
                            titulo=r.get("titulo", ""),
                            error_tipico=r.get("error_tipico", ""),
                            protocolo=r.get("protocolo", []),
                            doc_id=r.get("doc_id", "CASC-PROT-003")
                        ))

        # Generar recomendaciones prioritarias
        recomendaciones: List[str] = []
        if cascadas_activas:
            for c in cascadas_activas:
                recomendaciones.append(f"🛑 <b>Desescalar {c.farmaco_cascada}:</b> Trata el efecto de {c.farmaco_causal}. Optimizar dosis o rotar el fármaco primario.")

        if carga_acb.score_total >= 3:
            recomendaciones.append(
                f"🧠 <b>Auditar carga anticolinérgica (Score ACB = {carga_acb.score_total}):</b> "
                f"Retirar o rotar fármacos con puntaje 3 para evitar delirio y deterioro de memoria en adultos mayores."
            )

        if not recomendaciones:
            recomendaciones.append("✓ No se detectaron cascadas de prescripción activas ni carga anticolinérgica crítica en la combinación analizada.")

        resumen = (
            f"Análisis de {len(todos_farmacos)} fármacos: "
            f"{len(cascadas_activas)} cascadas activas detectadas, "
            f"Carga Anticolinérgica ACB = {carga_acb.score_total} ({carga_acb.estratificacion_riesgo})."
        )

        return ResultadoAnalisisRegimen(
            farmacos_analizados=todos_farmacos,
            cascadas_activas=cascadas_activas,
            riesgos_latentes=riesgos_latentes,
            carga_anticolinergica=carga_acb,
            resumen_ejecutivo=resumen,
            recomendaciones_prioritarias=recomendaciones
        )

    def get_cascadas_clinicas_compat(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Devuelve el diccionario compatible para la interfaz de Streamlit."""
        return self.cascadas_clinicas

    def listar_farmacos_catalogo(self) -> List[str]:
        """Lista alfabéticamente todos los fármacos disponibles en el catálogo para selección en interfaz."""
        nombres = set()
        for k in self.cascadas_clinicas.keys():
            nombres.add(k.split("(")[0].strip())
        for p in self.parejas_cascada:
            for item in p["farmaco_causal_nombre"].split("/"):
                item_clean = item.strip()
                if item_clean:
                    nombres.add(item_clean)
            for item in p["farmaco_cascada_nombre"].split("/"):
                item_clean = item.strip()
                if item_clean:
                    nombres.add(item_clean)
        for k in self.tabla_acb.keys():
            nombres.add(k.capitalize())
        return sorted([n for n in nombres if len(n) >= 3])
