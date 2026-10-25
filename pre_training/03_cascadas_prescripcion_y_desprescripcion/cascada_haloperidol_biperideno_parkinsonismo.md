---
id: "CASC-HAL-BIP-001"
tipo: "cascada_prescripcion"
titulo: "Cascada de Prescripción Clásica: Haloperidol a Biperideno por Parkinsonismo Secundario"
farmacos_involucrados: ["Haloperidol", "Biperideno", "Trihexifenidilo"]
fuentes_primarias: ["Jufe - Psicofarmacología Práctica", "Stahl - Prescriber's Guide"]
revisado_por: "Comite Clinico"
fecha_actualizacion: "2026-10-25"
estado_revision: "aprobado"
nivel_riesgo: "Alto"
palabras_clave: ["cascada de prescripcion", "haloperidol", "biperideno", "parkinsonismo secundario", "carga anticolinergica", "discinesia tardia", "desprescripcion"]
---

# Cascada de Prescripción: Haloperidol inducido y Biperideno

## 1. Resumen Ejecutivo Verbalizable (Voice Snippet)
> "Una cascada de prescripción ocurre cuando un efecto adverso es tratado erróneamente con un segundo fármaco en lugar de ajustar el causante. El ejemplo paradigmático es el haloperidol: su potente bloqueo dopaminérgico causa rigidez y parkinsonismo secundario. Prescribir biperideno de forma crónica añade una peligrosa carga anticolinérgica con deterioro cognitivo, constipación grave y riesgo de empeorar o desenmascarar discinesia tardía. El protocolo de desprescripción razonada de Sofia orienta a reducir a la dosis mínima eficaz de haloperidol o rotar a un antipsicótico atípico con bajo perfil motor, desescalando y retirando el biperideno gradualmente en dos a cuatro semanas."

## 2. Anatomía de la Cascada Clínica
1. **Fármaco Inicial:** Haloperidol (Antipsicótico típico de alta potencia).
2. **Efecto Adverso Generado:** Parkinsonismo secundario farmacológico (bloqueo > 80% de receptores D2 en la vía dopaminérgica nigroestriatal).
3. **Interpretación Errónea / Conducta Reactiva:** Se asume que el temblor o la rigidez requieren farmacoterapia obligatoria crónica, prescribiendo **Biperideno** (Akineton) o Trihexifenidilo.
4. **Daño Iatrogénico Acumulativo:**
   - *Carga Anticolinérgica:* Deterioro neurocognitivo, fallos ejecutivos y de memoria episódica, sedación, midriasis con riesgo de glaucoma de ángulo cerrado, retención aguda de orina e íleo paralítico.
   - *Riesgo Psiquiátrico:* El biperideno genera potencial de abuso, euforia y descompensaciones psicóticas con alucinaciones tóxicas en adultos mayores.
   - *Empeoramiento Motor Tardío:* El uso crónico de anticolinérgicos no previene la discinesia tardía e incluso puede agravar sus movimientos hipercinéticos o desenmascararla al alterar el balance neuroquímico estriatal, además de generar taquifilaxia y dependencia psicológica.

## 3. Algoritmo de Desprescripción y Resolución
```mermaid
flowchart TD
    A["Paciente con Haloperidol presenta Temblor/Rigidez"] --> B["¿Identificado como efecto adverso farmacológico?"]
    B -->|Sí| C["Paso 1: ¿Se puede reducir la dosis de Haloperidol?"]
    C -->|Sí| D["Reducir a dosis mínima eficaz (≤ 2-4 mg/día) y evaluar"]
    C -->|No / Recaída| E["Paso 2: Rotar a Antipsicótico Atípico<br>(Aripiprazol, Quetiapina u Olanzapina; sopesar riesgo metabólico)"]
    D --> F["Paso 3: Retirada gradual del Biperideno<br>(Reducir 1-2 mg cada semana)"]
    E --> F
    F --> G["Resolución exitosa sin carga anticolinérgica"]
```

## 4. Guía de Interrogatorio Clínico Verbal
- "Doctor, el temblor y la lentitud que presenta este paciente comenzaron tras ajustar el haloperidol. Antes de mantener biperideno de forma crónica, ¿hemos considerado optimizar la dosis a la mínima eficaz o rotar a un antipsicótico atípico con menor afinidad motora para proteger su rendimiento cognitivo y reducir la polifarmacia?"
