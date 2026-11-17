"""Explicação determinística da classificação registrada, independente da IA."""
from __future__ import annotations


def explain_prediction(context: dict) -> str:
    """Explain the recorded prediction, without inventing feature attribution."""
    prediction = context["previsao_2026"]
    risk = prediction.get("classe_risco")
    score = prediction.get("pontuacao_nao_calibrada")
    if risk not in {"alto", "médio", "baixo"}:
        return "Não há classificação de risco disponível para esta escola em 2026."
    text = f"O modelo classificou a escola com risco {risk} de apresentar IRD abaixo de 2 em 2026, usando os dados de 2025."
    if score is not None:
        value = f"{score:.3f}".replace(".", ",")
        ranges = {
            "alto": "a partir de 0,52",
            "médio": "de 0,10 até menos de 0,52",
            "baixo": "abaixo de 0,10",
        }
        text += f" A pontuação registrada foi {value}; a faixa de risco {risk} corresponde a pontuações {ranges[risk]}."
    return text


