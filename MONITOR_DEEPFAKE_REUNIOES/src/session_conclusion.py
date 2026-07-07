from __future__ import annotations

from src.risk.engine import RiskAssessment


def friendly_cycle_result(risk: RiskAssessment) -> tuple[str, str]:
    if risk.level == "ALTO":
        return (
            "Atencao: foram encontrados sinais relevantes.",
            "Recomenda-se revisar a identidade e o comportamento do video.",
        )
    if risk.level == "MODERADO":
        return (
            "A analise encontrou pontos que merecem revisao.",
            "Continue a sessao e observe se os sinais se repetem.",
        )
    return (
        "Nenhum sinal relevante neste periodo.",
        "A verificacao continua ate o recrutador encerrar.",
    )


def build_final_conclusion(
    assessments: list[dict],
    profile_name: str,
    total_periods: int | None = None,
) -> dict:
    total_periods = len(assessments) if total_periods is None else total_periods
    if not assessments:
        return {
            "profile": profile_name,
            "status": "INCONCLUSIVO",
            "title": "Analise inconclusiva",
            "message": (
                "Os periodos observados nao reuniram imagens faciais suficientes "
                "para uma avaliacao completa."
            ),
            "cycles": total_periods,
            "completed_cycles": 0,
            "inconclusive_cycles": total_periods,
        }

    risks = [item["risk"] for item in assessments]
    identities = [item["identity"] for item in assessments]
    deepfakes = [item["deepfake"] for item in assessments]
    liveness = [item["liveness"] for item in assessments]
    high_cycles = sum(item.level == "ALTO" for item in risks)
    moderate_cycles = sum(item.level == "MODERADO" for item in risks)
    compatible_cycles = sum(item.compatible for item in identities)
    valid_liveness = sum(item.valid for item in liveness)
    fake_values = [item.probability for item in deepfakes if item.probability is not None]
    calibrated_values = [
        item.calibrated_probability
        for item in deepfakes
        if item.calibrated_probability is not None
    ]
    average_fake_probability = (
        sum(fake_values) / len(fake_values) if fake_values else None
    )
    identity_diverged = compatible_cycles < len(identities) / 2
    average_calibrated_probability = (
        sum(calibrated_values) / len(calibrated_values) if calibrated_values else None
    )
    high_manipulation = (
        average_calibrated_probability is not None
        and average_calibrated_probability >= 0.80
    )
    maximum_risk = max(item.accumulated for item in risks)

    if high_cycles or identity_diverged or high_manipulation:
        status = "ATENCAO"
        title = "Foram encontrados sinais relevantes"
        if identity_diverged and high_manipulation:
            message = (
                "A identidade observada divergiu da referencia cadastrada e o detector "
                "encontrou forte indicio de manipulacao facial durante a sessao."
            )
        elif high_manipulation:
            message = "O detector encontrou forte indicio de manipulacao facial durante a sessao."
        else:
            message = "A identidade observada divergiu da referencia cadastrada durante a sessao."
    elif moderate_cycles:
        status = "REVISAR"
        title = "A sessao apresentou pontos para revisao"
        message = "Os sinais nao sao conclusivos, mas merecem verificacao do recrutador."
    else:
        status = "SEM_ALERTAS_RELEVANTES"
        title = "Nenhum alerta relevante foi identificado"
        message = "O resultado resume os sinais observados durante a sessao."

    return {
        "profile": profile_name,
        "status": status,
        "title": title,
        "message": message,
        "cycles": total_periods,
        "completed_cycles": len(assessments),
        "inconclusive_cycles": max(total_periods - len(assessments), 0),
        "high_cycles": high_cycles,
        "moderate_cycles": moderate_cycles,
        "identity_compatible_percent": round(100 * compatible_cycles / len(identities), 2),
        "liveness_valid_percent": round(100 * valid_liveness / len(liveness), 2),
        "deepfake_average_percent": (
            round(100 * average_fake_probability, 2)
            if average_fake_probability is not None
            else None
        ),
        "deepfake_calibrated_average_percent": (
            round(100 * average_calibrated_probability, 2)
            if average_calibrated_probability is not None
            else None
        ),
        "maximum_accumulated_risk": round(maximum_risk, 2),
    }
