import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


BASE_DIR = Path(__file__).parent
LOG_PATH = BASE_DIR.parent / "logs" / "autenticacao_liveness_ativo_lab04.csv"
GRAFICO_PATH = BASE_DIR / "grafico_metricas_lab04.png"
RESUMO_PATH = BASE_DIR / "resumo_metricas_lab04.csv"
THRESHOLD_IDENTIDADE = 20.210
TOTAL_ACOES = 4


def converter_booleano(valor):
    return valor.strip().lower() == "true"


def carregar_primeiras_vinte_tentativas():
    with open(LOG_PATH, newline="", encoding="utf-8-sig") as arquivo:
        linhas = [linha for linha in csv.DictReader(arquivo) if linha["data_hora"].strip()]

    reais = [linha for linha in linhas if linha["fonte_camera"] == "webcam_fisica"][:10]
    deeplive = [linha for linha in linhas if linha["fonte_camera"] == "camera_virtual"][:10]

    if len(reais) < 10 or len(deeplive) < 10:
        raise ValueError("São necessárias 10 tentativas reais e 10 DeepLiveCam.")

    selecionadas = reais + deeplive

    for execucao, linha in enumerate(selecionadas, 1):
        linha["execucao"] = execucao
        linha["distancia_media"] = float(linha["distancia_media"])
        linha["tempo_total_segundos"] = float(linha["tempo_total_segundos"])
        linha["identidade_valida"] = converter_booleano(linha["identidade_valida"])
        linha["challenge_valido"] = converter_booleano(linha["challenge_valido"])
        linha["quantidade_acoes"] = len([
            acao for acao in linha["acoes_concluidas"].split(" > ") if acao.strip()
        ])

    return reais, deeplive


def main():
    reais, deeplive = carregar_primeiras_vinte_tentativas()
    linhas = reais + deeplive

    execucoes_real = [linha["execucao"] for linha in reais]
    execucoes_deeplive = [linha["execucao"] for linha in deeplive]
    distancias_real = [linha["distancia_media"] for linha in reais]
    distancias_deeplive = [linha["distancia_media"] for linha in deeplive]
    tempos_real = [linha["tempo_total_segundos"] for linha in reais]
    tempos_deeplive = [linha["tempo_total_segundos"] for linha in deeplive]

    fig, (ax_metricas, ax_challenge) = plt.subplots(
        1,
        2,
        figsize=(18, 7),
        gridspec_kw={"width_ratios": [3.0, 1.25]}
    )
    ax_tempo = ax_metricas.twinx()

    fig.suptitle(
        "LAB04 - Métricas do Liveness Ativo: Real x DeepLiveCam",
        fontsize=16
    )

    ax_metricas.plot(
        execucoes_real,
        distancias_real,
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        label="Distância - Real"
    )
    ax_metricas.plot(
        execucoes_deeplive,
        distancias_deeplive,
        marker="o",
        linewidth=2.5,
        color="#d62728",
        label="Distância - DeepLiveCam"
    )
    ax_metricas.axhline(
        THRESHOLD_IDENTIDADE,
        color="#2ca02c",
        linestyle="--",
        linewidth=2,
        label=f"Threshold de identidade = {THRESHOLD_IDENTIDADE:.3f}"
    )

    ax_tempo.plot(
        execucoes_real,
        tempos_real,
        marker="s",
        linewidth=2.1,
        linestyle="--",
        color="#17becf",
        label="Tempo total - Real"
    )
    ax_tempo.plot(
        execucoes_deeplive,
        tempos_deeplive,
        marker="s",
        linewidth=2.1,
        linestyle="--",
        color="#ff7f0e",
        label="Tempo total - DeepLiveCam"
    )

    for linha in linhas:
        cor_distancia = "#1f77b4" if linha["fonte_camera"] == "webcam_fisica" else "#d62728"
        cor_tempo = "#17becf" if linha["fonte_camera"] == "webcam_fisica" else "#ff7f0e"
        deslocamento_d = -20 if linha["fonte_camera"] == "webcam_fisica" else 9
        deslocamento_t = 9 if linha["fonte_camera"] == "webcam_fisica" else -24

        ax_metricas.annotate(
            f"D={linha['distancia_media']:.3f}",
            (linha["execucao"], linha["distancia_media"]),
            textcoords="offset points",
            xytext=(0, deslocamento_d),
            ha="center",
            fontsize=7.0,
            color=cor_distancia
        )
        ax_tempo.annotate(
            f"T={linha['tempo_total_segundos']:.3f}s",
            (linha["execucao"], linha["tempo_total_segundos"]),
            textcoords="offset points",
            xytext=(0, deslocamento_t),
            ha="center",
            fontsize=7.0,
            color=cor_tempo
        )

    ax_metricas.axvline(10.5, color="#444444", linestyle="-.", linewidth=1.7)
    ax_metricas.text(5.5, 36.3, "10 testes reais", ha="center", color="#1f77b4", fontweight="bold")
    ax_metricas.text(15.5, 36.3, "10 testes DeepLiveCam", ha="center", color="#d62728", fontweight="bold")
    ax_metricas.set_title("Distância biométrica e tempo total por tentativa")
    ax_metricas.set_xlabel("Execução experimental")
    ax_metricas.set_ylabel("Distância biométrica média (L2)")
    ax_tempo.set_ylabel("Tempo total ou parcial (segundos)", color="#7c3aed")
    ax_metricas.set_xticks(range(1, 21))
    ax_metricas.set_ylim(10, 38)
    ax_tempo.set_ylim(0, max(tempos_real + tempos_deeplive) + 3)
    ax_metricas.grid(True, linestyle="--", alpha=0.30)

    handles_1, labels_1 = ax_metricas.get_legend_handles_labels()
    handles_2, labels_2 = ax_tempo.get_legend_handles_labels()
    ax_metricas.legend(
        handles_1 + handles_2 + [Line2D([], [], color="none")],
        labels_1 + labels_2 + ["D = distância | T = tempo total; em falhas, tempo parcial"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3
    )

    execucoes = np.arange(1, 21)
    acoes = [linha["quantidade_acoes"] for linha in linhas]
    cores = ["#2ca02c" if linha["challenge_valido"] else "#d62728" for linha in linhas]
    barras = ax_challenge.bar(execucoes, acoes, color=cores, width=0.72)

    for indice, (barra, valor, linha) in enumerate(zip(barras, acoes, linhas), 1):
        if linha["challenge_valido"]:
            deslocamento = 3 if indice % 2 else 9
            ax_challenge.annotate(
                f"{valor}/4",
                (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                textcoords="offset points",
                xytext=(0, deslocamento),
                ha="center",
                va="bottom",
                fontsize=7.5
            )
        else:
            ax_challenge.annotate(
                f"{valor}/4",
                (barra.get_x() + barra.get_width() / 2, barra.get_height() - 0.10),
                ha="center",
                va="top",
                fontsize=8.5,
                fontweight="bold",
                color="white"
            )
    ax_challenge.axvline(10.5, color="#444444", linestyle="-.", linewidth=1.5)
    ax_challenge.set_title("Ações concluídas no challenge")
    ax_challenge.set_xlabel("Execução")
    ax_challenge.set_ylabel("Quantidade de ações")
    ax_challenge.set_xticks(range(1, 21))
    ax_challenge.tick_params(axis="x", labelrotation=90)
    ax_challenge.set_ylim(0, 5.05)
    ax_challenge.grid(True, axis="y", linestyle="--", alpha=0.30)
    ax_challenge.legend(
        handles=[
            Line2D([0], [0], color="#2ca02c", linewidth=8, label="Challenge válido"),
            Line2D([0], [0], color="#d62728", linewidth=8, label="Challenge inválido")
        ],
        loc="lower center"
    )

    desafios_reais = sum(linha["challenge_valido"] for linha in reais)
    desafios_deeplive = sum(linha["challenge_valido"] for linha in deeplive)
    ax_challenge.text(
        0.5,
        0.58,
        f"Challenge concluído\nReal: {desafios_reais}/10\nDeepLiveCam: {desafios_deeplive}/10",
        transform=ax_challenge.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#6b7280",
            "alpha": 0.92
        }
    )

    plt.tight_layout()
    plt.savefig(GRAFICO_PATH, dpi=300, bbox_inches="tight")

    with open(RESUMO_PATH, "w", newline="", encoding="utf-8") as arquivo:
        writer = csv.writer(arquivo, delimiter=";")
        writer.writerow([
            "execucao",
            "tipo_teste",
            "distancia_media",
            "tempo_total_ou_parcial",
            "acoes_concluidas",
            "challenge_valido",
            "resultado"
        ])
        for linha in linhas:
            writer.writerow([
                linha["execucao"],
                "Real" if linha["fonte_camera"] == "webcam_fisica" else "DeepLiveCam",
                linha["distancia_media"],
                linha["tempo_total_segundos"],
                linha["quantidade_acoes"],
                linha["challenge_valido"],
                linha["resultado"]
            ])

    print(f"[INFO] Gráfico salvo em: {GRAFICO_PATH}")
    print(f"[INFO] Resumo salvo em: {RESUMO_PATH}")
    print(f"[INFO] Challenge real: {desafios_reais}/10")
    print(f"[INFO] Challenge DeepLiveCam: {desafios_deeplive}/10")


if __name__ == "__main__":
    main()
