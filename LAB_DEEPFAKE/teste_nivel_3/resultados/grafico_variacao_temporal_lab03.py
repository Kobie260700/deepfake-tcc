import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


BASE_DIR = Path(__file__).parent
TABELA_PATH = BASE_DIR / "tabela_variacao_temporal_lab03.csv"
GRAFICO_PATH = BASE_DIR / "grafico_variacao_temporal_lab03.png"
RESUMO_PATH = BASE_DIR / "resumo_variacao_temporal_lab03.csv"
THRESHOLD_IDENTIDADE = 20.210


def carregar_dados():
    with open(TABELA_PATH, newline="", encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo, delimiter=";"))

    for linha in linhas:
        linha["execucao"] = int(linha["execucao"])
        linha["min_variacao_testada"] = float(linha["min_variacao_testada"])
        linha["distancia_media"] = float(linha["distancia_media"])
        linha["variacao_temporal"] = float(linha["variacao_temporal"])

    reais = [linha for linha in linhas if linha["tipo_teste"] == "Real"]
    deeplive = [linha for linha in linhas if linha["tipo_teste"] == "DeepLiveCam"]
    return reais, deeplive


def calcular_desempenho(reais, deeplive, thresholds):
    desempenho = []

    for threshold in thresholds:
        aceitos_reais = sum(
            linha["variacao_temporal"] >= threshold for linha in reais
        )
        rejeitados_deeplive = sum(
            linha["variacao_temporal"] < threshold for linha in deeplive
        )
        desempenho.append({
            "threshold": threshold,
            "aceitacao_real": aceitos_reais / len(reais) * 100,
            "rejeicao_deeplive": rejeitados_deeplive / len(deeplive) * 100
        })

    return desempenho


def main():
    reais, deeplive = carregar_dados()
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    desempenho = calcular_desempenho(reais, deeplive, thresholds)

    execucoes_real = [linha["execucao"] for linha in reais]
    execucoes_deeplive = [linha["execucao"] for linha in deeplive]
    distancias_real = [linha["distancia_media"] for linha in reais]
    distancias_deeplive = [linha["distancia_media"] for linha in deeplive]
    variacoes_real = [linha["variacao_temporal"] for linha in reais]
    variacoes_deeplive = [linha["variacao_temporal"] for linha in deeplive]

    menor_real = min(variacoes_real)
    maior_deeplive = max(variacoes_deeplive)
    sobreposicao = maior_deeplive - menor_real

    fig, (ax_principal, ax_taxas) = plt.subplots(
        1,
        2,
        figsize=(17, 7),
        gridspec_kw={"width_ratios": [3.2, 1]}
    )
    ax_variacao = ax_principal.twinx()

    fig.suptitle(
        "LAB03 - Distância Biométrica e Variação Temporal: Real x DeepLiveCam",
        fontsize=16
    )

    # Distância biométrica no eixo esquerdo.
    ax_principal.plot(
        execucoes_real,
        distancias_real,
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        label="Distância - Real"
    )
    ax_principal.plot(
        execucoes_deeplive,
        distancias_deeplive,
        marker="o",
        linewidth=2.5,
        color="#d62728",
        label="Distância - DeepLiveCam"
    )
    ax_principal.axhline(
        THRESHOLD_IDENTIDADE,
        color="#2ca02c",
        linestyle="--",
        linewidth=2,
        label=f"Threshold de identidade = {THRESHOLD_IDENTIDADE:.3f}"
    )

    # Variação temporal no eixo direito.
    ax_variacao.plot(
        execucoes_real,
        variacoes_real,
        marker="s",
        linewidth=2.2,
        linestyle="--",
        color="#17becf",
        label="Variação - Real"
    )
    ax_variacao.plot(
        execucoes_deeplive,
        variacoes_deeplive,
        marker="s",
        linewidth=2.2,
        linestyle="--",
        color="#ff7f0e",
        label="Variação - DeepLiveCam"
    )

    for linha in reais + deeplive:
        cor_distancia = "#1f77b4" if linha["tipo_teste"] == "Real" else "#d62728"
        cor_variacao = "#17becf" if linha["tipo_teste"] == "Real" else "#ff7f0e"
        deslocamento_d = -21 if linha["tipo_teste"] == "Real" else 9
        deslocamento_v = 10 if linha["tipo_teste"] == "Real" else -27

        ax_principal.annotate(
            f"D={linha['distancia_media']:.3f}",
            (linha["execucao"], linha["distancia_media"]),
            textcoords="offset points",
            xytext=(0, deslocamento_d),
            ha="center",
            fontsize=7.1,
            color=cor_distancia
        )
        ax_variacao.annotate(
            f"V={linha['variacao_temporal']:.3f}\nT={linha['min_variacao_testada']:.1f}",
            (linha["execucao"], linha["variacao_temporal"]),
            textcoords="offset points",
            xytext=(0, deslocamento_v),
            ha="center",
            fontsize=7.0,
            color=cor_variacao
        )

    ax_principal.axvline(10.5, color="#444444", linestyle="-.", linewidth=1.7)
    ax_principal.text(
        5.5,
        36.3,
        "10 testes reais",
        ha="center",
        color="#1f77b4",
        fontweight="bold"
    )
    ax_principal.text(
        15.5,
        36.3,
        "10 testes DeepLiveCam",
        ha="center",
        color="#d62728",
        fontweight="bold"
    )

    ax_principal.set_title("Duas camadas de decisão nas 20 execuções")
    ax_principal.set_xlabel("Execução experimental")
    ax_principal.set_ylabel("Distância biométrica média (L2)", color="#374151")
    ax_variacao.set_ylabel("Variação temporal obtida", color="#7c3aed")
    ax_principal.set_xticks(range(1, 21))
    ax_principal.set_ylim(10, 38)
    ax_variacao.set_ylim(1, 8)
    ax_principal.grid(True, linestyle="--", alpha=0.30)

    handles_1, labels_1 = ax_principal.get_legend_handles_labels()
    handles_2, labels_2 = ax_variacao.get_legend_handles_labels()
    explicacao = Line2D([], [], color="none")
    ax_principal.legend(
        handles_1 + handles_2 + [explicacao],
        labels_1 + labels_2 + ["D = distância | V = variação obtida | T = mínimo testado"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3
    )

    # Taxas do liveness para cada valor testado.
    x = np.arange(len(thresholds))
    largura = 0.36
    aceitacao = [item["aceitacao_real"] for item in desempenho]
    rejeicao = [item["rejeicao_deeplive"] for item in desempenho]

    barras_real = ax_taxas.bar(
        x - largura / 2,
        aceitacao,
        largura,
        color="#1f77b4",
        label="Aceitação real"
    )
    barras_deeplive = ax_taxas.bar(
        x + largura / 2,
        rejeicao,
        largura,
        color="#d62728",
        label="Rejeição pelo liveness"
    )
    ax_taxas.bar_label(barras_real, fmt="%.0f%%", padding=3, fontsize=8)
    ax_taxas.bar_label(barras_deeplive, fmt="%.0f%%", padding=3, fontsize=8)
    ax_taxas.set_title("Efeito do MIN_VARIACAO_TEMPORAL")
    ax_taxas.set_xlabel("Threshold testado")
    ax_taxas.set_ylabel("Taxa (%)")
    ax_taxas.set_xticks(x, [f"{valor:.1f}" for valor in thresholds])
    ax_taxas.set_ylim(0, 115)
    ax_taxas.grid(True, axis="y", linestyle="--", alpha=0.30)
    ax_taxas.legend(loc="lower center")
    ax_taxas.text(
        0.5,
        0.52,
        "Identidade: separação clara.\n\nLiveness passivo:\nas variações se sobrepõem.\nO DeepLiveCam também produz\natividade temporal.",
        transform=ax_taxas.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#374151",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "#6b7280",
            "alpha": 0.94
        }
    )

    plt.tight_layout()
    plt.savefig(GRAFICO_PATH, dpi=300, bbox_inches="tight")

    with open(RESUMO_PATH, "w", newline="", encoding="utf-8") as arquivo:
        writer = csv.writer(arquivo, delimiter=";")
        writer.writerow([
            "threshold_variacao",
            "aceitacao_real_percentual",
            "rejeicao_deeplive_por_liveness_percentual"
        ])
        for item in desempenho:
            writer.writerow([
                item["threshold"],
                round(item["aceitacao_real"], 1),
                round(item["rejeicao_deeplive"], 1)
            ])
        writer.writerow([])
        writer.writerow(["threshold_identidade_fixo", THRESHOLD_IDENTIDADE])
        writer.writerow(["menor_variacao_real", round(menor_real, 4)])
        writer.writerow(["maior_variacao_deeplive", round(maior_deeplive, 4)])
        writer.writerow(["sobreposicao_variacao", round(sobreposicao, 4)])

    print(f"[INFO] Gráfico salvo em: {GRAFICO_PATH}")
    print(f"[INFO] Resumo salvo em: {RESUMO_PATH}")


if __name__ == "__main__":
    main()
