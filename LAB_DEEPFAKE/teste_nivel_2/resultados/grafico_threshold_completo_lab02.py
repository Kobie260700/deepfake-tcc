import csv
import statistics as stats
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BASE_DIR = Path(__file__).parent
TABELA_PATH = BASE_DIR / "tabela_experimental_lab02.csv"
GRAFICO_PATH = BASE_DIR / "grafico_threshold_completo_lab02.png"
CALIBRACAO_PATH = BASE_DIR / "tabela_calibracao_completa_lab02.csv"


def carregar_dados():
    with open(TABELA_PATH, newline="", encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo, delimiter=";"))

    if len(linhas) != 20:
        raise ValueError("A tabela experimental deve conter exatamente 20 execucoes.")

    campos_vazios = [
        linha["execucao"]
        for linha in linhas
        if not linha["distancia"].strip() or not linha["resultado"].strip()
    ]

    if campos_vazios:
        execucoes = ", ".join(campos_vazios)
        raise ValueError(
            "Preencha distancia e resultado na tabela para as execucoes: "
            f"{execucoes}"
        )

    for linha in linhas:
        linha["execucao"] = int(linha["execucao"])
        linha["threshold"] = float(linha["threshold"])
        linha["distancia"] = float(linha["distancia"].replace(",", "."))

    reais = [linha for linha in linhas if linha["tipo_teste"] == "Real"]
    deeplive = [linha for linha in linhas if linha["tipo_teste"] == "DeepLiveCam"]

    if len(reais) != 10 or len(deeplive) != 10:
        raise ValueError("A tabela deve conter 10 testes reais e 10 DeepLiveCam.")

    return reais, deeplive


def gerar_grafico(reais, deeplive):
    distancias_real = [linha["distancia"] for linha in reais]
    distancias_deeplive = [linha["distancia"] for linha in deeplive]

    maior_real = max(distancias_real)
    menor_deeplive = min(distancias_deeplive)
    margem_classes = menor_deeplive - maior_real
    threshold_final = (maior_real + menor_deeplive) / 2

    if margem_classes <= 0:
        raise ValueError(
            "As classes se sobrepoem: nao existe margem positiva para calcular "
            "o threshold pelo ponto medio."
        )

    limite_inferior = max(0, min(distancias_real) - margem_classes * 0.15)
    limite_superior = max(distancias_deeplive) + margem_classes * 0.15

    fig, (ax_dist, ax_regua) = plt.subplots(
        1,
        2,
        figsize=(17, 6.8),
        gridspec_kw={"width_ratios": [2.35, 1]}
    )

    fig.suptitle("LAB02 - Definição Experimental do Threshold", fontsize=16)

    execucoes_real = [linha["execucao"] for linha in reais]
    execucoes_deeplive = [linha["execucao"] for linha in deeplive]

    ax_dist.plot(
        execucoes_real,
        distancias_real,
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        label="Usuário real - 10 testes"
    )

    ax_dist.plot(
        execucoes_deeplive,
        distancias_deeplive,
        marker="o",
        linewidth=2.5,
        color="#d62728",
        label="DeepLiveCam - 10 testes"
    )

    for linha in reais:
        ax_dist.annotate(
            f'D={linha["distancia"]:.3f}\nT={linha["threshold"]:.0f}',
            (linha["execucao"], linha["distancia"]),
            textcoords="offset points",
            xytext=(0, -24),
            ha="center",
            fontsize=7.5,
            color="#1f77b4"
        )

    for linha in deeplive:
        ax_dist.annotate(
            f'D={linha["distancia"]:.3f}\nT={linha["threshold"]:.0f}',
            (linha["execucao"], linha["distancia"]),
            textcoords="offset points",
            xytext=(0, 13),
            ha="center",
            fontsize=7.5,
            color="#d62728"
        )

    ax_dist.axhline(
        threshold_final,
        color="#2ca02c",
        linestyle="--",
        linewidth=2,
        label=f"Threshold final = {threshold_final:.3f}"
    )
    ax_dist.axhline(
        maior_real,
        color="#1f77b4",
        linestyle=":",
        linewidth=2,
        label=f"Maior distância real = {maior_real:.3f}"
    )
    ax_dist.axhline(
        menor_deeplive,
        color="#d62728",
        linestyle=":",
        linewidth=2,
        label=f"Menor distância DeepLiveCam = {menor_deeplive:.3f}"
    )
    ax_dist.fill_between(
        range(1, 21),
        maior_real,
        menor_deeplive,
        color="#2ca02c",
        alpha=0.10
    )
    ax_dist.axvline(10.5, color="#444444", linestyle="-.", linewidth=1.8)

    altura_titulo = limite_superior - (limite_superior - limite_inferior) * 0.04
    ax_dist.text(
        5.5,
        altura_titulo,
        "10 testes reais",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color="#1f77b4",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}
    )
    ax_dist.text(
        15.5,
        altura_titulo,
        "10 testes DeepLiveCam",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color="#d62728",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}
    )

    ax_dist.set_title("Distâncias obtidas nas 20 execuções experimentais")
    ax_dist.set_xlabel("Execução experimental")
    ax_dist.set_ylabel("Distância vetorial")
    ax_dist.set_xticks(range(1, 21))
    ax_dist.set_ylim(limite_inferior, limite_superior)
    ax_dist.grid(True, linestyle="--", alpha=0.35)

    handles, labels = ax_dist.get_legend_handles_labels()
    handles.append(Line2D([], [], color="none"))
    labels.append("D = distância obtida | T = threshold definido naquele teste")
    ax_dist.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2
    )

    ax_regua.set_title("Régua de decisão pelo ponto médio")
    ax_regua.set_xlim(0, 1)
    ax_regua.set_ylim(limite_inferior, limite_superior)
    ax_regua.set_xticks([])
    ax_regua.set_ylabel("Distância vetorial")
    ax_regua.grid(True, axis="y", linestyle="--", alpha=0.35)

    ax_regua.axhspan(limite_inferior, maior_real, color="#1f77b4", alpha=0.10)
    ax_regua.axhspan(maior_real, menor_deeplive, color="#2ca02c", alpha=0.12)
    ax_regua.axhspan(menor_deeplive, limite_superior, color="#d62728", alpha=0.10)

    marcadores = [
        (maior_real, "Maior real", "#1f77b4", 0.20, 12, "center"),
        (threshold_final, "Ponto médio\nThreshold final", "#2ca02c", 0.50, -46, "right"),
        (menor_deeplive, "Menor DeepLive", "#d62728", 0.80, 12, "center")
    ]

    for valor, texto, cor, x, offset_x, alinhamento in marcadores:
        ax_regua.scatter(x, valor, s=85, color=cor, zorder=3)
        ax_regua.annotate(
            f"{texto}\n{valor:.3f}",
            (x, valor),
            textcoords="offset points",
            xytext=(offset_x, 12),
            ha=alinhamento,
            fontsize=8.8,
            color=cor,
            fontweight="bold"
        )

    ax_regua.annotate(
        "",
        xy=(0.5, menor_deeplive),
        xytext=(0.5, maior_real),
        arrowprops={"arrowstyle": "<->", "color": "#2ca02c", "linewidth": 2}
    )
    ax_regua.text(
        0.62,
        threshold_final,
        f"Margem = {menor_deeplive:.3f} - {maior_real:.3f}\nMargem = {margem_classes:.3f}",
        va="center",
        fontsize=9.3,
        color="#2ca02c",
        fontweight="bold"
    )
    ax_regua.text(
        0.5,
        maior_real + margem_classes * 0.32,
        f"Threshold = ({maior_real:.3f} + {menor_deeplive:.3f}) / 2\nThreshold = {threshold_final:.3f}",
        ha="center",
        fontsize=9.3,
        color="#2ca02c",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#2ca02c", "alpha": 0.95}
    )

    plt.tight_layout()
    plt.savefig(GRAFICO_PATH, dpi=300, bbox_inches="tight")

    metricas = [
        ["Maior distância real", round(maior_real, 3)],
        ["Menor distância DeepLiveCam", round(menor_deeplive, 3)],
        ["Média real", round(stats.mean(distancias_real), 3)],
        ["Média DeepLiveCam", round(stats.mean(distancias_deeplive), 3)],
        ["Margem entre classes", round(margem_classes, 3)],
        ["Threshold final", round(threshold_final, 3)]
    ]

    with open(CALIBRACAO_PATH, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")
        escritor.writerow(["Métrica", "Valor"])
        escritor.writerows(metricas)

    print(f"[INFO] Gráfico salvo em: {GRAFICO_PATH}")
    print(f"[INFO] Tabela de calibração salva em: {CALIBRACAO_PATH}")


def main():
    reais, deeplive = carregar_dados()
    gerar_grafico(reais, deeplive)


if __name__ == "__main__":
    main()
