import csv
import statistics as stats
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================
# Dados experimentais LAB01
# =========================

thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]

thresholds_real = [
    0.45, 0.45, 0.50, 0.50, 0.55,
    0.55, 0.60, 0.60, 0.65, 0.65
]

thresholds_deeplive = [
    0.45, 0.45, 0.50, 0.50, 0.55,
    0.55, 0.60, 0.60, 0.65, 0.65
]

distancias_real = [
    0.228, 0.237, 0.218, 0.224, 0.218,
    0.216, 0.207, 0.228, 0.218, 0.222
]

distancias_deeplive = [
    0.681, 0.662, 0.680, 0.670, 0.688,
    0.668, 0.684, 0.686, 0.687, 0.668
]

# =========================
# Calibracao
# =========================

maior_real = max(distancias_real)
menor_deeplive = min(distancias_deeplive)
media_real = stats.mean(distancias_real)
media_deeplive = stats.mean(distancias_deeplive)
margem_classes = menor_deeplive - maior_real
threshold_final = (maior_real + menor_deeplive) / 2


# =========================
# Caminhos
# =========================

BASE_DIR = Path(__file__).parent
grafico_path = BASE_DIR / "grafico_threshold_completo_lab01.png"
tabela_path = BASE_DIR / "tabela_calibracao_completa_lab01.csv"


# =========================
# Grafico
# =========================

fig, (ax_dist, ax_regua) = plt.subplots(
    1,
    2,
    figsize=(17, 6.8),
    gridspec_kw={"width_ratios": [2.35, 1]}
)

fig.suptitle("LAB01 - Definição Experimental do Threshold", fontsize=16)


# -------------------------
# Painel 1: 20 execucoes
# -------------------------

execucoes_real = list(range(1, 11))
execucoes_deeplive = list(range(11, 21))

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

for x, y, threshold in zip(execucoes_real, distancias_real, thresholds_real):
    ax_dist.annotate(
        f"D={y:.3f}\nT={threshold:.2f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, -24),
        ha="center",
        fontsize=7.5,
        color="#1f77b4"
    )

for x, y, threshold in zip(execucoes_deeplive, distancias_deeplive, thresholds_deeplive):
    ax_dist.annotate(
        f"D={y:.3f}\nT={threshold:.2f}",
        (x, y),
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
    label=f"Threshold final = {threshold_final:.2f}"
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

ax_dist.axvline(
    10.5,
    color="#444444",
    linestyle="-.",
    linewidth=1.8
)

ax_dist.text(
    5.5,
    0.755,
    "10 testes reais",
    ha="center",
    fontsize=10.5,
    fontweight="bold",
    color="#1f77b4",
    bbox={
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.85
    }
)

ax_dist.text(
    15.5,
    0.755,
    "10 testes DeepLiveCam",
    ha="center",
    fontsize=10.5,
    fontweight="bold",
    color="#d62728",
    bbox={
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.85
    }
)

ax_dist.set_title("Distâncias obtidas nas 20 execuções experimentais")
ax_dist.set_xlabel("Execução experimental")
ax_dist.set_ylabel("Distância facial")
ax_dist.set_xticks(range(1, 21))
ax_dist.set_ylim(0.14, 0.80)
ax_dist.grid(True, linestyle="--", alpha=0.35)

handles, labels = ax_dist.get_legend_handles_labels()
handles.append(Line2D([], [], color="none"))
labels.append("D = distância obtida | T = threshold definido naquele teste")
ax_dist.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)


# -------------------------
# Painel 2: Regua de calibracao
# -------------------------

ax_regua.set_title("Régua de decisão pelo ponto médio")
ax_regua.set_xlim(0, 1)
ax_regua.set_ylim(0.15, 0.75)
ax_regua.set_xticks([])
ax_regua.set_ylabel("Distância facial")
ax_regua.grid(True, axis="y", linestyle="--", alpha=0.35)

ax_regua.axhspan(
    0.15,
    maior_real,
    color="#1f77b4",
    alpha=0.10,
    label="Faixa real"
)

ax_regua.axhspan(
    menor_deeplive,
    0.75,
    color="#d62728",
    alpha=0.10,
    label="Faixa DeepLiveCam"
)

ax_regua.axhspan(
    maior_real,
    menor_deeplive,
    color="#2ca02c",
    alpha=0.12,
    label="Margem entre classes"
)

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
    arrowprops={
        "arrowstyle": "<->",
        "color": "#2ca02c",
        "linewidth": 2
    }
)

ax_regua.text(
    0.62,
    (maior_real + menor_deeplive) / 2,
    f"Margem = {menor_deeplive:.3f} - {maior_real:.3f}\nMargem = {margem_classes:.3f}",
    va="center",
    fontsize=9.3,
    color="#2ca02c",
    fontweight="bold"
)

ax_regua.text(
    0.5,
    0.37,
    f"Threshold = ({maior_real:.3f} + {menor_deeplive:.3f}) / 2\nThreshold = {threshold_final:.3f} ~= 0.45",
    ha="center",
    fontsize=9.3,
    color="#2ca02c",
    fontweight="bold",
    bbox={
        "boxstyle": "round,pad=0.35",
        "facecolor": "white",
        "edgecolor": "#2ca02c",
        "alpha": 0.95
    }
)

ax_regua.text(
    0.5,
    0.18,
    "Real aceito\nabaixo do limite",
    ha="center",
    fontsize=8.8,
    color="#1f77b4"
)

ax_regua.text(
    0.5,
    0.705,
    "DeepLiveCam rejeitado\nacima do limite",
    ha="center",
    fontsize=8.8,
    color="#d62728",
    bbox={
        "boxstyle": "round,pad=0.2",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.80
    }
)

plt.tight_layout()
plt.savefig(grafico_path, dpi=300, bbox_inches="tight")


# =========================
# Tabela de apoio
# =========================

metricas = [
    ["Maior distancia real", round(maior_real, 3)],
    ["Menor distancia DeepLiveCam", round(menor_deeplive, 3)],
    ["Media real", round(media_real, 3)],
    ["Media DeepLiveCam", round(media_deeplive, 3)],
    ["Margem entre classes", round(margem_classes, 3)],
    ["Threshold final", round(threshold_final, 3)]
]

with open(tabela_path, "w", newline="", encoding="utf-8") as arquivo:
    escritor = csv.writer(arquivo, delimiter=";")
    escritor.writerow(["Metrica", "Valor"])
    escritor.writerows(metricas)

print("[INFO] Grafico salvo em:")
print(grafico_path)

print("[INFO] Tabela salva em:")
print(tabela_path)
