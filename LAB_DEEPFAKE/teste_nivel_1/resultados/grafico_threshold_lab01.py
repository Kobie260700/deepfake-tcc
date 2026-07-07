import matplotlib.pyplot as plt
import statistics as stats
import csv
from pathlib import Path

# =========================
# Dados experimentais LAB01
# =========================

real = [
    0.228, 0.237, 0.218, 0.224, 0.218,
    0.216, 0.207, 0.228, 0.218, 0.232
]

deeplive = [
    0.681, 0.662, 0.680, 0.670, 0.688,
    0.668, 0.684, 0.686, 0.687, 0.668
]

# =========================
# Cálculo estatístico
# =========================

maior_real = max(real)
menor_real = min(real)
media_real = stats.mean(real)
desvio_real = stats.stdev(real)

maior_deeplive = max(deeplive)
menor_deeplive = min(deeplive)
media_deeplive = stats.mean(deeplive)
desvio_deeplive = stats.stdev(deeplive)

# Threshold ideal
threshold_definido = (maior_real + menor_deeplive) / 2

# Margem entre as classes
margem_seguranca = menor_deeplive - maior_real

# =========================
# Caminhos
# =========================

BASE_DIR = Path(__file__).parent

grafico_path = BASE_DIR / "grafico_threshold_linha_lab01.png"
tabela_path = BASE_DIR / "tabela_calibracao_lab01.csv"

# =========================
# Configuração do gráfico
# =========================

plt.figure(figsize=(12, 7))

# Execuções
x_real = list(range(1, 11))
x_deeplive = list(range(11, 21))

# Plot linhas
plt.plot(
    x_real,
    real,
    marker="o",
    linewidth=2,
    label="Pessoa real"
)

plt.plot(
    x_deeplive,
    deeplive,
    marker="o",
    linewidth=2,
    label="Deep Live Cam"
)

# =========================
# Valores nos pontos
# =========================

for x, y in zip(x_real, real):
    plt.text(
        x,
        y + 0.006,
        f"{y:.3f}",
        ha="center",
        fontsize=8
    )

for x, y in zip(x_deeplive, deeplive):
    plt.text(
        x,
        y + 0.006,
        f"{y:.3f}",
        ha="center",
        fontsize=8
    )


# =========================
# Linha separação
# =========================

plt.axvline(
    x=10.5,
    linestyle=":",
    linewidth=2,
    label="Separação dos testes"
)



# =========================
# Layout
# =========================

plt.title("LAB01 — Variação das Distâncias Faciais por Execução")
plt.xlabel("Execução")
plt.ylabel("Distância facial")

plt.xticks(range(1, 21))
plt.grid(True)
plt.legend()
plt.tight_layout()

# =========================
# Salvar gráfico
# =========================

plt.savefig(grafico_path, dpi=300)
plt.show()

# =========================
# Exportar tabela CSV
# =========================

metricas = [
    ["Maior distância real", round(maior_real, 3)],
    ["Menor distância real", round(menor_real, 3)],
    ["Média real", round(media_real, 3)],
    ["Desvio padrão real", round(desvio_real, 3)],
    ["Maior distância DeepLive", round(maior_deeplive, 3)],
    ["Menor distância DeepLive", round(menor_deeplive, 3)],
    ["Média DeepLive", round(media_deeplive, 3)],
    ["Desvio padrão DeepLive", round(desvio_deeplive, 3)],
    ["Margem entre classes", round(margem_seguranca, 3)],
    ["Threshold definido", round(threshold_definido, 3)]
]

with open(tabela_path, mode="w", newline="", encoding="utf-8") as arquivo:
    escritor = csv.writer(arquivo, delimiter=";")
    escritor.writerow(["Métrica", "Valor"])
    escritor.writerows(metricas)

# =========================
# Output terminal
# =========================

print("\n===== CALIBRAÇÃO LAB01 =====")

for metrica, valor in metricas:
    print(f"{metrica}: {valor}")

print("\n[INFO] Gráfico salvo em:")
print(grafico_path)

print("\n[INFO] Tabela salva em:")
print(tabela_path)