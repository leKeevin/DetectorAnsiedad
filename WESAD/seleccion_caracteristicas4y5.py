# ==========================================================
# ANALISIS DE CARACTERISTICAS WESAD
# Paso 4: Ranking usando Factor de Fisher
# Paso 5: Seleccion Escalar hacia Adelante
# ==========================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# ==========================================================
# CONFIGURACION VISUAL PROFESIONAL
# ==========================================================

sns.set_theme(
    style="whitegrid",
    context="talk",
    palette="deep"
)

plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams["figure.dpi"] = 120

# ==========================================================
# 1. CARGAR DATASET
# ==========================================================

df = pd.read_csv("./lib/WESAD/wesad_preprocesado_punto3.csv")

print("\nDataset cargado correctamente")
print("Dimensiones:", df.shape)

# eliminar columnas que no son caracteristicas
X = df.drop(columns=["Time","Subject","Label"])
y = df["Label"]

features = list(X.columns)

print("\nCaracteristicas analizadas:")
print(features)

# ==========================================================
# 2. CALCULO DEL FACTOR DE FISHER
# ==========================================================

def fisher_score(feature, X, y):

    classes = np.unique(y)
    N = len(y)

    mu_k = np.mean(X[feature])

    numerator = 0
    denominator = 0

    for c in classes:

        X_c = X[y == c][feature]

        pj = len(X_c) / N
        mu_kj = np.mean(X_c)
        sigma_kj = np.std(X_c)

        numerator += pj * (mu_kj - mu_k)**2
        denominator += pj * (sigma_kj**2)

    if denominator == 0:
        return 0

    return numerator / denominator


fisher_scores = {}

for feature in features:
    fisher_scores[feature] = fisher_score(feature, X, y)

fisher_df = pd.DataFrame({
    "Feature": fisher_scores.keys(),
    "FisherScore": fisher_scores.values()
})

fisher_df = fisher_df.sort_values(by="FisherScore", ascending=False)

print("\n=================================")
print("Ranking de Caracteristicas (Fisher)")
print("=================================\n")

print(fisher_df)

# ==========================================================
# 3. GRAFICA PROFESIONAL DEL RANKING
# ==========================================================

plt.figure()

sns.barplot(
    data=fisher_df,
    x="FisherScore",
    y="Feature",
    palette="viridis"
)

plt.title("Ranking de Caracteristicas mediante Factor de Fisher", fontsize=16)
plt.xlabel("Fisher Score")
plt.ylabel("Caracteristica")

plt.tight_layout()

plt.savefig("ranking_fisher.png", dpi=300, bbox_inches="tight")

plt.show()

# ==========================================================
# 4. SELECCION ESCALAR HACIA ADELANTE
# ==========================================================

model = RandomForestClassifier(random_state=42)

selected_features = []
remaining_features = features.copy()

performance_history = []

print("\n=================================")
print("Proceso de Forward Feature Selection")
print("=================================\n")

while len(selected_features) < 5:

    best_feature = None
    best_score = 0

    for feature in remaining_features:

        current_features = selected_features + [feature]

        score = cross_val_score(
            model,
            X[current_features],
            y,
            cv=5,
            scoring="accuracy"
        ).mean()

        if score > best_score:
            best_score = score
            best_feature = feature

    selected_features.append(best_feature)
    remaining_features.remove(best_feature)

    performance_history.append(best_score)

    print("Feature agregada:", best_feature)
    print("Accuracy promedio (CV):", round(best_score,4))
    print()

print("=================================")
print("Top 5 Caracteristicas Seleccionadas")
print("=================================\n")

print(selected_features)

# ==========================================================
# 5. GRAFICA DEL DESEMPEÑO DEL MODELO
# ==========================================================

plt.figure()

sns.lineplot(
    x=range(1,6),
    y=performance_history,
    marker="o",
    linewidth=3
)

plt.title("Desempeño del Modelo durante Forward Feature Selection", fontsize=16)

plt.xlabel("Numero de caracteristicas seleccionadas")
plt.ylabel("Accuracy (Validacion Cruzada)")

plt.xticks(range(1,6))

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("forward_selection_accuracy.png", dpi=300, bbox_inches="tight")

plt.show()

# ==========================================================
# 6. GRAFICA: CARACTERISTICAS AGREGADAS DURANTE FORWARD SELECTION
# ==========================================================

forward_df = pd.DataFrame({
    "Feature": selected_features,
    "Accuracy": performance_history,
    "Step": range(1, len(selected_features) + 1)
})

plt.figure()

sns.lineplot(
    data=forward_df,
    x="Step",
    y="Accuracy",
    marker="o",
    linewidth=3
)

# Etiquetas con el nombre de la característica en cada punto
for i in range(len(forward_df)):
    plt.text(
        forward_df["Step"][i],
        forward_df["Accuracy"][i] + 0.002,
        forward_df["Feature"][i],
        ha="center",
        fontsize=11
    )

plt.title("Características seleccionadas durante Forward Selection", fontsize=16)

plt.xlabel("Iteración del algoritmo")
plt.ylabel("Accuracy (Validación Cruzada)")

plt.xticks(forward_df["Step"])

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("forward_selection_features.png", dpi=300, bbox_inches="tight")

plt.show()