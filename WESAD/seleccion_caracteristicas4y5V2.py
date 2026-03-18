import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# CONFIGURACIÓN VISUAL GLOBAL
# ==============================
sns.set_theme(style="whitegrid", context="talk")

def add_value_labels(ax):
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.3f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=9,
                    xytext=(0, 5),
                    textcoords='offset points')

# ==============================
# 1. CARGAR DATOS
# ==============================
df = pd.read_csv("./lib/WESAD/wesad_preprocesado_punto3.csv")

print("Dataset cargado:", df.shape)

features = [col for col in df.columns if col not in ['Time', 'Label', 'Subject']]
label_col = 'Label'

# ==============================
# 2. FACTOR DE FISHER
# ==============================
def fisher_score(df, feature, label_col):
    clases = df[label_col].unique()
    
    mu_k = df[feature].mean()
    num = 0
    den = 0
    
    for c in clases:
        df_c = df[df[label_col] == c]
        pj = len(df_c) / len(df)
        mu_kj = df_c[feature].mean()
        sigma_kj = df_c[feature].std()
        
        num += pj * (mu_kj - mu_k) ** 2
        den += pj * (sigma_kj ** 2)
    
    return num / den if den != 0 else 0

# Calcular Fisher
fisher_scores = {f: fisher_score(df, f, label_col) for f in features}

fisher_df = pd.DataFrame.from_dict(fisher_scores, orient='index', columns=['Fisher'])
fisher_df = fisher_df.sort_values(by='Fisher', ascending=False)

print("\nRanking Fisher:\n", fisher_df)

# ==============================
# 3. GRAFICA RANKING FISHER
# ==============================
plt.figure(figsize=(12,6))

ax = sns.barplot(
    x=fisher_df.index,
    y=fisher_df['Fisher'],
    palette="viridis"
)

plt.title("Ranking de Características según Factor de Fisher", fontsize=16)
plt.xlabel("Características")
plt.ylabel("Score de Fisher")
plt.xticks(rotation=45, ha='right')

add_value_labels(ax)
sns.despine()
plt.tight_layout()
plt.show()

# ==============================
# 4. SELECCIÓN ESCALAR HACIA ADELANTE
# ==============================
alpha1 = 1
alpha2 = 1

selected = []
remaining = list(fisher_df.index)
selection_scores = []

# ---- Iteración 1
s1 = fisher_df.index[0]
selected.append(s1)
remaining.remove(s1)
selection_scores.append(fisher_scores[s1])

print("\nIteración 1 - Seleccionada:", s1)

# ==============================
# Función correlación promedio
# ==============================
def mean_abs_correlation(df, feature, selected):
    corrs = []
    for s in selected:
        corr = np.corrcoef(df[feature], df[s])[0,1]
        corrs.append(abs(corr))
    return np.mean(corrs)

# ==============================
# Iteraciones siguientes
# ==============================
k_max = 5

for k in range(2, k_max + 1):
    scores = {}
    corr_plot = {}

    for f in remaining:
        corr = mean_abs_correlation(df, f, selected)
        corr_plot[f] = corr

        score = alpha1 * fisher_scores[f] - (alpha2 / (k - 1)) * corr
        scores[f] = score

    # Ordenar correlaciones
    corr_sorted = dict(sorted(corr_plot.items(), key=lambda x: x[1], reverse=True))

    # Selección
    best_feature = max(scores, key=scores.get)
    best_score = scores[best_feature]

    # ==============================
    # GRAFICA CORRELACIÓN (MEJORADA)
    # ==============================
    plt.figure(figsize=(12,5))

    colors = [
        "#d62728" if f == best_feature else "#1f77b4"
        for f in corr_sorted.keys()
    ]

    ax = sns.barplot(
        x=list(corr_sorted.keys()),
        y=list(corr_sorted.values()),
        palette=colors
    )

    plt.title(f"Redundancia (|Pearson|) con características seleccionadas\nIteración {k}", fontsize=15)
    plt.xlabel("Características restantes")
    plt.ylabel("Correlación promedio absoluta")
    plt.xticks(rotation=45, ha='right')

    add_value_labels(ax)
    sns.despine()
    plt.tight_layout()
    plt.show()

    # Actualizar listas
    selected.append(best_feature)
    remaining.remove(best_feature)
    selection_scores.append(best_score)

    print(f"Iteración {k} - Seleccionada: {best_feature} | Score: {best_score:.4f}")

# ==============================
# 5. RESULTADOS FINALES
# ==============================
print("\nTop 5 características seleccionadas:")
print(selected)

# ==============================
# 6. GRAFICA FINAL (SFS REAL)
# ==============================
plt.figure(figsize=(12,6))

labels = [f"{i+1}. {feat}" for i, feat in enumerate(selected)]

ax = sns.barplot(
    x=labels,
    y=selection_scores,
    palette="flare"
)

plt.title("Selección Escalar hacia Adelante\n(Score = Fisher - Redundancia)", fontsize=16)
plt.xlabel("Orden de selección")
plt.ylabel("Score del algoritmo")
plt.xticks(rotation=45, ha='right')

add_value_labels(ax)
sns.despine()
plt.tight_layout()
plt.show()