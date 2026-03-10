import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el dataset que generaste en el Punto 1
print("Cargando datos...")
df = pd.read_csv("wesad_features_punto1.csv")

# ==========================================
# 2. Análisis de Balance de Clases
# ==========================================
print("\n=== Distribución de Clases ===")
conteo = df['Label'].value_counts()
porcentaje = df['Label'].value_counts(normalize=True) * 100
for clase, (cant, porc) in enumerate(zip(conteo, porcentaje)):
    nombre = "Estrés" if clase == 1 else "No-Estrés"
    print(f"Clase {clase} ({nombre}): {cant} ventanas ({porc:.2f}%)")

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Label', palette='Set2')
plt.title('Balance de Clases (0: No-Estrés, 1: Estrés)')
plt.show()

# ==========================================
# 3. Distribución de Características y Outliers
# ==========================================
# Excluimos las columnas que no son señales numéricas para los boxplots
features = ['EDA_SCR', 'EDA_AUC', 'EDA_Tonic', 'HRV_RMSSD', 'HRV_SDNN', 'HRV_LFHF']

plt.figure(figsize=(15, 10))
for i, col in enumerate(features, 1):
    plt.subplot(2, 3, i)
    # Comparamos la distribución de cada característica entre las dos clases
    sns.boxplot(data=df, x='Label', y=col, palette='Set2')
    plt.title(f'Distribución de {col}')
plt.tight_layout()
plt.show()

# ==========================================
# 4. Matriz de Correlación
# ==========================================
plt.figure(figsize=(8, 6))
# Calculamos la correlación solo numéricas
correlacion = df[features].corr()
sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Matriz de Correlación de Características')
plt.show()