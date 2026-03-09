import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
import os
# 1. Carga de datos
df = pd.read_csv('./lib/StressLevel/StressLevelDataset.csv')

# 2. Preparación: X (características) y y (objetivo: stress_level)
# Eliminamos 'stress_level' de X porque es lo que queremos predecir
X = df.drop(columns=['stress_level'])
y = df['stress_level']

# 3. Cálculo de Información Mutua
# discrete_features=True porque tus datos son enteros (clases/niveles)
import_scores = mutual_info_classif(X, y, discrete_features=True, random_state=42)

# 4. Crear un DataFrame para visualizar los resultados
feature_info = pd.DataFrame({'Característica': X.columns, 'Información_Mutua': import_scores})
feature_info = feature_info.sort_values(by='Información_Mutua', ascending=False)

# 5. Visualización
plt.figure(figsize=(12, 8))
sns.barplot(x='Información_Mutua', y='Característica', data=feature_info, palette='viridis')
plt.title('Capacidad Predictiva para Niveles de Estrés (Información Mutua)')
plt.xlabel('Puntaje de Información Mutua (Mayor es mejor)')
plt.ylabel('Características')

# Guardar resultado
plt.tight_layout()
plt.savefig('./results/StressLevel/ranking_estres.png')
# plt.show()

print("Ranking de las top 5 características:")
print(feature_info.head(5))


# --- NUEVO: Exportar a archivo de texto ---
txt_path = os.path.join('./results/StressLevel/', 'ranking_depresion.txt')
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("RANKING DE CARACTERÍSTICAS SEGÚN INFORMACIÓN MUTUA\n")
    f.write(f"Objetivo: Clasificar niveles de Depresión\n")
    f.write("-" * 50 + "\n")
    f.write(f"{'Característica':<30} | {'Puntaje Información_Mutua':<10}\n")
    f.write("-" * 50 + "\n")
    for _, row in feature_info.iterrows():
        f.write(f"{row['Característica']:<30} | {row['Información_Mutua']:.4f}\n")

print(f"✅ Valores precisos guardados en: {txt_path}")