import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./lib/StressLevel/StressLevelDataset.csv')


# 1. Discretizamos cada columna
# pd.cut divide el rango de los datos en partes iguales (bins=3)
df['anxiety_level'] = pd.cut(df['anxiety_level'], bins=3, labels=False)
df['self_esteem'] = pd.cut(df['self_esteem'], bins=3, labels=False)
df['depression'] = pd.cut(df['depression'], bins=3, labels=False)

# Guardamos en la carpeta de resultados para que el siguiente script lo use
df.to_csv('./results/StressLevel/StressLevel_Discretized.csv', index=False)
print("✅ Dataset discretizado guardado en: ./results/StressLevel/StressLevel_Discretized.csv")

# --- NUEVO PASO: Mapear los niveles de estrés a nombres descriptivos ---
stress_map = {0: 'Estrés Bajo', 1: 'Estrés Medio', 2: 'Estrés Alto'}
df['stress_desc'] = df['stress_level'].map(stress_map)

# 3. Verificamos el resultado
print(df[['anxiety_level', 'self_esteem']].head())

plt.figure(figsize=(10, 6))
sns.countplot(x='anxiety_level', hue='stress_desc', data=df, hue_order=['Estrés Bajo', 'Estrés Medio', 'Estrés Alto'], palette='magma')
plt.title('Niveles de Ansiedad Percibida vs Estrés Real')
plt.xlabel('Categoría de Ansiedad')
plt.ylabel('Cantidad de Estudiantes')
plt.savefig('./results/StressLevel/ansiedad_categorica.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(x='self_esteem', hue='stress_desc', data=df, hue_order=['Estrés Bajo', 'Estrés Medio', 'Estrés Alto'], palette='magma')
plt.title('Niveles de Autoestima vs Estrés Real')
plt.xlabel('Categoría de Autoestima')
plt.ylabel('Cantidad de Estudiantes')
plt.savefig('./results/StressLevel/autoestima_categorica.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(x='depression', hue='stress_desc', data=df, hue_order=['Estrés Bajo', 'Estrés Medio', 'Estrés Alto'], palette='magma')
plt.title('Niveles de Depresión vs Estrés Real')
plt.xlabel('Categoría de Depresión')
plt.ylabel('Cantidad de Estudiantes')
plt.savefig('./results/StressLevel/depresion_categorica.png')
plt.close()

