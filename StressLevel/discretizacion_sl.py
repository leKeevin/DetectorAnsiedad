import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./lib/StressLevel/StressLevelDataset.csv')

# 1. Definimos las etiquetas que queremos usar
labels_3 = ['Bajo', 'Medio', 'Alto']

# 2. Discretizamos cada columna
# pd.cut divide el rango de los datos en partes iguales (bins=3)
df['anxiety_category'] = pd.cut(df['anxiety_level'], bins=3, labels=labels_3)
df['self_esteem_category'] = pd.cut(df['self_esteem'], bins=3, labels=labels_3)
df['depression_category'] = pd.cut(df['depression'], bins=3, labels=labels_3)

# --- NUEVO PASO: Mapear los niveles de estrés a nombres descriptivos ---
stress_map = {0: 'Estrés Bajo', 1: 'Estrés Medio', 2: 'Estrés Alto'}
df['stress_desc'] = df['stress_level'].map(stress_map)

# 3. Verificamos el resultado
print(df[['anxiety_level', 'anxiety_category', 'self_esteem', 'self_esteem_category']].head())

plt.figure(figsize=(10, 6))
sns.countplot(x='anxiety_category', hue='stress_desc', data=df, hue_order=['Estrés Bajo', 'Estrés Medio', 'Estrés Alto'], palette='magma')
plt.title('Niveles de Ansiedad Percibida vs Estrés Real')
plt.xlabel('Categoría de Ansiedad')
plt.ylabel('Cantidad de Estudiantes')
plt.savefig('./results/StressLevel/ansiedad_categorica.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(x='self_esteem_category', hue='stress_desc', data=df, hue_order=['Estrés Bajo', 'Estrés Medio', 'Estrés Alto'], palette='magma')
plt.title('Niveles de Autoestima vs Estrés Real')
plt.xlabel('Categoría de Autoestima')
plt.ylabel('Cantidad de Estudiantes')
plt.savefig('./results/StressLevel/autoestima_categorica.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(x='depression_category', hue='stress_desc', data=df, hue_order=['Estrés Bajo', 'Estrés Medio', 'Estrés Alto'], palette='magma')
plt.title('Niveles de Depresión vs Estrés Real')
plt.xlabel('Categoría de Depresión')
plt.ylabel('Cantidad de Estudiantes')
plt.savefig('./results/StressLevel/depresion_categorica.png')
plt.close()
