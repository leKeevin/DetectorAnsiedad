import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

#RUTA DE GUARDADO DE LAS GRAFICAS   
if not os.path.exists('./results/StressLevel/'):
    os.makedirs('./results/StressLevel/')
output_dir = './results/StressLevel/'

# Cargamos el dataset de estudiantes (ajusta la ruta a tu archivo)
df= pd.read_csv('./lib/StressLevel/StressLevelDataset.csv')


# 1. ¿Cómo se relacionan las variables? (Mapa de Calor)
plt.figure(figsize=(16, 10))

print("Dimensiones del dataset:", df.shape)
print("\nPrimeras filas:")
print(df.head())

print("\nInformación general:")
print(df.info())

print("\nEstadísticas descriptivas:")
print(df.describe())


# Filtramos solo las variables más importantes para no saturar
cols_interes = ['anxiety_level', 'self_esteem', 'depression', 'sleep_quality', 
                'academic_performance', 'study_load', 'stress_level']
sns.heatmap(df[cols_interes].corr(), annot=True, cmap='RdYlGn', fmt='.2f')
plt.title('Mapa de Correlación: ¿Qué factores "viajan" juntos?')
plt.savefig(f'{output_dir}correlacion.png') # Guarda la imagen
print("Grafica 1 guardada como correlacion.png")

# 2. Relación Sueño vs Ansiedad por Nivel de Estrés
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='anxiety_level', y='sleep_quality', hue='stress_level', palette='viridis')
plt.title('Ansiedad vs Calidad de Sueño')
plt.savefig(f'{output_dir}ansiedad_vs_sueno.png')
print("Grafica 2 guardada como ansiedad_vs_sueno.png")

# 3. Violín Plot: Autoestima según el nivel de estrés
plt.figure(figsize=(10, 6))
sns.violinplot(x='stress_level', y='self_esteem', data=df, palette='Pastel1')
plt.title('Distribución de la Autoestima por Nivel de Estrés')
plt.savefig(f'{output_dir}autoestima_stress.png')
print("Grafica 3 guardada como autoestima_stress.png")

df.hist(figsize=(18, 12), bins=20)
plt.suptitle("Distribución de Variables del Dataset", fontsize=16)
plt.savefig(f'{output_dir}histogramas.png')

plt.figure(figsize=(14, 8))
sns.boxplot(data=df[cols_interes])
plt.xticks(rotation=45)
plt.title("Detección de Valores Atípicos")
plt.savefig(f'{output_dir}boxplots.png')

sns.pairplot(df[cols_interes], hue='stress_level', corner=True)
plt.savefig(f'{output_dir}pairplot.png')

mean_by_stress = df.groupby('stress_level')[cols_interes[:-1]].mean()

mean_by_stress.plot(kind='bar', figsize=(12, 6))
plt.title("Promedio de Variables por Nivel de Estrés")
plt.xticks(rotation=0)
plt.savefig(f'{output_dir}promedios_stress.png')

sns.countplot(x='stress_level', data=df)
plt.title("Distribución de Niveles de Estrés")
plt.savefig(f'{output_dir}conteo_stress.png')