import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_eda_pipeline(file_path, output_subdir):
    # --- CONFIGURACIÓN DE RUTAS ---
    output_dir = f'./results/{output_subdir}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. CARGA Y EXAMEN INICIAL ---
    df = pd.read_csv(file_path)
    
    print(f"\n=== Análisis de {output_subdir} ===")
    print(f"Dimensiones: {df.shape}")
    print("\nValores Nulos:\n", df.isnull().sum().sum()) # Resumen rápido de nulos
    
    # Guardar resumen estadístico en CSV para consulta rápida
    df.describe().to_csv(f'{output_dir}summary_statistics.csv')

    # --- 2. ANÁLISIS DE LA VARIABLE OBJETIVO ---
    plt.figure(figsize=(6, 4))
    sns.countplot(x='stress_level', data=df, palette='viridis')
    plt.title("Balance de Clases (Nivel de Estrés)")
    plt.savefig(f'{output_dir}01_distribucion_target.png')
    plt.close()

    # --- 3. ANÁLISIS DE DISTRIBUCIÓN (Histogramas) ---
    df.hist(figsize=(15, 10), bins=20, color='skyblue', edgecolor='black')
    plt.suptitle("Distribución Global de Variables", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{output_dir}02_histogramas_general.png')
    plt.close()

    # --- 4. ANÁLISIS DE CORRELACIÓN ---
    # Usamos variables de interés para el mapa de calor
    cols_interes = ['anxiety_level', 'self_esteem', 'depression', 'sleep_quality', 
                    'academic_performance', 'study_load', 'stress_level']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols_interes].corr(), annot=True, cmap='RdYlGn', fmt='.2f')
    plt.title('Mapa de Correlación')
    plt.savefig(f'{output_dir}03_correlacion.png')
    plt.close()

    # --- 5. DETECCIÓN DE OUTLIERS (Boxplots) ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[cols_interes])
    plt.xticks(rotation=45)
    plt.title("Detección de Valores Atípicos en Variables Clave")
    plt.savefig(f'{output_dir}04_boxplots.png')
    plt.close()

    # --- 6. RELACIONES MULTIVARIADAS (Pairplot) ---
    # OJO: Pairplot es pesado. Solo con las variables más importantes.
    # g = sns.pairplot(df[cols_interes], hue='stress_level', corner=True, palette='coolwarm')
    # g.fig.suptitle("Relaciones Cruzadas por Nivel de Estrés", y=1.02)
    # plt.savefig(f'{output_dir}05_pairplot.png')
    # plt.close()

    print(f"✅ Pipeline completado. Resultados en: {output_dir}")

# Ejecución
run_eda_pipeline('./lib/StressLevel/StressLevelDataset.csv', 'StressLevel')