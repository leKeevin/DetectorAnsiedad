import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

def exploratory_analysis_pipeline(file_path, dataset_name):
    print(f"--- Iniciando EDA de: {dataset_name} ---")
    
    # 1. Carga de datos
    df = pd.read_csv(file_path)
    
    # 2. Inspección básica
    print("\n> Estructura del Dataset:")
    print(df.info())
    
    print("\n> Primeras filas:")
    print(df.head())
    
    # 3. Estadísticas descriptivas
    print("\n> Resumen Estadístico:")
    print(df.describe().T)
    
    # 4. Verificación de valores nulos
    print("\n> Valores faltantes:")
    print(df.isnull().sum())
    
    # 5. Visualización: Distribución del Target (Nivel de Ansiedad/Estrés)
    # Nota: Ajustar 'stress_level' según el nombre exacto de la columna en el CSV
    target_col = 'stress_level' if 'stress_level' in df.columns else df.columns[-1]
    
    plt.figure(figsize=(10, 5))
    sns.countplot(x=target_col, data=df, palette='viridis')
    plt.title(f'Distribución de Niveles en {dataset_name}')
    plt.show()
    
    # 6. Matriz de Correlación
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlación de Variables - {dataset_name}')
    plt.show()

# --- Ejecución ---
# Para el Student Stress Dataset
exploratory_analysis_pipeline('./lib/StressLevel/StressLevelDataset.csv', 'Student Stress')

# Nota para WESAD: Este dataset suele venir en archivos .pkl. 
# Si vas a usar la versión de Kaggle en CSV, el pipeline funcionará igual.