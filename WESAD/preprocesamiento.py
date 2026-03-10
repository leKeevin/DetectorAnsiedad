import pandas as pd
from sklearn.preprocessing import StandardScaler

# ========================================================
# 1. CARGAR DATOS
# ========================================================
print("Cargando datos del Punto 1...")
df = pd.read_csv("wesad_features_punto1.csv")

# ========================================================
# 2. LIMPIEZA BASADA EN EL ANÁLISIS EXPLORATORIO
# ========================================================
# Eliminamos EDA_Tonic porque tenía una correlación de 0.96 con EDA_AUC
if 'EDA_Tonic' in df.columns:
    print("Eliminando la variable redundante: EDA_Tonic...")
    df = df.drop(columns=['EDA_Tonic'])

# Mostramos el desbalance severo que detectaste en la gráfica de barras
print("\n--- Balance de Clases ---")
print(df['Label'].value_counts())
print("Nota: Desbalance severo detectado (~12:1). Avisar al equipo para el Factor de Fisher.")

# ========================================================
# 3. NORMALIZACIÓN (Z-SCORE / STANDARD SCALER)
# ========================================================
# Definimos las columnas que son señales y necesitan normalización
# HRV_SDNN y HRV_RMSSD se quedan para que el Factor de Fisher decida cuál es mejor
columnas_señales = ['EDA_SCR', 'EDA_AUC', 'HRV_RMSSD', 'HRV_SDNN', 'HRV_LFHF']

print("\nAplicando StandardScaler (Z-Score) a las señales para manejar outliers y escalas...")
scaler = StandardScaler()

# Copiamos el dataframe para no sobreescribir el original directamente
df_preprocesado = df.copy()

# Ajustamos y transformamos solo las columnas de señales (respetando Time, Label y Subject)
df_preprocesado[columnas_señales] = scaler.fit_transform(df[columnas_señales])

# ========================================================
# 4. GUARDAR RESULTADOS
# ========================================================
nombre_archivo_salida = "wesad_preprocesado_punto3.csv"
df_preprocesado.to_csv(nombre_archivo_salida, index=False)

print(f"\n¡Preprocesamiento completado exitosamente!")
print(f"Archivo guardado como: {nombre_archivo_salida}")
print("Puedes entregar este archivo a tu compañero para los Puntos 4 y 5.")