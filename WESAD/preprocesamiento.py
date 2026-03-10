import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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
# ... (código anterior igual) ...

# Nueva lista de columnas con los refuerzos que acabamos de agregar
columnas_señales = [
    'EDA_SCR', 'EDA_AUC', 
    'HRV_RMSSD', 'HRV_SDNN', 'HRV_MeanNN', 
    'HRV_LF', 'HRV_HF', 'HRV_LFHF'
]

print("\nAplicando StandardScaler (Z-Score) a las 8 señales...")
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

# ========================================================
# 5. GENERAR GRÁFICA DE COMPROBACIÓN (Para la Diapositiva)
# ========================================================
print("\nGenerando gráfica de los datos preprocesados...")

# Como ahora todas las variables tienen la misma escala, podemos ponerlas en una sola gráfica
plt.figure(figsize=(14, 7))

# Transformamos los datos para que Seaborn los grafique todos juntos fácilmente
df_melted = df_preprocesado.melt(
    id_vars=['Label'], 
    value_vars=columnas_señales, 
    var_name='Característica', 
    value_name='Valor Normalizado (Z-Score)'
)

# Creamos el boxplot
sns.boxplot(data=df_melted, x='Característica', y='Valor Normalizado (Z-Score)', hue='Label', palette='Set2')
plt.title('Resultado del Preprocesamiento: Todas las características en la misma escala estadística', fontsize=14)
plt.axhline(0, color='red', linestyle='--', alpha=0.5) # Línea en el 0 para mostrar que están centradas
plt.xticks(rotation=15)
plt.tight_layout()

# Guardamos la imagen para tu presentación
plt.savefig("boxplots_normalizados.png")
plt.show()

print("¡Gráfica generada y guardada como 'boxplots_normalizados.png'!")