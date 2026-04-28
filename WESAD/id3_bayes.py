import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
import matplotlib.pyplot as plt

# ========================================================
# 1. CARGA DE DATOS PREPROCESADOS
# ========================================================
FILE_NAME = "wesad_preprocesado_punto3.csv"
print(f"Cargando {FILE_NAME}...")
df = pd.read_csv(FILE_NAME)

# Definimos nuestras X (todas las características) y nuestra y (etiqueta)
X = df[['EDA_SCR', 'EDA_AUC', 'HRV_RMSSD', 'HRV_SDNN', 'HRV_MeanNN', 'HRV_LF', 'HRV_HF', 'HRV_LFHF']]
y = df['Label']

# ========================================================
# 2. DIVISIÓN DE DATOS (70% Entrenamiento / 30% Prueba)
# ========================================================
# Usamos stratify=y para mantener la proporción del desbalance 12:1 en ambos sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"Datos de entrenamiento: {len(X_train)} | Datos de prueba: {len(X_test)}")

# ========================================================
# 3. DEFINICIÓN DE MODELOS
# ========================================================
modelos = {
    "Bayes Ingenuo (Gaussiano)": GaussianNB(),
    "Árbol de Decisión (ID3)": DecisionTreeClassifier(criterion='entropy', random_state=42)
}

resultados = []

# ========================================================
# 4. ENTRENAMIENTO Y EVALUACIÓN
# ========================================================
plt.figure(figsize=(12, 5))

for i, (nombre, clf) in enumerate(modelos.items()):
    # Entrenamiento
    clf.fit(X_train, y_train)
    
    # Predicción
    y_pred = clf.predict(X_test)
    
    # Cálculo de Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nMatriz de confusión ({nombre}):")
    print(cm)
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred, digits=4))
    
    resultados.append({
        "Modelo": nombre,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })
    
    # Graficar Matriz de Confusión
    plt.subplot(1, 2, i+1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz: {nombre}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')

plt.tight_layout()
plt.savefig("matrices_confusion_punto3.png")
plt.show()

# ========================================================
# 5. REPORTE FINAL DE RESULTADOS
# ========================================================
df_resultados = pd.DataFrame(resultados)
print("\n" + "="*50)
print("RESULTADOS FINALES - PUNTO 3 (WESAD TODAS LAS CARACTERÍSTICAS)")
print("="*50)
print(df_resultados.to_string(index=False))
print("="*50)

# ========================================================
# 6. VISUALIZACIÓN DEL ÁRBOL DE DECISIÓN
# ========================================================
print("\nGenerando gráfico del Árbol de Decisión...")

# Extraemos el modelo del árbol que ya fue entrenado en el bucle anterior
arbol_clf = modelos["Árbol de Decisión (ID3)"]

# Creamos una figura grande para que los textos no se encimen
plt.figure(figsize=(20, 10))

# Graficamos el árbol. 
# max_depth=3 limita la vista a los primeros 3 niveles para que sea legible.
plot_tree(arbol_clf, 
          feature_names=X.columns, 
          class_names=['No-Estrés', 'Estrés'],
          filled=True, 
          rounded=True,
          max_depth=3, 
          fontsize=10)

plt.title("Estructura Interna del Árbol de Decisión (ID3) - WESAD", fontsize=16)
plt.tight_layout()

# Guardamos en ultra alta resolución por si la ocupas en la presentación
plt.savefig("arbol_decision_estructura.png", dpi=300)
plt.show()

print("¡Gráfico guardado como 'arbol_decision_estructura.png'!")