import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
# Modelos
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import CategoricalNB
from chefboost import Chefboost as chef

import matplotlib.pyplot as plt




# ==========================
# CARGAR DATASET
# ==========================
df = pd.read_csv("./../results/StressLevel/StressLevel_Discretized.csv")

# ==========================
# VARIABLES
# ==========================
X = df.drop("stress_level", axis=1)
y = df["stress_level"]

# ==========================
# DIVIDIR DATOS
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ==========================
# MODELO ID3
# ==========================
id3 = DecisionTreeClassifier(
    criterion="entropy",   # <- ID3
    random_state=42
)

id3.fit(X_train, y_train)

pred_id3 = id3.predict(X_test)

#Muestra el árbol de decisión
plt.figure(figsize=(40,20))
plot_tree(id3, filled=True, feature_names=X.columns, fontsize=8)
# plt.show()
plt.savefig('./../results/StressLevel/arbol_decision_id3.svg', dpi=300, bbox_inches='tight')

print("===== RESULTADOS ID3 =====")
print("Accuracy:", accuracy_score(y_test, pred_id3))
print(classification_report(y_test, pred_id3))
print(confusion_matrix(y_test, pred_id3))


# ==========================
# MODELO NAIVE BAYES
# ==========================
nb = CategoricalNB()

nb.fit(X_train, y_train)

pred_nb = nb.predict(X_test)

print("\n===== RESULTADOS NAIVE BAYES =====")
print("Accuracy:", accuracy_score(y_test, pred_nb))
print(classification_report(y_test, pred_nb))
print(confusion_matrix(y_test, pred_nb))

