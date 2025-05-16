import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# === 1. Cargar dataset ===
df = pd.read_csv("edah_dataset_mejorado.csv")

# === 2. Variables predictoras y etiquetas ===
X = df[[f"Respuesta_{i}" for i in range(1, 21)]]
y_tipo = df["Tipo_TDAH"]
y_nivel = df["Nivel_TDAH"]

# === 3. Normalización de los datos ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. Balanceo de clases con SMOTE ===
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y_tipo)

# === 5. División de datos ===
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_res, y_nivel, test_size=0.2, random_state=42)

# === 6. Búsqueda de hiperparámetros usando GridSearchCV para Random Forest ===
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# === 7. Modelo Random Forest - Tipo de TDAH ===
modelo_tipo = RandomForestClassifier(random_state=42)
grid_search_t = GridSearchCV(estimator=modelo_tipo, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_t.fit(X_train_t, y_train_t)

# Mejor modelo para Tipo de TDAH
mejor_modelo_tipo = grid_search_t.best_estimator_
y_pred_tipo = mejor_modelo_tipo.predict(X_test_t)

# === 8. Modelo Random Forest - Nivel de TDAH ===
modelo_nivel = RandomForestClassifier(random_state=42)
grid_search_n = GridSearchCV(estimator=modelo_nivel, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_n.fit(X_train_n, y_train_n)

# Mejor modelo para Nivel de TDAH
mejor_modelo_nivel = grid_search_n.best_estimator_
y_pred_nivel = mejor_modelo_nivel.predict(X_test_n)

# === 9. Evaluación del modelo - Tipo TDAH ===
print("=== Clasificación - Tipo TDAH ===")
print(classification_report(y_test_t, y_pred_tipo))

# === 10. Evaluación del modelo - Nivel TDAH ===
print("\n=== Clasificación - Nivel TDAH ===")
print(classification_report(y_test_n, y_pred_nivel))

# === 11. Matrices de Confusión ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test_t, y_pred_tipo), annot=True, cmap='Blues', fmt='d')
plt.title("Matriz de Confusión - Tipo TDAH")
plt.xlabel("Predicción")
plt.ylabel("Real")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test_n, y_pred_nivel), annot=True, cmap='Greens', fmt='d')
plt.title("Matriz de Confusión - Nivel TDAH")
plt.xlabel("Predicción")
plt.ylabel("Real")

plt.tight_layout()
plt.show()

# === 12. Curvas ROC - Tipo TDAH ===
# Seleccionar la probabilidad de la clase positiva ("Inatento") para la clase "Tipo_TDAH"
fpr_t, tpr_t, _ = roc_curve(y_test_t, mejor_modelo_tipo.predict_proba(X_test_t)[:, 0], pos_label="Inatento")
roc_auc_t = auc(fpr_t, tpr_t)

plt.figure(figsize=(6, 6))
plt.plot(fpr_t, tpr_t, color='blue', lw=2, label=f'AUC Tipo TDAH = {roc_auc_t:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Tipo TDAH')
plt.legend(loc='lower right')
plt.show()

# === 13. Curvas ROC - Nivel TDAH ===
# Seleccionar la probabilidad de la clase positiva ("Severo") para la clase "Nivel_TDAH"
fpr_n, tpr_n, _ = roc_curve(y_test_n, mejor_modelo_nivel.predict_proba(X_test_n)[:, 2], pos_label="Severo")
roc_auc_n = auc(fpr_n, tpr_n)

plt.figure(figsize=(6, 6))
plt.plot(fpr_n, tpr_n, color='green', lw=2, label=f'AUC Nivel TDAH = {roc_auc_n:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Nivel TDAH')
plt.legend(loc='lower right')
plt.show()

# === 14. Validación cruzada ===
print("\n=== Validación cruzada - Tipo TDAH ===")
scores_tipo = cross_val_score(mejor_modelo_tipo, X_scaled, y_tipo, cv=5, scoring='accuracy')
print(f"Precisión promedio en validación cruzada: {scores_tipo.mean():.2f}")

print("\n=== Validación cruzada - Nivel TDAH ===")
scores_nivel = cross_val_score(mejor_modelo_nivel, X_scaled, y_nivel, cv=5, scoring='accuracy')
print(f"Precisión promedio en validación cruzada: {scores_nivel.mean():.2f}")
