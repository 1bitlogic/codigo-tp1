import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

# === 1. Cargar el dataset ===
df = pd.read_csv("edah_dataset_mejorado.csv")  # Cambia el nombre según tu archivo

# === 2. Seleccionar variables predictoras ===
X = df[[f"Respuesta_{i}" for i in range(1, 21)]]

# === 3. Codificar etiquetas ===
le_tipo = LabelEncoder()
y_tipo_encoded = le_tipo.fit_transform(df["Tipo_TDAH"])

le_nivel = LabelEncoder()
y_nivel_encoded = le_nivel.fit_transform(df["Nivel_TDAH"])

# === 4. Dividir en entrenamiento y prueba ===
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X, y_tipo_encoded, test_size=0.2, random_state=42)
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X, y_nivel_encoded, test_size=0.2, random_state=42)

# === 5. Calcular los pesos de las clases ===
class_weights_tipo = compute_class_weight('balanced', classes=np.unique(y_train_t), y=y_train_t)
class_weights_nivel = compute_class_weight('balanced', classes=np.unique(y_train_n), y=y_train_n)

# === 6. Definir el modelo de XGBoost ===
modelo_tipo = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
modelo_nivel = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# === 7. Parámetros para GridSearchCV ===
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 150],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# === 8. Aplicar GridSearchCV para Tipo TDAH ===
grid_search_tipo = GridSearchCV(estimator=modelo_tipo, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=2)
grid_search_tipo.fit(X_train_t, y_train_t)
mejor_modelo_tipo = grid_search_tipo.best_estimator_

# === 9. Aplicar GridSearchCV para Nivel TDAH ===
grid_search_nivel = GridSearchCV(estimator=modelo_nivel, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=2)
grid_search_nivel.fit(X_train_n, y_train_n)
mejor_modelo_nivel = grid_search_nivel.best_estimator_

# === 10. Hacer predicciones ===
y_pred_tipo = mejor_modelo_tipo.predict(X_test_t)
y_pred_nivel = mejor_modelo_nivel.predict(X_test_n)

# === 11. Reportes con etiquetas decodificadas ===
print("=== Clasificación - Tipo TDAH ===")
print(classification_report(le_tipo.inverse_transform(y_test_t), le_tipo.inverse_transform(y_pred_tipo)))

print("\n=== Clasificación - Nivel TDAH ===")
print(classification_report(le_nivel.inverse_transform(y_test_n), le_nivel.inverse_transform(y_pred_nivel)))

# === 12. Matrices de Confusión ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(le_tipo.inverse_transform(y_test_t), le_tipo.inverse_transform(y_pred_tipo)), 
            annot=True, cmap='Blues', fmt='d')
plt.title("Matriz de Confusión - Tipo TDAH")
plt.xlabel("Predicción")
plt.ylabel("Real")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(le_nivel.inverse_transform(y_test_n), le_nivel.inverse_transform(y_pred_nivel)), 
            annot=True, cmap='Greens', fmt='d')
plt.title("Matriz de Confusión - Nivel TDAH")
plt.xlabel("Predicción")
plt.ylabel("Real")

plt.tight_layout()
plt.show()
