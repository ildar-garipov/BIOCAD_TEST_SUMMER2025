import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Загружаем данные
X = pd.read_csv('X_features_factorized.csv')
y = pd.read_csv('y_target.csv').values.ravel()

# Удаляем столбец index
if 'index' in X.columns:
    X = X.drop('index', axis=1)

print(f"Размерность данных: X={X.shape}, y={y.shape}")
print(f"Распределение классов: {np.bincount(y)}")
print(f"Процент положительных примеров: {y.mean()*100:.1f}%")

# Список признаков
print("\nПризнаки:")
for i, col in enumerate(X.columns, 1):
    print(f"{i}. {col}")

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Стандартизируем признаки (включая факторизованный location)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Словарь для хранения результатов
results = {}

# 1. Логистическая регрессия
print("\n1. Логистическая регрессия")
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_pred_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr),
    'recall': recall_score(y_test, y_pred_lr),
    'f1': f1_score(y_test, y_pred_lr),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_lr)
}

print(classification_report(y_test, y_pred_lr))

# 2. Random Forest (не требует стандартизации, но используем исходные данные)
print("\n2. Random Forest")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf),
    'recall': recall_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_rf)
}

print(classification_report(y_test, y_pred_rf))

# 3. Gradient Boosting
print("\n3. Gradient Boosting")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
y_pred_proba_gb = gb.predict_proba(X_test)[:, 1]

results['Gradient Boosting'] = {
    'accuracy': accuracy_score(y_test, y_pred_gb),
    'precision': precision_score(y_test, y_pred_gb),
    'recall': recall_score(y_test, y_pred_gb),
    'f1': f1_score(y_test, y_pred_gb),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_gb)
}

print(classification_report(y_test, y_pred_gb))

# 4. SVM
print("\n4. SVM")
svm = SVC(probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
y_pred_proba_svm = svm.predict_proba(X_test_scaled)[:, 1]

results['SVM'] = {
    'accuracy': accuracy_score(y_test, y_pred_svm),
    'precision': precision_score(y_test, y_pred_svm),
    'recall': recall_score(y_test, y_pred_svm),
    'f1': f1_score(y_test, y_pred_svm),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_svm)
}

print(classification_report(y_test, y_pred_svm))

# Визуализация результатов
results_df = pd.DataFrame(results).T
print("\nСравнение моделей:")
print(results_df)

# График сравнения метрик
plt.figure(figsize=(12, 8))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
x = np.arange(len(results_df.index))
width = 0.15

for i, metric in enumerate(metrics):
    plt.bar(x + i*width, results_df[metric], width, label=metric)

plt.xlabel('Модели')
plt.ylabel('Значение метрики')
plt.title('Сравнение производительности моделей')
plt.xticks(x + width*2, results_df.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison_factorized.png', dpi=300, bbox_inches='tight')
plt.show()

# Анализ влияния местоположения мутации
plt.figure(figsize=(10, 6))
location_encoder = joblib.load('location_encoder.joblib')
location_mapping = dict(zip(location_encoder.transform(location_encoder.classes_), 
                           location_encoder.classes_))

# Создаем DataFrame для анализа
location_analysis = pd.DataFrame({
    'location_code': X_test['mutation_location_encoded'],
    'location_name': X_test['mutation_location_encoded'].map(location_mapping),
    'prediction': y_pred_rf,
    'actual': y_test
})

# Вычисляем точность по местоположениям
location_accuracy = location_analysis.groupby('location_name').apply(
    lambda x: (x['prediction'] == x['actual']).mean()
).sort_values(ascending=False)

plt.bar(location_accuracy.index, location_accuracy.values)
plt.xlabel('Местоположение мутации')
plt.ylabel('Точность предсказания')
plt.title('Точность предсказания по местоположениям мутаций')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('accuracy_by_location.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nТочность по местоположениям:")
print(location_accuracy)

# Подбор гиперпараметров для лучшей модели
print("\nПодбор гиперпараметров для Random Forest...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Лучший ROC AUC: {grid_search.best_score_:.3f}")

# Оценка модели с лучшими параметрами
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
y_pred_proba_best_rf = best_rf.predict_proba(X_test)[:, 1]

print("\nРезультаты Random Forest с оптимальными параметрами:")
print(classification_report(y_test, y_pred_best_rf))
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_best_rf):.3f}")

# Сохраняем лучшую модель
joblib.dump(best_rf, 'best_random_forest_model_factorized.joblib')
joblib.dump(scaler, 'scaler_factorized.joblib')

print("\nМодель и scaler сохранены в файлы:")
print("- best_random_forest_model_factorized.joblib")
print("- scaler_factorized.joblib")
print("- location_encoder.joblib (уже сохранен ранее)")

# ROC кривые для всех моделей
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 8))
for model_name, predictions in [
    ('Logistic Regression', y_pred_proba_lr),
    ('Random Forest', y_pred_proba_rf),
    ('Gradient Boosting', y_pred_proba_gb),
    ('SVM', y_pred_proba_svm),
    ('Best RF', y_pred_proba_best_rf)
]:
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.show()