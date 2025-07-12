import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import cross_validate, train_test_split

# === 1. Carregar dados do MovieLens 100k ===
data = Dataset.load_builtin('ml-100k')

# === 2. Dividir treino e teste ===
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# === 3. Algoritmo baseado em fatores latentes (SVD) ===
model_svd = SVD()
model_svd.fit(trainset)
predictions_svd = model_svd.test(testset)

# Avaliação SVD
print("SVD - Métricas de Avaliação:")
accuracy.rmse(predictions_svd)
accuracy.mae(predictions_svd)

# === 4. Algoritmo baseado em filtragem colaborativa com KNN ===
model_knn = KNNBasic(sim_options={'user_based': False})  # Item-based
model_knn.fit(trainset)
predictions_knn = model_knn.test(testset)

# Avaliação KNN
print("\nKNN - Métricas de Avaliação:")
accuracy.rmse(predictions_knn)
accuracy.mae(predictions_knn)

# === 5. Comparação por cross-validation ===
print("\nComparação com Cross-Validation:")
results_svd = cross_validate(SVD(), data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
results_knn = cross_validate(KNNBasic(sim_options={'user_based': False}), data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# === 6. Visualizar comparações ===
def plot_comparison(metric):
    svd_scores = results_svd[metric]
    knn_scores = results_knn[metric]
    plt.figure(figsize=(8, 5))
    plt.plot(svd_scores, label="SVD", marker='o')
    plt.plot(knn_scores, label="KNN (Item-based)", marker='s')
    plt.ylabel(metric)
    plt.title(f"Comparison of {metric} by Fold")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_comparison('test_rmse')
plot_comparison('test_mae')
