from sklearn.datasets import load_iris
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, euclidean_distances
import matplotlib.pyplot as plt


def train_test_split(X, y, ratio=0.8):
    indices = np.random.permutation(X.shape[0])
    train_len = int(X.shape[0] * ratio)
    train_indices = indices[:train_len]
    test_indices = indices[train_len:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_train, y_train, X_test, y_test

def knn(X_train, y_train, X_test, k, dist):
    def classify_single(x):
        dists = [dist(x, i) for i in X_train]
        indices = np.argpartition(dists, k)[:k]
        return np.argmax(np.bincount(y_train[indices]))

    return [classify_single(x) for x in X_test]


def precision_recall(y_pred, y_test):
    class_precision_recall = []
    for c in np.unique(y_test):
        tp = len([i for i in range(len(y_pred)) if y_pred[i] == c and y_test[i] == c])
        fp = len([i for i in range(len(y_pred)) if y_pred[i] == c and y_test[i] != c])
        fn = len([i for i in range(len(y_test)) if y_pred[i] != c and y_test[i] == c])
        precision = tp / (tp + fp) if tp + fp > 0 else 0.
        recall = tp / (tp + fn) if tp + fn > 0 else 0.
        class_precision_recall.append((c, precision, recall))
    return class_precision_recall


def print_precision_recall(result):
    for c, precision, recall in result:
        print("class:", c, "\nprecision:", precision, "\nrecall:", recall, "\n")


def manual_euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

if __name__ == '__main__':

    iris = load_iris()
    X, y = iris.data, iris.target

    #podzial danych
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.8)

    #uzycie funkcji knn i uzycie manual_euclidean bo ten z biblioteki nie dzialal
    y_pred = knn(X_train, y_train, X_test, k=3, dist=manual_euclidean)

    results = precision_recall(y_pred, y_test)
    print_precision_recall(results)

    # wypisanie metryk: accuracy precision recall f1-score
    print("KLASYFIKACJA IRIS")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    #wizualizacja t-sne
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)  #wizualizacja całego zbioru, algorytm analizuje zbior i
    # umieszcza punkty tak aby dane w 4d lezaly blisko siebie w 2d

    #tworzenie wykresu
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', edgecolors='k')

    plt.colorbar(scatter, ticks=[0, 1, 2], format=plt.FuncFormatter(lambda val, loc: iris.target_names[int(val)]))

    plt.title("Wizualizacja t-SNE zbioru Iris (2D)")
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

