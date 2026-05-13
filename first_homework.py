import numpy as np
import sys

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def cosine_distance(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 1.0
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def kmeans(X, k, metric='euclidean', max_iters=100, e=1e-4, seed=42):
    np.random.seed(seed)
    n = X.shape[0]
    centroids = X[np.random.choice(n, k, replace=False)].copy()

    for _ in range(max_iters):
        distances = np.zeros((n, k))
        for i, point in enumerate(X):
            for j, cent in enumerate(centroids):
                if metric == 'euclidean':
                    distances[i, j] = euclidean_distance(point, cent)
                elif metric == 'manhattan':
                    distances[i, j] = manhattan_distance(point, cent)
                else:
                    distances[i, j] = cosine_distance(point, cent)

        labels = np.argmin(distances, axis=1)

        new_centroids = centroids.copy()
        for j in range(k):
            if np.any(labels == j):
                new_centroids[j] = X[labels == j].mean(axis=0)

        if np.allclose(centroids, new_centroids, atol=e):
            break
        centroids = new_centroids

    return centroids, labels

if __name__ == "__main__":
    data = sys.stdin.read().strip().split()
    if not data:
        sys.exit(0)

    k = int(data[0])
    metric = data[1]
    coords = list(map(float, data[2:]))
    X = np.array(coords).reshape(-1, 2)

    centroids, _ = kmeans(X, k, metric)
    centroids = centroids[np.argsort(centroids[:, 0])]

    for c in centroids:
        print(f"{c[0]:.3f} {c[1]:.3f}")
