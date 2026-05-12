import numpy as np


class HierarchicalClustering:

    def __init__(self, linkage='average'):
        if linkage not in ['single', 'complete', 'average', 'ward']:
            raise ValueError(f"Неизвестный метод связи: {linkage}")
        self.linkage = linkage
        self.linkage_matrix = None

    def euclidean_distance(self, o1, o2):
        return np.sqrt(np.sum(np.square(o1 - o2)))

    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]

        clusters = [[i] for i in range(n_samples)]
        self.linkage_matrix = np.zeros((n_samples - 1, 4))
        cluster_ids = list(range(n_samples))
        next_cluster_id = n_samples

        for step in range(n_samples - 1):
            min_dist = float('inf')
            merge_i, merge_j = 0, 1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._compute_distance(clusters[i], clusters[j], X)
                    if dist < min_dist:
                        min_dist = dist
                        merge_i = i
                        merge_j = j

            new_cluster = clusters[merge_i] + clusters[merge_j]
            new_id = next_cluster_id
            next_cluster_id += 1

            self.linkage_matrix[step, 0] = cluster_ids[merge_i]
            self.linkage_matrix[step, 1] = cluster_ids[merge_j]
            self.linkage_matrix[step, 2] = min_dist
            self.linkage_matrix[step, 3] = len(new_cluster)

            del clusters[max(merge_i, merge_j)]
            del clusters[min(merge_i, merge_j)]
            del cluster_ids[max(merge_i, merge_j)]
            del cluster_ids[min(merge_i, merge_j)]
            clusters.append(new_cluster)
            cluster_ids.append(new_id)
        return self

    def _compute_distance(self, cluster1, cluster2, X):
        if self.linkage == 'single':
            min_dist = float('inf')
            for i in cluster1:
                for j in cluster2:
                    dist = self.euclidean_distance(X[i], X[j])
                    if dist < min_dist:
                        min_dist = dist
            return min_dist

        elif self.linkage == 'complete':
            max_dist = 0
            for i in cluster1:
                for j in cluster2:
                    dist = self.euclidean_distance(X[i], X[j])
                    if dist > max_dist:
                        max_dist = dist
            return max_dist

        elif self.linkage == 'average':
            total_dist = 0
            for i in cluster1:
                for j in cluster2:
                    total_dist += self.euclidean_distance(X[i], X[j])
            return total_dist / (len(cluster1) * len(cluster2))

        elif self.linkage == 'ward':
            n1 = len(cluster1)
            n2 = len(cluster2)
            centroid1 = np.mean([X[i] for i in cluster1], axis=0)
            centroid2 = np.mean([X[i] for i in cluster2], axis=0)
            squared_dist = np.sum((centroid1 - centroid2) ** 2)
            return (n1 * n2 * squared_dist) / (n1 + n2)

    def predict(self, n_clusters=None, height=None):
        n_samples = self.linkage_matrix.shape[0] + 1

        if height is not None:
            return self.get_labels_by_height(height, n_samples)
        if n_clusters == 1:
            return np.zeros(n_samples, dtype=int)
        if n_clusters == n_samples:
            return np.arange(n_samples)

        merge_steps = n_samples - n_clusters
        idx = merge_steps - 1
        cutoff_distance = self.linkage_matrix[idx, 2] + 1e-6
        return self.get_labels_by_height(cutoff_distance, n_samples)

    def get_labels_by_height(self, height, n_samples):
        cluster_assignments = {i: [i] for i in range(n_samples)}
        next_cluster_id = n_samples

        for step in range(len(self.linkage_matrix)):
            dist = self.linkage_matrix[step, 2]
            if dist > height:
                break
            idx1 = int(self.linkage_matrix[step, 0])
            idx2 = int(self.linkage_matrix[step, 1])
            if idx1 in cluster_assignments and idx2 in cluster_assignments:
                merged = cluster_assignments[idx1] + cluster_assignments[idx2]
                del cluster_assignments[idx1]
                del cluster_assignments[idx2]
                cluster_assignments[next_cluster_id] = merged
                next_cluster_id += 1

        labels = np.zeros(n_samples, dtype=int)
        for cluster_id, points in cluster_assignments.items():
            for point in points:
                labels[point] = cluster_id

        unique_labels = np.unique(labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_mapping[label] for label in labels])
        return labels


if __name__ == "__main__":
    import sys
    data = sys.stdin.read().strip().split('\n')

    if not data or not data[0].strip():
        sys.exit(0)

    first_line = data[0].split()
    k = int(first_line[0])
    linkage_method = first_line[1]

    points = []
    for line in data[1:]:
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 2:
                points.append([float(parts[0]), float(parts[1])])

    X = np.array(points)
    model = HierarchicalClustering(linkage=linkage_method)
    model.fit(X)
    labels = model.predict(n_clusters=k)

    for label in labels:
        print(label)
