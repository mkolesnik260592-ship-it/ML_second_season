import numpy as np

# Здесь вставьте полный код вашего класса HierarchicalClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram



class HierarchicalClustering:

    def __init__(self, linkage='average'):
        """
        Параметры:
        ----------
        linkage : str
            Метод связи между кластерами:
            - 'single': минимальное расстояние
            - 'complete': максимальное расстояние
            - 'average': среднее расстояние
            - 'ward': минимизация дисперсии
        """
        if linkage not in ['single', 'complete', 'average', 'ward']:
            raise ValueError(f"Неизвестный метод связи: {linkage}")
        self.linkage = linkage
        self.linkage_matrix = None  # Матрица связей для дендрограммы

    def euclidean_distance(self, o1, o2):
        euc_distance =  np.sqrt(np.sum(np.square(o1 - o2)))
        return euc_distance

    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]

        # Инициализация: каждый объект -- отдельный кластер
        # Кластеры представлены как списки индексов точек
        clusters = [[i] for i in range(n_samples)]


        # Матрица связей: (n_samples - 1) строк по 4 столбца
        # [индекс_кластера1, индекс_кластера2, расстояние, количество_точек]
        self.linkage_matrix = np.zeros((n_samples - 1, 4))

        # Вспомогательная структура для отслеживания "активных" кластеров
        # Каждый кластер имеет уникальный ID: 0..n_samples-1 для исходных точек,
        # n_samples..2*n_samples-2 для новых кластеров
        cluster_ids = list(range(n_samples))
        next_cluster_id = n_samples

        # Итеративное объединение кластеров
        for step in range(n_samples - 1):
            min_dist = float('inf')
            merge_i, merge_j = 0, 1

            # Поиск пары кластеров с минимальным расстоянием
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._compute_distance(clusters[i], clusters[j], X)
                    if dist < min_dist:
                        min_dist = dist
                        merge_i = i
                        merge_j = j

            # Объединение кластеров
            new_cluster = clusters[merge_i] + clusters[merge_j]
            new_id = next_cluster_id
            next_cluster_id += 1

            # Сохранение информации в матрицу связей
            # Индексы кластеров в формате удобном для scipy:
            # - исходные точки: 0..n_samples-1
            # - новые кластеры: n_samples + step
            self.linkage_matrix[step, 0] = cluster_ids[merge_i]
            self.linkage_matrix[step, 1] = cluster_ids[merge_j]
            self.linkage_matrix[step, 2] = min_dist
            self.linkage_matrix[step, 3] = len(new_cluster)

            # Обновление списка кластеров и их ID
            # Удаляем в обратном порядке, чтобы не нарушить индексацию
            # Добавляем новые

            del clusters[max(merge_i, merge_j)]
            del clusters[min(merge_i, merge_j)]
            del cluster_ids[max(merge_i, merge_j)]
            del cluster_ids[min(merge_i, merge_j)]
            clusters.append(new_cluster)
            cluster_ids.append(new_id)
        return self

    def _compute_distance(self, cluster1, cluster2, X):
        """Вычисление расстояния между двумя кластерами в зависимости от метода связи"""
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
        """
        Предсказание меток кластеров

        Параметры:
        ----------
        n_clusters : int, optional
            Желаемое количество кластеров (надо выбрать с height)
        height : float, optional
            Высота среза дендрограммы (надо выбрать с n_clusters)

        Возвращает:
        -----------
        labels : np.ndarray
            Метки кластеров для каждой точки
        """
        n_samples = self.linkage_matrix.shape[0] + 1

        if height is not None:
            return self.get_labels_by_height(height, n_samples)
        if n_clusters == 1:
            return np.zeros(n_samples, dtype=int)
        if n_clusters == n_samples:
            return np.arange(n_samples)
        # добавьте граничные случаи
        merge_steps = n_samples - n_clusters
        idx = merge_steps - 1
        cutoff_distance = self.linkage_matrix[idx, 2] + 1e-6
        # логика под выбор числа кластеров
        return self.get_labels_by_height(cutoff_distance, n_samples)

    def get_labels_by_height(self, height, n_samples):
        cluster_assignments = {i: [i] for i in range(n_samples)}
        next_cluster_id = n_samples

        print(f"Height = {height}")
        for step in range(len(self.linkage_matrix)):
            dist = self.linkage_matrix[step, 2]
            if dist > height:
                print(f"  Stop at step {step}, dist={dist} > {height}")
                break
            idx1 = int(self.linkage_matrix[step, 0])
            idx2 = int(self.linkage_matrix[step, 1])
            if idx1 in cluster_assignments and idx2 in cluster_assignments:
                merged = cluster_assignments[idx1] + cluster_assignments[idx2]
                del cluster_assignments[idx1]
                del cluster_assignments[idx2]
                cluster_assignments[next_cluster_id] = merged
                next_cluster_id += 1
                print(f"  Step {step}: merged {idx1} and {idx2}, now {len(cluster_assignments)} clusters")

        # ВЫНЕСИТЕ ЭТИ СТРОКИ ИЗ ЦИКЛА (уменьшите отступ)
        print(f"Final number of clusters: {len(cluster_assignments)}")

        # Формируем метки для каждой точки
        labels = np.zeros(n_samples, dtype=int)
        for cluster_id, points in cluster_assignments.items():
            for point in points:
                labels[point] = cluster_id

        unique_labels = np.unique(labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_mapping[label] for label in labels])
        print(self.linkage_matrix[:5])
        return labels

    def plot_dendrogram(self, labels=None, figsize=(10, 6), **kwargs):
        """
        Построение дендрограммы

        Параметры:
        ----------
        labels : list, optional
            Подписи для листьев дендрограммы
        figsize : tuple
            Размер фигуры matplotlib
        **kwargs : dict
            Дополнительные параметры для scipy.cluster.hierarchy.dendrogram
        """
        if self.linkage_matrix is None:
            raise ValueError("Сначала вызовите метод fit()")

        plt.figure(figsize=figsize)
        dendrogram(self.linkage_matrix, labels=labels, **kwargs)
        plt.title('Дендрограмма иерархической кластеризации')
        plt.xlabel('Индекс объекта')
        plt.ylabel('Расстояние')
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    # Вставьте тестовые данные прямо в код для проверки
    test_input = """2 average
-0.9299848075453587 9.78172085735123
-4.234115455565783 8.45199859895735
-2.978672008987702 9.556846171784287
4.524230553839861 1.6720659883514433
-2.9726153158652124 8.548556374628065
4.848742431232857 0.013499560060956428
4.038172223998704 3.82544786844967
-3.5220287433871738 9.328533460793595
4.095496111702919 2.0840922736505982
-2.743350997776086 8.78014917124914
5.378345416223512 2.1445379651307026
3.488885258805799 2.348867702286404
-1.0435488541311961 8.788509827711787
-2.4416694183648264 7.589537941984865
-3.417221698573961 7.60198242686303
-2.267235351486716 7.101005883540523
4.626381611490167 0.9154587549848316
3.3116927873296707 2.1700309198098555
4.039240146309297 1.6814759341474552
5.4624237483312905 0.7523260339697098"""

    lines = test_input.strip().split('\n')
    first_line = lines[0].split()
    k = int(first_line[0])
    linkage_method = first_line[1]

    points = []
    for line in lines[1:]:
        x, y = map(float, line.split())
        points.append([x, y])

    X = np.array(points)

    model = HierarchicalClustering(linkage=linkage_method)
    model.fit(X)
    labels = model.predict(n_clusters=k)

    for label in labels:
        print(label)
