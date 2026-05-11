import numpy as np
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

            return

        elif self.linkage == 'complete':

            return

        elif self.linkage == 'average':

            return

        elif self.linkage == 'ward':

            return

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

        # добавьте граничные случаи

        # логика под выбор числа кластеров
        return self.get_labels_by_height(cutoff_distance, n_samples)

    def get_labels_by_height(self, height, n_samples):
        # Инициализация: каждый объект -- отдельный кластер
        cluster_assignments = {i: [i] for i in range(n_samples)}
        next_cluster_id = n_samples

        # Объединяем кластеры, пока расстояние <= height

        # Формируем метки для каждой точки

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
