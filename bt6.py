import numpy as np
from sklearn.datasets import load_iris

# Load dataset trực tiếp từ sklearn
iris = load_iris()

# Dữ liệu đặc trưng (features)
X = iris.data  

 # Nhãn thực tế (targets)
y = iris.target 

# 1. Hàm tính khoảng cách Euclid
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# 2. Khởi tạo tâm cụm ngẫu nhiên
def initialize_centroids(X, k):
    np.random.seed(42)  # Để kết quả có thể lặp lại
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

# 3. Phân cụm dựa trên khoảng cách
def assign_clusters(X, centroids):
    clusters = []
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

# 4. Cập nhật tâm cụm
def update_centroids(X, clusters, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        points = X[clusters == i]
        centroids[i] = np.mean(points, axis=0)
    return centroids

# 5. Hàm K-means chính
def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# 6. Đánh giá F1-Score
def f1_score(clusters, y):
    from collections import Counter
    f1_scores = []
    for i in np.unique(clusters):
        true_labels = y[clusters == i]
        most_common = Counter(true_labels).most_common(1)[0][0]
        tp = sum(true_labels == most_common)
        fp = sum(true_labels != most_common)
        fn = sum(y[clusters != i] == most_common)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    return np.mean(f1_scores)

# 7. RAND Index
def rand_index(clusters, y):
    tp_fp = 0
    tp_fn = 0
    tp = 0
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            same_cluster = clusters[i] == clusters[j]
            same_class = y[i] == y[j]
            tp_fp += same_cluster
            tp_fn += same_class
            tp += same_cluster and same_class
    fp = tp_fp - tp
    fn = tp_fn - tp
    tn = len(X) * (len(X) - 1) // 2 - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

# 8. Normalized Mutual Information (NMI)
def nmi(clusters, y):
    from math import log
    total = len(y)
    mutual_info = 0
    for c in np.unique(clusters):
        for t in np.unique(y):
            intersect = np.sum((clusters == c) & (y == t))
            if intersect > 0:
                mutual_info += intersect / total * log((intersect * total) / (np.sum(clusters == c) * np.sum(y == t)))