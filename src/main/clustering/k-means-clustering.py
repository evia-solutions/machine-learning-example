from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def execute_k_means():
    # Generate synthetic data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Create KMeans instance
    kmeans = KMeans(n_clusters=4)

    # Fit and predict
    y_kmeans = kmeans.fit_predict(X)

    # Plotting the clusters
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    execute_k_means()
