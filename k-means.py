import numpy as np
import matplotlib.pyplot as plt


def plot_k_means(data, centroids, assignments, k, iteration):
    """Plot the data points and centroids for visualization."""
    plt.clf()  # Clear the current figure
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    for i in range(k):
        # Plot data points assigned to each cluster
        cluster_data = data[assignments == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
        
        # Plot centroids
        plt.scatter(centroids[i, 0], centroids[i, 1], s=200, c=colors[i], edgecolors='k', marker='*', label=f'Centroid {i+1}')
    
    plt.title(f'K-means Clustering: Iteration {iteration}')
    plt.legend()
    plt.pause(0.5)  # Pause to display the plot
    
    
def initialize_centroids_randomly(data, k):
    indices = np.random.permutation(data.shape[0])
    centroids = data[indices[:k]]
    return centroids

def assign_datapoints(data, centroids):
    distance = np.sqrt(((data - centroids[:,np.newaxis])**2).sum(axis = 2))
    return np.argmin(distance, axis=0)

def update_centroids(data, assignments, k):
    centroids = np.array([data[assignments==i].mean(axis=0) for i in range(k)])
    return centroids

def k_means(data, k=3,max_iter=300):
    
    # initialize centroids
    centroids = initialize_centroids_randomly(data, k)
    
    for i in range(max_iter):
        
        old_centroids = centroids
        # assign data point to centroids
        
        assignments = assign_datapoints(data, centroids)
        # update centroids
        centroids = update_centroids(data, assignments, k)
        plot_k_means(data, centroids, assignments, k, i + 1)  # Plot at each iteration
        if np.any(centroids==old_centroids):
            print('k_means done')
            break
        
    return centroids, assignments