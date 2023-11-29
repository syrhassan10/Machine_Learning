import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def calculate_cost(data, centroids, clusters):
    cost = 0
    for idx, point in enumerate(data):
        centroid = centroids[int(clusters[idx])]
        cost += np.sum((point - centroid) ** 2)
    return cost


def k_means(data, K, max_iters):
    # Initialize centroids
    centroids = data[np.random.choice(data.shape[0], K, replace=False)]
    pixel_to_cluster_map = np.zeros(data.shape[0], dtype=int)  # Specify dtype as int
    prev_pixel_to_cluster_map = np.zeros(data.shape[0], dtype=int)  # Specify dtype as int
    previous_cost = float('inf')
    for i in range(max_iters):
        for idx, point in enumerate(data):
            distances = []

            for centroid in centroids:
                distance = euclidean_distance(point, centroid)
                distances.append(distance)

            closest_centroid_index = np.argmin(distances)

            pixel_to_cluster_map[idx] = closest_centroid_index  # nearest cluster centroid for each data point

        # Update step
        new_centroids = []
        for k in range(K):
            cluster_points = data[pixel_to_cluster_map == k]
            if len(cluster_points) == 0:
                new_centroid = centroids[k]
            else:
                new_centroid = cluster_points.mean(axis=0)  # average of R G B in this cluster returns (1, 3) numpy list
            new_centroids.append(new_centroid)
        new_centroids = np.array(new_centroids)

        # Calculate cost for this iteration
        cost = calculate_cost(data, new_centroids, pixel_to_cluster_map)
        diff = np.abs(previous_cost - cost)
        print('Old Mappings:')
        print(prev_pixel_to_cluster_map)
        print(prev_pixel_to_cluster_map.shape)
        print('New Mappings:')
        print(pixel_to_cluster_map)
        print(pixel_to_cluster_map.shape)

        if diff < 5000 or np.array_equal(pixel_to_cluster_map, prev_pixel_to_cluster_map) :
            break
        else:

            print(diff)

            #print(centroids)
            prev_pixel_to_cluster_map = pixel_to_cluster_map.copy()

            centroids = new_centroids
            previous_cost = cost

    return pixel_to_cluster_map, centroids

image = mpimg.imread('messi.jpg')
rgb_pixels = image.reshape(-1, 3)  
print(rgb_pixels.shape)


K = 1  # Number of clusters
clusters, centroids = k_means(rgb_pixels, K,100)
compressed_image = centroids[clusters].reshape(image.shape)
compressed_image = Image.fromarray(compressed_image.astype('uint8'))
compressed_image.save('compressed_image.jpg')

# Read the image

#original_height = 2451
#original_width = 3677
#reconstructed_image = processed_pixels.reshape((original_height, original_width, 3))

#plt.show()
#plt.imshow(image)