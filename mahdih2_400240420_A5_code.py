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

def calculate_mse(original_image, reconstructed_image):
    return ((original_image - reconstructed_image) ** 2).mean()

def initialize_centroids(data, K):
    n_samples = data.shape[0]
    
    # Choose the first centroid randomly
    centroids = [data[np.random.choice(n_samples)]]

    # Choose remaining centroids
    for i in range(1, K):
        distances = np.array([np.linalg.norm(data - centroid, axis=1) for centroid in centroids])
        max_distance_index = np.argmax(np.min(distances, axis=0))
        centroids.append(data[max_distance_index])

    return np.array(centroids)


def k_means(data, K, max_iters,init_strat):

    #init_start  0 - random , 1 - large space between centriods
    if init_strat == 1:
        centroids = initialize_centroids(data, K)
    else:
        centroids = data[np.random.choice(data.shape[0], K, replace=False)]
    pixel_to_cluster_map = np.zeros(data.shape[0], dtype=int)  
    prev_pixel_to_cluster_map = np.zeros(data.shape[0], dtype=int) 
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
        #print('Old Mappings:')
        #print(prev_pixel_to_cluster_map)
        #print(prev_pixel_to_cluster_map.shape)
        #print('New Mappings:')
        #print(pixel_to_cluster_map)
        #print(pixel_to_cluster_map.shape)

        if diff < 5000 or np.array_equal(pixel_to_cluster_map, prev_pixel_to_cluster_map) or np.allclose(centroids, new_centroids, atol=1) :
            break
        else:

            #print(diff)

            #print(centroids)
            prev_pixel_to_cluster_map = pixel_to_cluster_map.copy()

            centroids = new_centroids
            previous_cost = cost

    return pixel_to_cluster_map, centroids, i

image = mpimg.imread('messi.jpg')
print(image.shape)
rgb_pixels = image.reshape(-1, 3)  
print(rgb_pixels.shape)


K_val = [2,3,10,20,40]
mse_errors = []
init_method =0
for i in range(3):
    if(i==2):
        init_method = 1
    mse_errors = []
    iterations = []
    for k in K_val:
        clusters, centroids,iteration = k_means(rgb_pixels, k,100,init_method)
        compressed_image = centroids[clusters].reshape(image.shape)
        compressed_image = Image.fromarray(compressed_image.astype('uint8'))
        if(init_method == 0):
            if(i == 0):
                file_name = 'random1_segmented_Image1_K=' + str(k) + '.jpg'
            else:
                file_name = 'random2_segmented_Image1_K=' + str(k) + '.jpg'

        else:
            file_name = 'large_space_segmented_Image1_K=' + str(k) + '.jpg'

        mse = calculate_mse(image, compressed_image)
        compressed_image.save(file_name)
        mse_errors.append(mse)
        iterations.append(iteration)
    print(mse_errors)
    print(iterations)

