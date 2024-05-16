import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances_argmin_min
import scipy.io
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# Function to calculate angular error
def calculate_angular_error(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180.0 / np.pi

# Load the dataset
mat1 = scipy.io.loadmat('ColorCheckerData.mat')
data = mat1['REC_groundtruth']


#mat1 = scipy.io.loadmat('real_illum_568..mat')
#data1 = mat1['real_rgb'] 
#data = data1 / 4095.0

mean_errors = []
median_errors = []
worst_25_percent_errors = []

# Loop over different numbers of clusters
for k in range(1, 101):
    
    # Fit Spectral Clustering model
    clustering = SpectralClustering(n_clusters=k, random_state=8).fit(data)

    # Get labels
    labels = clustering.labels_

    # Calculate mean of each cluster
    centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

    # Find the closest centroid for each data point
    closest, _ = pairwise_distances_argmin_min(data, centroids)

    # Calculate angular error for each data point
    errors = [calculate_angular_error(data[i], centroids[closest[i]]) for i in range(data.shape[0])]

    # Calculate mean error
    mean_error = np.mean(errors)
    mean_errors.append(mean_error)
    
    # Calculate median error
    median_error = np.median(errors)
    median_errors.append(median_error)
    
    errors.sort(reverse=True)
    worst_25_percent = errors[:int(len(errors) * 0.25)]
    worst_25_percent_mean_error = np.mean(worst_25_percent)
    worst_25_percent_errors.append(worst_25_percent_mean_error)


plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), mean_errors, label='Mean Error')
plt.plot(range(1, 101), median_errors, label='Median Error')
plt.plot(range(1, 101), worst_25_percent_errors, label='Worst 25% Mean Error') 
plt.style.use('ggplot')
plt.title('REC Spectral Clustering')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Angular Error')
plt.grid(True)
plt.legend()
#plt.show()



mean_errors = []
median_errors = []
worst_25_percent_errors = []


for k in [10, 20, 40, 60]:
    # Fit Spectral Clustering model
    clustering = SpectralClustering(n_clusters=k, random_state=8).fit(data)

    # Get labels
    labels = clustering.labels_

    # Calculate mean of each cluster
    centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])


    # Find the closest centroid for each data point
    closest, _ = pairwise_distances_argmin_min(data, centroids)

    # Calculate angular error for each data point
    errors = [calculate_angular_error(data[i], centroids[closest[i]]) for i in range(data.shape[0])]

    # Calculate mean error
    mean_error = np.mean(errors)

    # Calculate median error
    median_error = np.median(errors)

    # Calculate worst 25% mean error
    errors.sort(reverse=True)
    worst_25_percent = errors[:int(len(errors) * 0.25)]
    worst_25_percent_mean_error = np.mean(worst_25_percent)

    # Print the results
    print(f'K: {k}')
    print(f'Mean Error: {mean_error}')
    print(f'Median Error: {median_error}')
    print(f'Worst 25% Mean Error: {worst_25_percent_mean_error}')
    print('---------------------------------')