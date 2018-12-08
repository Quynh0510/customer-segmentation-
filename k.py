
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display 
import visuals as vs



data = pd.read_csv("Wholesale_customers_data.csv")
data.drop(['Region', 'Channel'], axis = 1, inplace = True)


log_data = data.applymap(np.log)  # ln(data)


from sklearn.decomposition import PCA
# Apply PCA by fitting the good data with the same number of dimensions as features
num_features = log_data.shape[1]
pca = PCA(n_components = num_features, random_state = 0)
pca = pca.fit(log_data)


# Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components = 2, random_state = 0)
pca = pca.fit(log_data)

# Transform the good data using the PCA fit above
reduced_data = pca.transform(log_data)


# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Use silhouette score to find optimal number of clusters to segment the data
num_clusters = np.arange(3, 8)
kmeans_results = {}
for size in num_clusters:
    kmeans = KMeans(n_clusters = size).fit(reduced_data)
    preds = kmeans.predict(reduced_data)
    kmeans_results[size] = silhouette_score(reduced_data, preds)

h = []
for i in kmeans_results.keys() :
    h.append([i, kmeans_results[i]])

df = pd.DataFrame(h, columns=['k', 'gt'])

plt.scatter(x=df['k'], y=df['gt'], marker='s')
plt.show()

# k = 4



