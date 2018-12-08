
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

kmeans = KMeans(n_clusters = 4).fit(reduced_data)
preds = kmeans.predict(reduced_data)

h = []
for i in range(len(preds)) :
    h.append([i, preds[i]])

df = pd.DataFrame(h, columns=['index', 'num_cluster'])
print(df)
plt.show()

# k = 4



