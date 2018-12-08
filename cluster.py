
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display 
import visuals as vs



data = pd.read_csv("Wholesale_customers_data.csv")
data.drop(['Region', 'Channel'], axis = 1, inplace = True)

# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.5, figsize = (20, 15), diagonal = 'hist') # kde là vẽ theo duong, hist la ve theo bieu do cot.

# Scale the data using the natural logarithm
log_data = data.applymap(np.log)  # ln(data)

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.5, figsize = (20, 10), diagonal = 'kde')

feature_outliers  = []

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5*(Q3 - Q1)
    # Display the outliers
    # print ("Data points considered outliers for the feature '{}':".format(feature))
    # display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    feature_outliers.append(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))].index)


# Flatten list of outliers from each iteration of the loop above
outliers_flattened = [index for feature in feature_outliers for index in feature]

# Count the number of features for which a given observation is considered an outlier
from collections import Counter
outlier_count = Counter(outliers_flattened)
# Drop observations that are an outlier in 3 or more dimensions of the feature-space
outliers = [observation for observation in outlier_count.elements() if outlier_count[observation] >= 2]

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

# Examine whether or not removing outliers affected the mean or variance in the data
display(log_data.describe())
display(good_data.describe())


# The following observations are considered outliers for more than one feature based on Tukey's method of outlier detection

# display(set([observation for observation in outlier_count.elements() if outlier_count[observation] >= 2]))

# print ("{} observations were removed from the dataset.".format(
#     len(set([observation for observation in outlier_count.elements() if outlier_count[observation] >= 2]))))

from sklearn.decomposition import PCA
# Apply PCA by fitting the good data with the same number of dimensions as features
num_features = good_data.shape[1]
pca = PCA(n_components = num_features, random_state = 0)
pca = pca.fit(good_data)


# Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components = 2, random_state = 0)
pca = pca.fit(good_data)

# Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)


# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

# Create a biplot
vs.biplot(good_data, reduced_data, pca)
plt.show()





