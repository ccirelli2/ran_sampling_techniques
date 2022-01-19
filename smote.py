# -*- coding: utf-8 -*-
"""

Ref : https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
Created on Wed Jan 13 09:25:11 2021
@author: chris.cirelli
"""

# Import Libraries
import imblearn
from sklearn.datasets import make_classification
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Create Imbalanced Dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.99],
                           flip_y=0, random_state=1)

# Get Count y
cnt = Counter(y)
print(cnt)


# Plot Classes
for label, _ in cnt.items():
	row_ix = np.where(y == label)[0]
	plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
plt.legend()
plt.show()



# Over Sample Minority Class & Plot New Relationship
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

cnt = Counter(y)
print(cnt)

# Plot Classes
for label, _ in cnt.items():
	row_ix = np.where(y == label)[0]
	plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
plt.legend()
plt.show()

# Undersampling. 



