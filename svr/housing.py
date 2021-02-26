from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.loadtxt("../data/housing.data", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), dtype='float32')
y = np.loadtxt("../data/housing.data", usecols=13, dtype='float32')

regr = make_pipeline(StandardScaler(), SVR())

# Fit the SVM model according to the given training data
regr.fit(X, y)

# Perform regression on samples in X.
predictions = regr.predict(X)

# Return the coefficient of determination R^2 of the prediction.
score = regr.score(X, y)
