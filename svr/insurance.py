from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.genfromtxt("../data/insurance.csv", delimiter=",", usecols=[0, 1, 2, 3, 4, 5], dtype="float32", skip_header=1)
y = np.genfromtxt("../data/insurance.csv", delimiter=",", usecols=6, dtype="float32", skip_header=1)

regr = make_pipeline(StandardScaler(), SVR())

# Fit the SVM model according to the given training data
regr.fit(X, y)

# Perform regression on samples in X.
print(regr.predict(X))

# Return the coefficient of determination R^2 of the prediction.
print(regr.score(X, y))
