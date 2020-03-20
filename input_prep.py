# Function to prepare input for 1D CNN

def input_prep(X, n_features):
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    return X
