import numpy as np
from sklearn.model_selection import train_test_split

from ._utils import demean, scale


def exponential_dataset(components, random_state=0):
    N = 1000
    rng = np.random.default_rng(random_state)
    X=rng.normal(size=(N,50))
    X, X_te = train_test_split(X, Y, test_size=0.2, random_state=random_state)
    X, X_te = scale(*demean(X, X_te))
    return X, X_te
