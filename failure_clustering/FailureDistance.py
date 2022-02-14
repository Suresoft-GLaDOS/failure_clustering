import numpy as np
from sklearn.utils import check_X_y
from scipy.spatial.distance import pdist, squareform
from utils.hypergraph import Hypergraph

class NoFailingTestError(Exception):
    """Raised when there is no failing test (0 not in y)"""
    pass

class NotSupportedMeasureError(Exception):
    """Raised when the failure distance measure is not supported"""
    pass

class FailureDistance:
    def __init__(self, measure):
        supported_measures = ['jaccard', 'braycurtis', 'canberra',
            'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 
            'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 
            'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 
            'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 
            'sokalsneath', 'sqeuclidean', 'yule', 'hdist']

        if measure not in supported_measures:
            raise NotSupportedMeasureError(f"Supported measure: {supported_measures}")

        self.measure = measure
    
    @staticmethod
    def validate_input(X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=bool,
            ensure_2d=True, y_numeric=True, multi_output=False)
        if np.invert(y).sum() == 0:
            raise NoFailingTestError
        return X, y

    def get_distance_matrix(self, X, y, weights=None, return_index=False):
        """
        Compute a distance matrix for the coverage matrix X and test results y
        Example:
        ```
        X = np.array([
            [0, 1, 1, 0, 1, 0], # coverage of t0 
            [1, 0, 0, 1, 0, 0], # coverage of t1
            [0, 1, 0, 1, 1, 0], # coverage of t2
            [1, 1, 0, 0, 1, 1], # coverage of t3
            [1, 1, 0, 0, 1, 1], # coverage of t4
        ], dtype=bool)

        y = np.array([
            0, # t0: FAIL
            0, # t1: FAIL
            0, # t2: FAIL
            1, # t3: PASS
            1, # t4: PASS
        ], dtype=bool)
        ```
        Return: a square-form distance matrix for failing test cases
        """
        X, y = self.validate_input(X, y)
        if self.measure == 'hdist':
            assert weights is not None

        is_failure = (y == 0)

        if self.measure == 'hdist':
            is_valid_hyperedge = X.sum(axis=0) > 0
            new_X = X[:, is_valid_hyperedge]
            HG = Hypergraph(new_X, w=weights[is_valid_hyperedge])
            matrix = HG.hdist_matrix(vertices=is_failure)
        else:
            matrix = squareform(pdist(X[is_failure, :], metric=self.measure))

        if return_index:
            # Return the original indices of failing test cases
            index = np.where(y == 0)[0]
            return matrix, index
        else:
            return matrix

if __name__ == "__main__":
    X = np.array([
        [0, 1, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 1, 0],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1],
    ], dtype=bool)
    y = np.array([0, 0, 0, 1, 1], dtype=bool)
    w = np.array([0.25, 0.40, 1.00, 1.00, 0.40, 0.00])

    fd = FailureDistance(measure='hdist')
    print("- measure: hdist")
    print(fd.get_distance_matrix(X, y, weights=w))

    fd = FailureDistance(measure='jaccard')
    print("- measure: jaccard")
    print(fd.get_distance_matrix(X, y, weights=w))