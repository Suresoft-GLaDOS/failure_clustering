import numpy as np
import warnings
from sklearn.utils import check_X_y
from scipy.spatial.distance import pdist, squareform
from .utils.hypergraph import Hypergraph

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
        Compute a pairwise failure distance matrix.
        
        Returns the square-form distance matrix. There are one optional
        output.

        * the indices of the array `y` that correspond to failing test cases
        
        Parameters
        ----------
        X : array_like
            2-D coverage matrix with shape (# test cases, # components)
        y : array_like
            1-D test result vector with shape (# test cases, )
        weights: array_like
            1-D weight vector of components with shape (# components, )
        return_index : bool, optional
            If True, also return the indices of `y` that correspond to
            failing test cases

        Returns
        -------
        matrix : ndarray
        a square-form distance matrix for failing test cases
        failing_indices : ndarray, optional
        The indices of the occurrences of the failing test cases in the
        input array `y`. Only provided if `return_index` is True
        """
        X = np.asanyarray(X)
        y = np.asanyarray(y)

        X, y = self.validate_input(X, y)
        if self.measure == 'hdist':
            if weights is None:
                raise Exception("No weights are provided")
            weights = np.asanyarray(weights)
        else:
            warnings.warn("The parameter w will be ignored")

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
            failing_indices = np.where(is_failure)[0]
            return matrix, failing_indices
        else:
            return matrix