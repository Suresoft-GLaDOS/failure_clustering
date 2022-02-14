import numpy as np

class Hypergraph:
    def __init__(self, A, w):
        """
        - A is an N*M incidence matrix, where N is the number of vertices,
          and M is the number of hyperedges.
        - w is an M dimensional vector of nonnegative weights for the
          hyperedges.
        """
        assert ((A == 0)|(A == 1)).all()
        # A vertex should be contained in at least one hyperedge
        assert (A.sum(axis=0) > 0).all()
        # A hyperedge should contain in at least one vertex
        assert (A.sum(axis=1) > 0).all()
        assert A.shape[1] == w.shape[0]
        assert (w >= 0).all()

        has_positive_weight = w > 0
        self.A = A[:, has_positive_weight].astype(bool)
        self.w = w[has_positive_weight].astype(np.float32)
        self.N, self.M = self.A.shape

        # Initialize vertices and hyperedges
        self.vertices = [self.Vertex(self, i) for i in range(self.N)]
        self.hyperedges = [self.Hyperedge(self, i) for i in range(self.M)]

    def De(self, pow=1):
        """
        Degree diagonal matrix for hyperedges
        """
        _De = np.zeros((self.M, self.M))
        for j in range(self.M):
            _De[j,j] = np.power(self.hyperedges[j].degree, pow)
        return _De

    def W(self, pow=1):
        """
        Weight diagonal matrix for hyperedges
        """
        _W = np.zeros((self.M, self.M))
        for j in range(self.M):
            _W[j,j] = np.power(self.w[j], pow)
        return _W

    def linkage_matrix(self, vertices=None):
        """
        return _L such that _L[i,j] is the sum of w(e)/deg(e)
        for all e that connects i and j
        where i and j are the indices of vertices
        """
        if vertices is None:
            _A = self.A
        else:
            _A = self.A[vertices, :]
        _L = np.matmul(_A, 
                np.matmul(self.W(),
                    np.matmul(self.De(pow=-1), _A.T)))
        return _L

    def normalized_linkage_matrix(self, vertices=None):
        """
        return _Lhat such that _Lhat[i,j] is
        the normalized linkage between the vertex i and j
        """
        _L = self.linkage_matrix(vertices=vertices)
        _N = _L * np.eye(_L.shape[0])
        for i in range(_L.shape[0]):
            _N[i,i] = 1/_N[i,i]
        _Lhat = np.matmul(_N, _L) + np.matmul(_L, _N)
        return _Lhat/2

    def hdist_matrix(self, vertices=None):
        return 1 - self.normalized_linkage_matrix(vertices=vertices)

    # nested class
    class Vertex:
        def __init__(self, HG, index):
            self.HG = HG
            self.index = index
            self.deg = (self.HG.A[self.index, :] * self.HG.w).sum()
        
        @property
        def degree(self):
            return self.deg

    # nested class
    class Hyperedge:
        def __init__(self, HG, index):
            self.HG = HG
            self.index = index
            self.deg = self.HG.A[:, self.index].sum().astype(np.float32)
        
        @property
        def degree(self):
            return self.deg

if __name__ == "__main__":
    A = np.array([
        [0, 1, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 1, 0],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1],
    ], dtype=bool)
    w = np.array([0.25, 0.40, 1.00, 1.00, 0.40, 0.00])

    print("Incidence Matrix (Row: vertex, Column: hyperedge)")
    print(A)

    HG = Hypergraph(A, w)

    for i, vertex in enumerate(HG.vertices):
        print(f"Degree of vertex-{i}: {vertex.degree}")

    for j, hyperedge in enumerate(HG.hyperedges):
        print(f"Degree of hyperedge-{j}: {hyperedge.degree}")
    
    print(HG.hdist_matrix(vertices=[0, 1, 2]))