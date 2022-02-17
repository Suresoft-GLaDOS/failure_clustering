import numpy as np

class Agglomerative:
    """
    Recursively merges the closest pair of clusters
    """
    def __init__(self, linkage):
        # Define how to calculate the inter-cluster distance
        assert linkage in ["complete", "average", "single"]
        self.linkage = linkage
        self.labels = None
        self.mdist = None # minimum intercluster distance
        
    def run(self, distance_matrix, stopping_criterion=None):
        """
        Perform Agglomerative Hierarchical Clustering
        that recursively merges the closest pair of clusters

        Returns the clustering results
        
        Parameters
        ----------
        distance_matrix : array_like
            a square-form distance matrix for failing test cases
        stopping_criterion : str, optional
            a stopping criterion used to decide where to stop merging clusters. 
            Can be None or “min_intercluster_distance_elbow”

        Returns
        -------
        labels : list
            a list of clustering results from all AHC iterations
            labels[i] stores the clustering result when # clusters is N - i.
            Only provided if `stopping_criterion` is None.
        label : ndarray, optional
            a clustering result decided by the given stopping criterion.
            Only provided if `stopping_criterion` is not None.
        """
        criteria = ['min_intercluster_distance_elbow']
        assert stopping_criterion is None or stopping_criterion in criteria
        # Initialize minimum distances and labels history
        # mdist[i] is the minimum intercluster distance when # clusters is N - i
        self.mdist = [] 
        self.labels = []

        # Initialize ic_dist (inter-cluster distance) to distance_matrix
        ic_distance = np.asanyarray(distance_matrix).copy()

        N = distance_matrix.shape[0]
        for i in range(N):
            ic_distance[i,i] = float("Inf")

        clusters = set(range(N))
        label = np.array(list(range(N)))
        self.labels.append(label)

        while len(clusters) > 1:
            label = self.labels[-1]

            # find the closest clusters i and j
            min_idx = ic_distance.argmin()
            i, j = int(min_idx/N), int(min_idx % N)
            if i > j:
                i, j = j, i
            assert i < j
            min_dist = ic_distance[i,j]

            self.mdist.append(float(min_dist))

            new_label = label.copy()
            # Merge two clusters i and j to one cluster i
            new_label[new_label == j] = i
            ic_distance[j, :], ic_distance[:, j] = float("Inf"), float("Inf")
            clusters.remove(j)
    
            # Update inter-cluster distances
            c1 = i
            in_c1 = new_label == c1
            for c2 in clusters:
                if c1 == c2:
                    continue
                in_c2 = new_label == c2
                if self.linkage == 'average':
                    new_dist = distance_matrix[in_c1, :][:, in_c2].sum()
                    new_dist /= (in_c1.sum() * in_c2.sum())
                elif self.linkage == 'single':
                    new_dist = distance_matrix[in_c1, :][:, in_c2].min()
                elif self.linkage == 'complete':
                    new_dist = distance_matrix[in_c1, :][:, in_c2].max()

                ic_distance[c1, c2], ic_distance[c1, c2] = new_dist, new_dist

            self.labels.append(new_label)

        self.mdist.append(1)

        if stopping_criterion is None:
            assert len(self.labels) == N and len(self.mdist) == N
            return self.labels
        elif stopping_criterion == 'min_intercluster_distance_elbow':
            elbow_point = np.argmax(np.diff([0.0] + self.mdist))
            return self.labels[elbow_point]
        else:
            raise Exception(f"Not supported stopping criterion. Supported criteria: {criteria}")