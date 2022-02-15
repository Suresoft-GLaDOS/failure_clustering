import numpy as np

class Agglomerative:
    def __init__(self, linkage):
        # Define how to calculate the inter-cluster distance
        assert linkage in ["complete", "average", "single"]
        self.linkage = linkage

        # minimum intercluster distance
        self.mdist = None
        self.labels = None
        
    def run(self, distance_matrix, stopping_criterion=None):
        """
        Return k clustering candidates and the corresponding minimum intercluster distances where k is the number of objects
        - labels[i] is the clustering result when # clusters is k - i
        - mdist[i] is the minimum intercluster distance when # clusters is k - i
        """
        # Initialize minimum distances and labels history
        self.mdist = []
        self.labels = []

        # Initialize ic_dist (inter-cluster distance) to distance_matrix
        ic_distance = distance_matrix.copy()

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
            in_i = new_label == i
            for c in clusters:
                if i == c:
                    continue
                in_c = new_label == c
                if self.linkage == 'average':
                    new_dist = distance_matrix[in_i, :][:, in_c].sum()
                    new_dist /= (in_i.sum() * in_c.sum())
                elif self.linkage == 'single':
                    new_dist = distance_matrix[in_i, :][:, in_c].min()
                elif self.linkage == 'complete':
                    new_dist = distance_matrix[in_i, :][:, in_c].max()

                ic_distance[i, c], ic_distance[c, i] = new_dist, new_dist

            self.labels.append(new_label)

        self.mdist.append(-1)

        if stopping_criterion is None:
            return self.labels
        else:
            raise Exception("Not supported stopping criterion")