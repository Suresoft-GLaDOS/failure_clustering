# Failure Clustering

## Environment
- Developed & tested under Python 3.9.1
- Installing dependencies:
```python
python -m pip install -r requirements.txt
```

## Installation
```
git clone https://github.com/Suresoft-GLaDOS/failure_clustering
cd failure_clustering
pip install setuptools
python setup.py install
```

## Getting started

! The full example script is provided in [`main.ipynb`](./main.ipynb)

### Calculating the distance between failing test cases via [hypergraph modeling](https://arxiv.org/pdf/2104.10360.pdf)

- prerequisite: [`SBFL-engine`](https://github.com/Suresoft-GLaDOS/SBFL)

```python
from sbfl.base import SBFL
from failure_clustering.base import FailureDistance

test_names = ["T1", "T2", "T3", "T4", "T5"]

# Coverage of T1, ..., T5
X = [
    [0, 1, 1, 0, 1, 0], # Coverage of T1 
    [1, 0, 0, 1, 0, 0], # Coverage of T2
    [1, 1, 0, 0, 1, 1], # Coverage of T3
    [0, 1, 0, 1, 1, 0], # Coverage of T4
    [1, 1, 0, 0, 1, 1], # Coverage of T5
]

# Test results of T1, ..., T5
y = [0, 0, 1, 0, 1] # 0: FAIL, 1: PASS

#Calculating weights of program elements
sbfl = SBFL(formula='Tarantula')
w = sbfl.fit_predict(X, y)
print(w)
"""
[0.25 0.4  1.   1.   0.4  0.  ]
"""

fd = FailureDistance(measure='hdist')
distance_matrix, failure_indices = fd.get_distance_matrix(
    X, y, weights=w, return_index=True)

# pairwise distances among failing test cases (T1, T2, T4)
print(distance_matrix) 
"""
[[0.         1.         0.77380952]
 [1.         0.         0.21428572]
 [0.77380952 0.21428572 0.        ]]
"""
print(failure_indices)
"""
[0 1 3]
"""
```
- Supported measures for `FailureDistance`
    - `jaccard`, `braycurtis`, `canberra`, `chebyshev`, `cityblock`, `correlation`, `cosine`, `dice`, `euclidean`, `hamming`, `jaccard`, `jensenshannon`, `kulsinski`, `kulczynski1`, `mahalanobis`, `matching`, `minkowski`, `rogerstanimoto`, `russellrao`, `seuclidean`, `sokalmichener`, `sokalsneath`, `sqeuclidean`, `yule`, `hdist`

### Running Agglomerative Hierarchical Clustering

You can provide a stopping criterion when performing clustering. If the stopping criterion is not provided, a list of clustering results at all iterations are returned.

1. The stopping criterion can be a threshold value, such as `0.5`. The clustering stops once the minimum intercluster distance exceeds the threshold.
    ```python
    from failure_clustering.clustering import Agglomerative

    aggl = Agglomerative(linkage='complete')
    clustering = aggl.run(distance_matrix, stopping_criterion=0.5)

    print(clustering)
    for i, cluster in zip(failure_indices, clustering):
        print(f"Cluster of {test_names[i]}: {cluster}")
    """
    [0, 1, 1]
    Cluster of T1: 0
    Cluster of T2: 1
    Cluster of T4: 1
    """
    ```
    - FYI, `aggl.mdist` stores the minimum intercluster distance at each iteration.

2. `min_intercluster_distance_elbow` stops merging clusters at the elbow point of the minimum intercluster distance curve

    ```python
    from failure_clustering.clustering import Agglomerative

    aggl = Agglomerative(linkage='complete')
    clustering = aggl.run(distance_matrix, 
        stopping_criterion='min_intercluster_distance_elbow')

    print(clustering)
    for i, cluster in zip(failure_indices, clustering):
        print(f"Cluster of {test_names[i]}: {cluster}")
    """
    [0, 1, 1]
    Cluster of T1: 0
    Cluster of T2: 1
    Cluster of T4: 1
    """
    ```