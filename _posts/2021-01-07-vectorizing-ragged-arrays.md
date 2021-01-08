---
layout: post
title:  Vectorizing Ragged Arrays (Numpy Gems, Part 4)
date:   2021-01-07
categories: tools
---
Frequently, we run into situations where we need to deal with arrays of varying sizes in `numpy`. These result in much slower code that deals with different sizes individually. Luckily, by extracting commutative and associative operations, we can vectorize even in such scenarios, resulting in significant speed improvements. This is especially pronounced when doing the same thing with deep learning packages like `torch`, because vectorization matters a lot more on a GPU.

For instance, take a typical k-means implementation, which has an inner loop for a naive algorithm like the following.

```python
import numpy as np
from scipy.spatial.distance import cdist

def centroids(X_nd, label_n):
    """
    Given X_nd, a 2-dimensional array of n d-dimensional points,
    and n cluster assignments label_n (a 1-d array of n labels,
    ints in range [0, k)), return (c_kd, dist_n) the k centroids c_kd and the
    squared Euclidean distances dist_n from each point to its centroid.
    
    Intentionally zero out any empty clusters.
    """
    n, d = X_nd.shape
    k = label_n.max() + 1
    c_kd = np.zeros((k, d))
    dist_n = np.zeros(n)
    for i in range(k):
        ilabel_n = label_n == i
        if not ilabel_n.sum():
            continue
        X_id = X_nd[ilabel_n]
        c_kd[i] = X_id.mean(axis=0)
        dist_n[ilabel_n] = cdist(c_kd[i:i+1, :], X_id, 'sqeuclidean').ravel()
    return c_kd, dist_n    
```

We want to do the same thing (mean and compute pairwise square distances) to each of these mixed-size `X_id` arrays, but the `for i in range(k)` loop is difficult to vectorize.

Luckily, notice that our main reduction (`np.mean`) over the ragged arrays is a composition of two operations: `sum / count`. Extracting the reduction operation (the sum) into its own step will let us use our numpy gem, `np.cumsum` + `np.diff`, to aggregate across ragged arrays.

Then we can take adjacent differences to recover per-cluster means. This "accumulate ragged" trick will work for any respectable [ufunc](https://numpy.org/doc/stable/reference/ufuncs.html) with a negation. The key to making it work is to sort such that each cluster is contiguous.

```python
def inverse_permutation(p):
    ip = np.empty_like(p)
    ip[p] = np.arange(len(p))
    return ip

def vcentroids(X, label):
    """
    Vectorized version of centroids.
    """        
    # order points by cluster label
    ix = np.argsort(label)
    label = label[ix]
    Xz = X[ix]
    
    # compute pos where pos[i]:pos[i+1] is span of cluster i
    d = np.diff(label, prepend=0) # binary mask where labels change
    pos = np.flatnonzero(d) # indices where labels change
    pos = np.repeat(pos, d[pos]) # repeat for 0-length clusters
    pos = np.append(np.insert(pos, 0, 0), len(X))
    
    # accumulate dimension sums
    Xz = np.concatenate((np.zeros_like(Xz[0:1]), Xz), axis=0)
    Xsums = np.cumsum(Xz, axis=0)

    # reduce by taking differences of accumulations exactly at the
    # endpoints for cluster indices, using pos array
    Xsums = np.diff(Xsums[pos], axis=0)
    counts = np.diff(pos)
    c = Xsums / np.maximum(counts, 1)[:, np.newaxis]
    
    # re-broadcast centroids for final distance calculation
    repeated_centroids = np.repeat(c, counts, axis=0)
    aligned_centroids = repeated_centroids[inverse_permutation(ix)]
    dist = np.sum((X - aligned_centroids) ** 2, axis=1)
    
    return c, dist
```

```python
np.random.seed(1234)

n = 10000
d = 10
k = 10000
x = np.random.randn(n, d)
label = np.random.randint(k, size=n)
c0, dists0 = centroids(x, label)
c1, dists1 = vcentroids(x, label)
np.allclose(c0, c1), np.allclose(dists0, dists1)
```

    (True, True)

```python
%timeit centroids(x, label)
%timeit vcentroids(x, label)
```

    398 ms ± 3.27 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    3.16 ms ± 104 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

Wow! A 100x, on a CPU no less. And all thanks to using the key technique:

1. Sort by ragged array membership
1. Perform an accumulation
1. Find boundary indices, compute adjacent differences

Thanks to my friend [Ben Eisner](https://scholar.google.com/citations?user=RWe-v0UAAAAJ&hl=en) for inspiring this post with his [SO](https://stackoverflow.com/questions/65623906/pytorch-how-to-vectorize-indexing-and-computation-when-indexed-tensors-are-diff) question.


[Try the notebook out yourself.](/assets/2021-01-07-vectorizing-ragged-arrays.ipynb)

