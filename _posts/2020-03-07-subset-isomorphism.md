---
layout: post
title:  "Numpy Gems, Part 3"
date:   2020-03-07
categories: tools 
---
# Subset Isomorphism

Much of scientific computing revolves around the manipulation of indices. Most formulas involve sums of things and at the core of it the formulas differ by which things we're summing.

Being particularly clever about indexing helps with that. A complicated example is the [FFT](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm). A less complicated example is computing the inverse of a permutation:


```python
import numpy as np
np.random.seed(1234)
x = np.random.choice(10, replace=False, size=10)
s = np.argsort(x)
inverse = np.empty_like(s)
inverse[s] = np.arange(len(s), dtype=int)
np.all(x == inverse)
```




    True



The focus of this post is to expand on a maybe-useful, vectorizable isomorphism between indices, that comes up all the time: indexing pairs. In particular, it's often the case that we'd want to come up with an _a priori_ indexing scheme into a weighted, complete undirected graph on \\(V\\) vertices and \\(E\\) edges.

In particular, our edge set is \\(\binom{[V]}{2}=\left\\{(0, 0), (0, 1), \cdots, (V-2, V-1)\right\\}\\), the set of ordered \\(2\\)-tuples. Our index set is \\(\left[\binom{V}{2}\right]=\left\\{0, 1, \cdots, \frac{V(V-1)}{2} - 1\right\\}\\) (note we're 0-indexing here).

Can we come up with an isomorphism between these two sets that vectorizes well?

A natural question is why not just use a larger index. Say we're training a [GGNN](https://arxiv.org/abs/1511.05493), and we want to maintain embeddings for our edges. Our examples might be in a format where we have two vertices \\((v_1, v_2)\\) available. We'd like to index into an edge array maintaining the corresponding embedding. Here, you may very well get away with using an array of size \\(V^2\\). That takes about twice as much memory as you need, though.

A deeper problem is simply that you can _represent_ invalid indices, and if your program manipulates the indices themselves, this can cause bugs. This matters in settings like [GraphBLAS](http://graphblas.org/) where you're trying to vectorize classical graph algorithms.

The following presents a completely static isomorphism that doesn't need to know \\(V\\) in advance.


```python
# an edge index is determined by the isomorphism from
# ([n] choose 2) to [n choose 2]

# mirror (i, j) to (i, j - i - 1) first. then:

# (0, 0) (0, 1) (0, 2)
# (1, 0) (1, 1)
# (2, 0)

# isomorphism goes in downward diagonals
# like valence electrons in chemistry

def c2(n):
    return n * (n - 1) // 2

def fromtup(i, j):
    j = j - i - 1
    diagonal = i + j
    return c2(diagonal + 1) + i

def totup(x):
    # https://math.stackexchange.com/a/1417583
    # sqrt is valid as long as we work with numbers that are small
    # note, importantly, this is vectorizable
    diagonal = (1 + np.sqrt(8 * x + 1).astype(np.uint64)) // 2 - 1
    i = x - c2(diagonal + 1)
    j = diagonal - i
    j = j + i + 1
    return i, j

nverts = 1343
edges = np.arange(c2(nverts), dtype=int)
np.all(fromtup(*totup(edges)) == edges)
```




    True



This brings us to our first numpy gem of this post, to check that our isomorphism is surjective, `np.triu_indices`.


```python
left, right = totup(edges)
expected_left, expected_right = np.triu_indices(nverts, k=1)
from collections import Counter
Counter(zip(left, right)) == Counter(zip(expected_left, expected_right))
```




    True



The advantage over indexing into `np.triu_indices` is of course the scenario where you _don't_ want to fully materialize all edges in memory, such as in frontier expansions for graph search.

You might be wondering how dangerous that `np.sqrt` is, especially for large numbers. Since we're concerned about the values of `np.sqrt` for inputs at least 1, and on this domain the mathematical function is sublinear, there's actually _less_ rounding error in representing the square root of an integer with a double than the input itself. [Details here](https://stackoverflow.com/a/22547057/1779853).

Of course, we're in trouble if `8 * x + 1` cannot even up to ULP error be represented by a 64-bit double. It's imaginable to have graphs on `2**32` vertices, so it's not a completely artificial concern, and in principle we'd want to have support for edges up to index value less than \\(\binom{2^{32}}{2}=2^{63} - 2^{32}\\). Numpy correctly refuses to perform the mapping in this case, throwing on `totup(2**61)`.

In this case, some simple algebra and recalling that we don't need a lot of precision anyway will save the day.


```python
x = 2**53
float(8 * x + 1) == float(8 * x)
```




    True




```python
def totup_flexible(x):
    x = np.asarray(x)
    assert np.all(x <= 2 ** 63 - 2**32)
    if x > 2 ** 53:
        s = np.sqrt(2) * np.sqrt(x)
        s = s.astype(np.uint64)
        # in principle, the extra multiplication here could require correction
        # by at most 1 ulp; luckily (s+1)**2 is representable in u64
        # because (sqrt(2)*sqrt(2**63 - 2**32)*(1+3*eps) + 1) is (just square it to see)
        s3 = np.stack([s - 1, s, s + 1]).reshape(-1, 3)
        s = 2 * s3[np.arange(len(s3)), np.argmin(s3 ** 2 - 2 * x, axis=-1)]
    else:
        s = np.sqrt(8 * x + 1).astype(np.uint64)
    add = 0 if x > 2 ** 53 else 1
    diagonal = (1 + s) // 2 - 1
    diagonal = diagonal.reshape(x.shape)
    i = x - c2(diagonal + 1)
    j = diagonal - i
    j = j + i + 1
    return i, j

x = 2 ** 63 - 2 ** 32
fromtup(*totup_flexible(x)) == x
```




    True



At the end of the day, this is mostly useful not for the 2x space savings but for online algorithms that don't know \\(V\\) ahead of time.

That said, you can expand the above approach to an isomorphism betwen larger subsets, e.g., between \\(\binom{[V]}{k}\\) and \\(\left[\binom{V}{k}\right]\\) for \\(k>2\\) (if you do this, I'd be really interested in seeing what you get). To extend this to higher dimensions, you can either directly generalize the geometric construction above, by slicing through \\(k\\)-dimensional cones with \\((k-1)\\)-dimensional hyperplanes, and recursively iterating through the nodes. But, easier said than done.

That's not to say this is unilaterally better than the simpler representation \\(V^k\\). Because the space wasted by the "easy" representation \\(V^k\\) compared to this "hard" isomorphism-based one is \\(k!\\), but the objects we're talking about have size \\(n^k\\), the memory savings isn't really a good argument for using this indexing. It's not a constant worth scoffing at, but the main reason to use this is that it's online, and has no "holes" in the indexing.

[Try the notebook out yourself](/assets/2020-03-07-subset-isomorphism/subset-isomorphism.ipynb).
