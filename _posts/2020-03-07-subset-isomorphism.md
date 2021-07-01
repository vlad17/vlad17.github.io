---
layout: post
title:  "Numpy Gems, Part 3"
date:   2020-03-07
categories: tools 
meta_keywords: numpy, tricks, subsets, isomorphism
featured_image: data:image/png;base64,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
---

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



The focus of this post is to expand on a vectorizable isomorphism between indices, that comes up all the time: indexing pairs. In particular, it's often the case that we'd want to come up with an _a priori_ indexing scheme into a weighted, complete undirected graph on \\(V\\) vertices and \\(E\\) edges.

In particular, our edge set is \\(\binom{[V]}{2}=\left\\{(0, 0), (0, 1), \cdots, (V-2, V-1)\right\\}\\), the set of ordered \\(2\\)-tuples. Our index set is \\(\left[\binom{V}{2}\right]=\left\\{0, 1, \cdots, \frac{V(V-1)}{2} - 1\right\\}\\) (note we're 0-indexing here).

This often comes up when needing to sample undirected edges: sure, you can write a sampler which simply re-samples or re-orders vertex pairs that aren't ordered, but that's a bit wonky and takes up extra lines of code.

Can we come up with an isomorphism between these two sets that vectorizes well (and can ideally be written in one line)?

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
