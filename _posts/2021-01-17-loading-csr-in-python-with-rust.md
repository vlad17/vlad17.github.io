---
layout: post
title:  Fast SVMlight to CSR in Python with Rust
date:   2021-01-17
categories: tools
featured_image: https://www.mathworks.com/help/examples/matlab/win64/PlotSparsityPatternExample_02.png
---

Lots of sparse datasets are kept around in a convenient text format called [SVMlight](http://svmlight.joachims.org/). It's easy to manipulate with unix tools and very easily compressed so it's perfect to distribute.

However, the main way that's available to access this format in python is dreadfully slow due to a natural lack of parallelism. [svm2csr](https://github.com/vlad17/svm2csr) is a quick python package I wrote with a rust extension that parallelizes SVMlight parsing for faster loads in python. Check it out!

P.S., here's what this format looks like:

```
-1 2:1 4:0.165975 5:0.103448 6:0.176471 11:0.285714
-1 17:0.760482 18:0.820882
1 4:0.0580913 5:0.0896552 6:0.176471 11:0.142857 21:1
```

Corresponding to labels `-1, -1, 1` and a 3-by-23 sparse matrix with 12 nonzero entries.
