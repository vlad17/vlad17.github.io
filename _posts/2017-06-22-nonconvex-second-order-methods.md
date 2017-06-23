---
layout: post
title:  "Nonconvex Second Order Methods"
date:   2017-06-22
categories: machine-learning optimization deep-learning
---

# Nonconvex Second Order Methods

This is a high-level overview of the methods for second order local improvement of nonconvex costs with a particular focus on neural networks (NNs).

Make sure to read the [general overview post]({{ site.baseurl }}{% post_url 2017-06-19-neural-network-optimization-methods %}) first. Also, we'll borrow the same setting from the introduction of the [first-order methods post]({{ site.baseurl }}{% post_url 2017-06-20-nonconvex-first-order-methods %}) and we will generally add an assumption that \\(f\in\mathcal{C}^2\\), if not even more smooth. Some of these methods might never explicitly touch the Hessian, but their analysis and intuition depend critically on it.

TODO [caution dismissal of 2nd order, like Goodfellow does](https://stats.stackexchange.com/questions/253632/why-is-newtons-method-not-widely-used-in-machine-learning/253655)

TODO [Goodfellow 2nd order notes](http://www.deeplearningbook.org/contents/optimization.html)

TODO [overview](https://arxiv.org/abs/1706.03131)

## Full-Matrix Methods

Inspiration, full gradient: [Newton's Method](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization)

[Guass-Newton](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm)

[Levenberg–Marquardt](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)

## L-BFGS

## Saddle-Free Stochastic Newton

[Dauphin et al 2014](https://arxiv.org/abs/1406.2572)

## Conjugate Gradient

[overview](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)

[list of update rules](https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method)

[Zhang and Li 2011](http://www.sciencedirect.com/science/article/pii/S0096300311007016)

## Cubic Regularization

[Agarwal et al 2017](https://arxiv.org/abs/1611.01146)

[Stochastic subsample](https://arxiv.org/abs/1705.05933)

[More analysis](https://link.springer.com/article/10.1007%2Fs10107-006-0706-8)

[Useful discussion](https://arxiv.org/pdf/1702.00763.pdf)

## Other Links

[deep-hessian-free](http://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf)

[backprop curvature](https://pdfs.semanticscholar.org/126a/a4a6d5775957b89944f958bd3307322b3348.pdf)

[Stochastic meta descent](https://pdfs.semanticscholar.org/42e2/1cd78f578fa6ce61b06b99848697da85ed76.pdf)
