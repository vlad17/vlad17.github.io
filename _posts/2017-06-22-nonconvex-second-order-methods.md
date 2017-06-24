---
layout: post
title:  "Nonconvex Second Order Methods"
date:   2017-06-22
categories: machine-learning optimization deep-learning
---

# Nonconvex Second Order Methods

[TODO blog post is under construction, please disregard]

This is a high-level overview of the methods for second order local improvement of nonconvex costs with a particular focus on neural networks (NNs).

Make sure to read the [general overview post]({{ site.baseurl }}{% post_url 2017-06-19-neural-network-optimization-methods %}) first. Also, we'll borrow the same setting from the introduction of the [first-order methods post]({{ site.baseurl }}{% post_url 2017-06-20-nonconvex-first-order-methods %}) and we will generally add an assumption that \\(f\in\mathcal{C}^2\\), if not even more smooth. Some of these methods might never explicitly touch the Hessian, but their analysis and intuition depend critically on it.

TODO [caution dismissal of 2nd order, like Goodfellow and Kingma in Adam do (noisiness arg)](https://stats.stackexchange.com/questions/253632/why-is-newtons-method-not-widely-used-in-machine-learning/253655)

TODO [Goodfellow 2nd order notes](http://www.deeplearningbook.org/contents/optimization.html)

TODO [overview](https://arxiv.org/abs/1706.03131)

## Full-Matrix Methods


Second-order methods are based on Newton's method for optimization. This searches for critical points by solving $\nabla J = \bsz$ iteratively. This can be done by iterative line search for $\alpha_t$ and Newton updates:
$$\bsth_{t+1}=\bsth_{t}-\alpha_{t}H_{t}^{-1}\nabla_t$$
While this has fast convergence in terms of error [TODO source; under what assumptions; perhaps look in convex opt book?], the convergence is to an $\epsilon$-critical point. As mentioned above, most critical points are saddles, which causes Newton's method to go to a poor solution.

Worse yet, full inverse Hessian construction is cubic in the parameter count. Backprop-based Hessian-vector approximations can more efficiently reconstruct Newton-like iteration, in a family of quasi-Newtonian methods; yet these suffer from the issues mentioned above.


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
