---
layout: post
title:  "Nonconvex First Order Methods"
date:   2017-06-20
categories: machine-learning optimization deep-learning
---

# Nonconvex First Order Methods

This is a high-level overview of the methods for first order local improvement optimization methods for non-convex, Lipschitz, (sub)differentiable, composite, and regularized functions with efficient derivatives, with a particular focus on neural networks (NNs).

\\[
\argmin\_\vx f(\vx) = \argmin\_\vx \frac{1}{m}\sum\_{i=1}^mf\_i(\vx)+\Omega(\vx)
\\]


Make sure to read the [general overview post]({{ site.baseurl }}{% post_url 2017-06-19-neural-network-optimization-methods %}) first.

Notation (link to ml-notes) TODO

## Setting

To quickly recap the setting, the task is to find the minimum


## Algorithm 1

TODO
FTRL, rda

SGD, SGD + Momentum, AdaGrad (adadelta, Dual averaging),  RMSProp + Momentum, Adam
Momentum vs modern acceleration

Variance reduction?
