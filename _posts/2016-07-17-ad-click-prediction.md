---
layout: post
title:  "Ad Click Prediction"
date:   2016-07-09
categories: paper-series parallel distributed-systems online-learning scalability
---

[Paper link](http://dl.acm.org/citation.cfm?id=2488200)

# Ad Click Prediction: a View from the Trenches

**Published** August 2013

## Abstract

## Introduction

## Brief System Overview

##### Problem Statement

For any given a query, ad, and associated interaction and metadata represented as a real feature vector \\(\textbf{x}\in\mathbb{R}^d\\), provide an estimate of the probability that the user making the query will click on the ad. Solving this problem has beneficial implications for ad auction pricing in Google's online advertising business.

Further problem details:

* \\(d\\) is in the billions - naive dense representation of one data point would take GBs!
* Features are "extremely sparse"
* Serving must happen quickly, at a rate of billions of predictions per day. Though never mentioned explicitly in the paper, I'm assuming the training rate (dependent on the actual number of ads shown, not considered) is thus a small fraction of this but still considerable.

Given sparsity level and scalability requirements, online regularized logisitic regression seems to be the way to go. How do we build a holistic machine learning solution for it?

[Vowpal Wabbit](https://arxiv.org/abs/1110.4198) was developed a few years before a solution to these kinds of problems, but handled several orders of magnitudes less (its dictionary 

## Online Learning and Sparsity

##### Preamble

The paper uses the compressed notation for the sum of the \\(i\\)-th gradients: \\(\textbf{g}\_{1:t}=\sum\_{i=1}^t\textbf{g}\_i\\), where the gradient is computed from the logistic loss of the \\(t\\)-th example \\(\textbf{x}\_t\\), given for the prediction \\(y_t\in\{0,1\}\\) given the current weight \\(\textbf{w}\_t\\).

Predictions are made according to the sigmoid, where \\(p\_t=\sigma(\textbf{w}\_t\cdot\textbf{x}\_t\\) and \\(\sigma(a)=(1+\exp(-a))^{-1}\\). Loss is:
\\[
\ell\_t(\textbf{w}\_t)=-y\_t\log p\_t-(1-y\_t)\log(1-p\_t)
\\]

Its unregularized gradient is:
\\[
\nabla \ell_t(\textbf{w}\_t)=(p\_t-y\_t)\textbf{x}\_t
\\]

##### Sparsity

Completely-0 sparsity is essential because of the large number of features (thus susceptibility to overfitting) and because a sparse coefficient representation scales memory consumption with non-zeros.

\\(L\_1\\) penalty subgradient approaches alone aren't good enough. \\(L\_1\\) won't actively discourage zeros [like \\(L\_2\\) does](stackoverflow), but while usually considered "sparsity-inducing" it's more accurately "sparsity ambivalent": as a weight gets smaller, its penalty follows linearly.

Some alternative approaches have more active sparsity induction: FOBOS and RDA. I have no idea what they are, but apparently FTRL-Proximal is better anyway (see Table 1 in the paper).

##### FTRL-Proximal

FTRL-Proximal is an \\(L\_1\\)-regularized version of the Follow The Regularized Leader. Bare FTRL is a core online learning algorithm. It improves upon a naive one, called Follow The Leader, or FTL, where the historically best coefficients in hindsight are chosen for the next step:
\\[
\textbf{w}\_{t+1}=\underset{\textbf{w}}{\mathrm{argmin}}\sum\_{i=1}^t\ell_i(\textbf{w})
\\]
Of course, this strategy can be pathologically unstable; consider the expert problem where two experts are alternatingly correct, but the one you choose first is wrong - then you'd be switching to the wrong expert each time.

Let us denote gradient (continue pg. 75 hazan)
For convex losses \\(\ell\_t\\),

Update step + sigma definition

Get to closed form actual solution for update.

The above all coalesce into the following online algorithm:

TODO: algorithm1

### Per-Coordinate Learning Rates

TODO
Basically adagrad

## section 4 TODO

# Notes

## Observations

## Weaknesses

## Strengths

# Insight

## Takeaways

# Open Questions
