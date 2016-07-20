---
layout: post
title:  "Ad Click Prediction"
date:   2016-07-09
categories: paper-series parallel distributed-systems online-learning scalability
---

# Ad Click Prediction: a View from the Trenches

**Published** August 2013

[Paper link](http://dl.acm.org/citation.cfm?id=2488200)

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

[Vowpal Wabbit](https://arxiv.org/abs/1110.4198) was developed a few years before a solution to these kinds of problems, but handled several orders of magnitudes less (its dictionary size is \\(2^{18}\\) by default).

## Online Learning and Sparsity

##### Preamble

The paper uses the compressed notation for the sum of the \\(i\\)-th gradients: \\(\textbf{g}\_{1:t}=\sum\_{i=1}^t\textbf{g}\_i\\), where the gradient is computed from the logistic loss of the \\(t\\)-th example \\(\textbf{x}\_t\\), given for the prediction \\(y_t\in\{0,1\}\\) given the current weight \\(\textbf{w}\_t\\).

Predictions are made according to the sigmoid, where \\(p\_t=\sigma(\textbf{w}\_t\cdot\textbf{x}\_t)\\) and \\(\sigma(a)=(1+\exp(-a))^{-1}\\). Loss is:
\\[
\ell\_t(\textbf{w}\_t)=-y\_t\log p\_t-(1-y\_t)\log(1-p\_t)
\\]

Its unregularized gradient is:
\\[
\nabla \ell_t(\textbf{w}\_t)=(p\_t-y\_t)\textbf{x}\_t
\\]

The \\(i\\)-th element of a vector \\(\textbf{v}\\) will be be given by the unbolded letter \\(v\_i\\).

##### Sparsity

Completely-0 sparsity is essential because of the large number of features (thus susceptibility to overfitting) and because a sparse coefficient representation scales memory consumption with non-zeros.

\\(L\_1\\) penalty subgradient approaches alone aren't good enough. \\(L\_1\\) won't actively discourage zeros [like \\(L\_2\\) does](stackoverflow), but while usually considered _sparsity-inducing_ it's more accurately _sparsity ambivalent_: as a weight gets smaller, its penalty follows linearly.

Some alternative approaches have more active sparsity induction: FOBOS and RDA. I have no idea what they are, but apparently FTRL-Proximal is better anyway (see Table 1 in the paper).

##### FTRL-Proximal

FTRL-Proximal is an \\(L\_1\\)-regularized version of the Follow The Regularized Leader. FTRL is a core online learning algorithm. It improves upon a naive one, called Follow The Leader, or FTL, where the best coefficients in hindsight \\(\textbf{w}\_{t}^*\\) are chosen for the next step:

\\[
\textbf{w}\_{t+1}=\textbf{w}\_t^*=\underset{\textbf{w}}{\mathrm{argmin}}\sum\_{i=1}^t\ell\_i(\textbf{w})
\\]

Of course, this strategy can be pathologically unstable; consider the expert problem where two experts are alternatingly correct, but the one you choose first is wrong - then you'd be switching to the wrong expert each time.

Borrowing Prof. Hazan's notation \\(\nabla\_i=\nabla\ell\_i(\textbf{w}\_i)\\), the regret contribution for the \\(i\\)-th guess at time \\(t\\) is
\\(\ell\_i(\textbf{w}\_i)-\ell\_i(\textbf{w}\_{t}^*)\\).

This is bounded above by \\(\nabla\_t^T (\textbf{w}\_i-\textbf{w}\_{t}^*)\\) assuming convexity of each \\(\ell\_i\\).

If we look at the related equation for the upper bound of the total regret up to time \\(t\\):

\\[
\text{regret}\_t\le \sum_{i=1}^t \nabla\_t^T (\textbf{w}\_i-\textbf{w}\_{t}^*)
\\]

We see that FTL is equivalent to **optimizing that bound for the previous iterations**. In fact, optimizing this bound is all we _can_ do: we don't know each \\(\ell_i\\) ahead of time, so we solve the generic bound for any differentiable convex loss, only using the observed gradient \\(\nabla_i\\), instead. But, as the example above mentioned, it performs poorly because it can be unstable. FTRL introduces some notion of stability to the algorithm by changing the optimization goal to:

\\[
\textbf{w}\_{t+1}=\underset{\textbf{w}}{\mathrm{argmin}} \sum_{i=1}^t \eta_t\nabla\_t^T (\textbf{w}\_i-\textbf{w}\_{t}^*) +R(\textbf{w})
\\]

The \\(\eta\_t\\) factors control how much the regularization function affects the next guess. By observing that \\(\textbf{w}\_{t}^*\\) is a constant, and choosing \\(R\\) appropriately, we have the new convex update for a particular FTRL algorithm, FTRL-Proximal:

![ftrlprox](/assets/2016-07-17-ad-click-prediction/ftrlprox-update.png){: .center-image }

We inductively define \\(\sigma\_{1:i}=\eta\_{i}^{-1}\\) (for \\(\eta_t=O(t^{-1})\\) this is eventually montonically nonincreasing) and let \\(\textbf{g}\_i=\nabla\_i\\). As noted in the paper, if \\(\lambda_1=0\\) and \\(\eta_i=i^{-1/2}\\) then FTRL-Proximal reduces to OGD: \\(\textbf{w}\_{t+1}=\textbf{w}\_t-\eta\_t\textbf{g}_t\\) (verify this by deriving the optimization equation wrt \\(\textbf{w}\\)).

Through a derivation done fairly well in the paper, the weight update step for the proximal algorithm can be done quickly with the intermediate state \\(\textbf{z}\_t=\textbf{g}\_{1:t}-\sum\_{s=1}^t\sigma\_s\textbf{w}\_s\\). Updating the intermediate state is constant-time too.

![ftrlprox](/assets/2016-07-17-ad-click-prediction/noadapt-update.png){: .center-image }

### Per-Coordinate Learning Rates

TODO: motivation, describe how it's basically adagrad

TODO: paste the algorithm here

## TODO: Continue with section 4.

# Notes

## Observations

## Weaknesses

## Strengths

# Insight

## Takeaways

# Open Questions

* Maybe we can't solve the regret bound for the adversarial case where arbitrary differentiable convex loss functions are presented to us. However, we know \\(\ell_i\\) take a log-loss shape always. Perhaps there exists an analytical solution to the exact FTRL problem:
\\[
\textbf{w}\_{t+1}=\underset{\textbf{w}}{\mathrm{argmin}}\;\; \eta_t\sum\_{i=1}^t\left(\ell\_i(\textbf{w})-\ell\_i(\textbf{w}\_{t}^*)\right) +R(\textbf{w})
\\]