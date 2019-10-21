---
layout: post
title:  "Compressed Sensing and Subgaussians"
date:   2019-09-11
categories: machine-learning
---

# Compressed Sensing and Subgaussians

Candes and Tao came up with a broad characterization of compressed sensing solutions [a while ago](https://statweb.stanford.edu/~candes/papers/RIP.pdf). Partially inspired by a past homework problem, I'd like to explore an area of this setting.

This post will dive into the compressed sensing context and then focus on a proof that squared subgaussian random variables are subexponential (the relation between the two will be explained).

## Compressed Sensing

For context, we're interested in the setting where we observe an \\(n\\)-dimensional vector \\(\vy\\) that is a random linear transformation \\(X\\) of a hidden \\(p\\)-dimensional vector \\(\vx_*\\):

\\[
\vy = X\vx_*
\\]

In general, in this setting, we could have \\(p>n\\). If we wanted to recover \\(\vx_*\\), the system may be underdetermined. So a least-squares solution \\((X^\top X)^{-1}X^\top\vy\\) may not exist or may be unstable due to very small \\(\lambda_\min(X^{\top} X)\\).

In cases where we have knowledge of sparsity, however, that \\(\norm{\vx}_0=k<p,n\\), we can actually find the result.

In particular, the \\(\ell_0\\) estimator, which finds
\\(
\vx\_0=\argmin\_{\vx:\norm{\vx}\_0\le k}\norm{\vy-X\vx}\_2
\\), will converge, in the sense that the risk \\(\E\norm{\vy-X\vx}\_2\\) is bounded above by \\(O\pa{\frac{k\log p}{n}}\\). This can be used to show that under some straightforward assumptions on \\(k,X\\) we actually converge to the true answer \\(\vx\_*\\). Moreover, while this method seems to depend on \\(k\\) we can imagine doing hyperparameter search on \\(k\\).

This all looks great, in that we can recover the original entries of sparse \\(\vx\_*\\), but the problem is solving the minimization problem under the constraint \\(\norm{\vx}\_0\le k\\) is computationally difficult. This is a non-convex set of points with at most \\(k\\) non-zero entries. We'd need to check every subset to find the optimum (_question to self:_ do we really? You'd think that in a non-adversarial stochastic-\\(X\\) situation you might want to use \\(2k\\) instead of \\(k\\) and then use a greedy algorithm like backward selection and it'd be good enough).

This is why Tao and Candes' work is so cool. They take the efficiently-computable LASSO estimator,
\\[
\vx\_\lambda = \argmin\_{\vx:\norm{\vx}\_0\le k}\norm{\vy-X\vx}\_2
^2+\lambda\norm{\vx}\_1\,,
\\]
and show that under a certain condition on \\(X\\), the _Restricted Isometry Property_ (RIP), \\(\vx\_\lambda = \vx\_0\\). In essence, the RIP property requires that \\(X\\) has nearly unit eigenvalues with high probability, so it's almost an isometry. Technically, there's a relaxed condition called the restricted eigenvalue condition implied by RIP where we get a weaker result that implies LASSO has the same risk as \\(\ell_0\\).

All this is motivation for understanding the question: **what practical conditions on \\(X\\) ensure the RIP?**

It turns out we can characterize a broad class of distributions for the entries of \\(X\\) that enable this.

## Subgaussian Random Variables

Subgaussian random variables have heavy tails. In particular, \\(Y\in\sg(\sigma^2)\\) when
\\[
\E\exp(\lambda Y)\le\exp\pa{\frac{1}{2}\lambda^2\sigma^2}
\\]

By the Taylor expansion of \\(\exp\\), Markov's inequality, and elementary properties of expectation, we can use the above to show all sorts of properties.

* Subgaussian variance. \\(\var Y\le \sigma^2\\)
* Zero mean. \\(\E Y = 0\\)
* 2-homogeneity. \\(\alpha Y\in\sg(\sigma^2\alpha^2)\\)
* Light tails. \\(\P\ca{\abs{Y}>t}\le 2\exp\pa{\frac{-t^2}{2\sigma^2}}\\)
* Additive closure. \\(Z\in\sg(\eta^2 )\independent Y\\) implies \\(Y+Z\in\sg(\sigma^2+\eta^2)\\)
* Higher moments. \\(\E Y^{4k}\le 8k(2\sigma)^{4k}(2k-1)!\\)

## Subexponential Random Variables

Subexponential random variables are like subgaussians, but their tails can be heavy. In particular, \\(Y\in\se(\sigma^2,s)\\) satisfies the equation for \\(\sg(\sigma^2)\\) for \\(\abs{\lambda}<s\\).

We don't really need to know much else about these, but it's clear we can show similar additive closure and homogeneity properties as in the subgaussian case as long as we do bookkeeping on the second parameter \\(s\\).

It turns out that RIP holds for \\(X\\) with high probability if \\(\vu^\top X^\top X\vu\in\se(nc, c')\\) for some constants \\(c,c'\\) and any unit vector \\(\vu\\).

When entries of \\(X\\) are independent and identically distributed, \\(\vu\\) can essentially be taken to be a standard unit vector without loss of generality. This requires some justification but it's intuitive so I'll skip it for brevity. This lets us simplify the problem to asking if \\(\norm{X\_1}^2\in\se(nc, c')\\), where \\(X\_1\\) is the first column of \\(X\\).

So let's take the entries of \\(X\\) to be iid, which, due to additive closure, means that the previous condition can just be \\({X}_{11}^2\in\se(c,c')\\).

## Squared Subgaussians

Turns out, if the entries of \\(X\\) are subgaussian and iid, all of the above conditions hold. In particular, we need to show that the first entry \\(X_11\\), when squared, is squared exponential.

We focus on a loose but good-enough bound for this use case.

Suppose \\(Z\in\sg(\sigma^2)\\). Then \\(Z^2-\E Z^2\in \se(c\sigma^4,\sigma^{-2}/8)\\), again, being very loose with the bound here.

First, consider an arbitrary rv \\(Y\\). By the conditional Jensen's Inequality, for any \\(\lambda\\) and \\(Y'\sim Y\\) iid,
\\[
\E\exp\pa{\lambda (Y-\E Y)}=\E\exp\pa{\CE{\lambda (Y-Y')}{Y}}\le \E\CE{\exp\pa{\lambda (Y-Y')}}{Y}=\E\exp\pa{\lambda (Y-Y')}\,.
\\]
Then let \\(\epsilon\\) be an independent Rademacher random variable, and notice we can replace \\(Y-Y'\disteq \epsilon(Y-Y')\\) above. Now choose \\(Y=X^2\\). Then by Taylor expansion and dominated convergence,
\\[
\E\exp\pa{\lambda \pa{X^2-\E X^2}}\le \E \exp\pa{\lambda  \epsilon \pa{X^2-(X')^2}}=\sum_{k=0}^\infty\frac{\lambda^k\E\ha{\epsilon^k(X^2-(X')^2)^k}}{k!}\,.
\\]
Next, notice for odd \\O(k\\), \\(\epsilon^k=\epsilon\\) so by symmetry the odd terms vanish, leaving the MGF bound
\\[
\E\exp\pa{\lambda \pa{X^2-\E X^2}}\le\sum_{k=0}^\infty\frac{\lambda^{2k}\E\ha{\pa{X^2-(X')^2}^{k}}}{(2k)!}\le 2\sum_{k=0}^\infty\frac{\lambda^{2k}\E\ha{X^{4k}}}{(2k)!}\,,
\\]
where above we use the fact that \\(x\mapsto x^p\\) is montonic and \\(\abs{X^2-(X')^2}\le X^2\\) when \\(\abs{X}>\abs{X'}\\), which occurs half the time by symmetry. The other half of the time, we get an equivalent expression. By subgaussian higher moments,
\\[
\E \exp\pa{\lambda (X^2-\E X^2)}\le 1+c\sum_{k=1}^\infty \frac{k\pa{4\sigma^2\lambda}^{2p}(2k-1)!}{(2k)!}=1+c\sum_{p=1}^\infty\pa{4\sigma^2\lambda}^{2p}
\\]
Next we assume, crudely, that \\(4\sigma^2\lambda\le 2^{-1/2}\\), so the head of the series above is at least as large as the tail (since the ratio decreases by at least \\(1/2\\)). Then,
\\[
\E \exp\pa{\lambda (X^2-\E X^2)}\le 1+c(2\sigma^2\lambda)^2\le \exp(c\sigma^4\lambda^2)\,.
\\]
