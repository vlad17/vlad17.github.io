---
layout: post
title:  "Stop Anytime Multiplicative Weights, Part 1"
date:   2019-12-12
categories: machine-learning
---

# Stop Anytime Multiplicative Weights, Part 1

Multiplicative weights is a simple, randomized algorithm for picking an option among \\(n\\) choices against an adversarial environment.

The algorithm has widespread applications, but its analysis frequently introduces a learning rate parameter, \\(\epsilon\\), which we'll be trying to get rid of.

In this first post, we introduce multiplicative weights and make some practical observations. We follow [Arora's survey](https://www.cs.princeton.edu/~arora/pubs/MWsurvey.pdf) for the most part.

## Problem Setting

We play \\(T\\) rounds. On the \\(t\\)-th round, the player is to make a (possibly randomized) choice \\(I_t\in[n]\\), and then observes the losses for all of the choices, the vector \\(M\_{\cdot, j\_t}\\), corresponding to the \\(j\_t\\)-th column of a matrix \\(M\\) unknown to the player. Note \\(j\_t\\) can be any fixed sequence here, perhaps adversarially chosen with advance knowledge of the distribution of all \\(I\_t\\) but not the actual chance value of \\(I\_t\\) itself.

The goal is to have vanishing regret; that is, our average loss should tend to the best loss we'd be observing for a single fixed choice in hindsight.
\\[
\frac{1}{T}\mathbb{E}\max_{i}\left(\sum\_t M(I\_t, j\_t) - M(i, j\_t)\right)
\\]

This turns out to be a powerful, widely applicable setting, precisely because we have guarantees in spite of any selected sequence of columns \\(j\_t\\), possibly adversarially chosen.

It turns out the above expected regret will have the same guarantees as psuedo-regret \\(\mathbb{E}\left[\sum\_t M(I\_t, j\_t)\right] - \min_i \sum\_tM(i, j\_t)\\) because in our setting \\(j\_t\\) is fixed ([other adversaries exist](http://localhost:4000/2019/12/12/stop-anytime-multiplicative-weights-pt1.html), but in this setting where the algorithm doesn't depend on its own choices even a weak adaptive adversary would work just as well as one that specifies its sequence up-front).

## Psuedo-regret Guarantees

intros M notation, w0, update rule

**Fixed-rate MW theorem**.

_Proof_. TODO(HERE)

Whoopie. We proved it. Let's look into the implications of such a theorem.

## Stock Market Experts

There's a fairly classical example here where we ask \\(n\\) experts if they think the stock market is going to go up or down the next day. We can view this as an \\(n\times 2^n\\) binary matrix where we can choose one of these experts' advice each day and then the outcome is whether each expert was correct (each column corresponding to one of the \\(2^n\\) binary outcomes for each of the experts.

The theorem then guarantees that if we pick an expert according to the MWUA we'll on average have a correct guess rate within a diminishing margin of the best expert.

But that example is kind of boring and outdated. Ever since the dot-com bubble the stock market diversified its outcome set from just "up" or "down" at the end of the day. 


TODO(HERE) the real experiment where experts = vanguard ETFs, monthly rebalance, s&p adjusted returns.

Plot portfolio from best expert, MW (eps 0.01), uniform investment in all ETFs, s&P.

One nice thing about this example is it exemplifies the online nature of the algorithm: you don't need to specify \\(A\\) up-front, and can still capture adversarial phenomena like stock markets (as long as you're not such a large player that you start affecting the stock market with your live order).

As the rest of the survey explores, this is really useful when \\(A\\) is impossibly large but it's easy to find the adversarial column \\(j\_t\\) using its structure.

## The \\(\epsilon\\) Constant

To me, the elephant in the room is still this \\(\epsilon\le 1/2\\) that I choose. Let's see how the literature suggests we choose this number.

Let's bound first the "best action". We'll work with a slightly more common scenario where the largest entry of \\(M\\) is \\(\rho\\) (to apply MWUA one simply needs to feed loss \\(M/\rho\\), then in the final equation we need to replace all \\(M\\) with \\(M/\rho\\) as well).

Then \\(\min\_i M(i, j\_t)\le \rho\\). More generically, any upper bound \\(\lambda\\) on the game value \\(\lambda^*=\max\_\mathcal{P}\min\_iM(i,\mathcal{P})=\min\_\mathcal{D}\max\_jM(\mathcal{D}, j)\\) suffices:

\\[
\min_i M(i, j\_t)\le\max\_j\min\_i\le \lambda^*\le \lambda
\\]

Based on this, the regret bound above, which requires only knowledge of \\(\lambda\\), can be reduced to:
\\[
\frac{1}{T}\sum_t M(\mathcal{D}\_t, j\_t)\le \frac{\rho \log n}{\epsilon T} +\epsilon \lambda
\\]

Then for a fixed budget \\(T\\) its easy enough to observe that the optimal \\(\epsilon\_* (T)=\sqrt{\frac{\rho \log n}{\lambda T}} \\) (assuming values are large enough we don't need to worry about \\(\epsilon \le 1/2 \\) ), giving a gap of \\(2\sqrt{\frac{\lambda \rho \log n}{T }} \\)

Analogously, for fixed up-front \\(\epsilon\\), we get an optimal \\(T_*(\epsilon)=\frac{\rho \log n}{\lambda \epsilon^2}\\) with a gap of \\(2\epsilon\lambda\\)

Of course we'll do best by choosing \\(\lambda=\lambda^* \\), but figuring that value out takes solving the game, which is what we're trying to do in the first place ([Freund and Schapire 1999](https://cseweb.ucsd.edu/~yfreund/papers/games_long.pdf)).

In some scenarios, we might be looking to reach some absolute value of regret \\(\delta\\) as fast as possible, in which case Corollary 4 of [the survey](https://www.cs.princeton.edu/~arora/pubs/MWsurvey.pdf) essentially makes the same \\(\rho=\lambda > \lambda^* \\) upper bound, then since we know at best we can have \\(2\epsilon \lambda = \delta \\), where then \\(T\\) should be \\(  \frac{4\rho\lambda \log n }{\delta^2} \\). Note we differ from Corollary 4 by not allowing negative losses.

## Some Motivation

The above approaches gave us a few settings for \\(\epsilon\\).

* If you know your time horizon \\(T\\), use \\(\epsilon\_*(T) \\).
* If you want to get to regret \\(\delta\\), use \\(\epsilon\_\delta =\frac{\delta}{2\lambda} \\) and \\(T\_*(\epsilon\_\delta ) \\).

In fact, for settings where \\(\rho=1\\), the Freund and Schapire paper finds you cannot improve on this \\(T\_*(\epsilon\_\delta)\\) rate even by a constant with any online algorithm. So that's good to know, it's just downhill from there.

The Arora paper further shows that as long as \\(\rho = O(n^{1/8})\\) the lower bound
\\[
T=\Omega\left(\frac{\rho^2\log n}{\delta^2}\right)
\\]
holds up, where they find a matrix with \\(\lambda^*=\Omega(\rho)\\).

**However**, this isn't super helpful to our practical [Stock Market Experts](#stock-market-experts) example. I don't know my horizon \\(T\\). Maybe it's an exercise in modeling when I want to retire, but it sure would be nice to have a portfolio with a guarantee that I can pull out of the market any time and it won't be a problem in terms of regret guarantees.

Note that's not the case with fixed rates: if I build a portfolio using \\(\epsilon\_* (T\_1) \\) but pull out early at \\(T\_2 < T\_1 \\), my regret ratio can underperform by a factor growing in \\(\sqrt{ T\_1 /T\_2 }\\) compared to the \\(\epsilon\_* (T\_2) \\) portfolio, and vice versa if I stay in too long.

What would give me peace of mind would be an algorithm that gives the guarantee, such that for any time horizon \\(T\\), we can get performance within some fixed constant of the expected regret for the optimal mixture weights algorithm \\(\epsilon\_* (T) \\), something called an anytime algorithm from [multi-armed bandits](https://hal.inria.fr/hal-01736357).

TODO(new post): experimental evidence for 1/sqrt(t)

TODO(new post): stochastic adversary, just an identification problem. try OFU (different variances).
