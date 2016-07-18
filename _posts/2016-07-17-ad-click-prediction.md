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

##### Notation

TODO
gradient sum, sigmoid, loss

note on regret (no def, but mention + link)

##### Sparsity

Completely-0 sparsity is essential because of the large number of features (thus susceptibility to overfitting) and because a sparse coefficient representation scales memory consumption with non-zeros.

\\(L_1\\) penalty subgradient approaches alone aren't good enough. \\(L_1\\) won't actively discourage zeros [like \\(L_2\\) does](stackoverflow), but while usually considered "sparsity-inducing" it's more accurately "sparsity ambivalent": as a weight gets smaller, its penalty follows linearly.

Some alternative approaches have more active sparsity induction: FOBOS and RDA. I have no idea what they are, but apparently FTRL-Proximal is better anyway (see Table 1 in the paper).

##### FTRL-Proximal

FTRL-Proximal is an \\(L_1\\)-regularized version of the Follow The Regularized Leader.

Brief FTRL intro

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
