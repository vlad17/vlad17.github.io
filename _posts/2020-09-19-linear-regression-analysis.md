---
layout: post
title:  Linear Regression Analysis
date:   2020-09-19
categories: statistics
---

# Linear Regression Analysis

I just finished up reading Linear Regression Analysis by George Seber and Alan Lee. I really enjoyed it.

I came in with some background in theoretical statistics ([Keener](https://link.springer.com/book/10.1007/978-0-387-93839-4) and [van der Vaart](https://www.cambridge.org/core/books/asymptotic-statistics/A3C7DAD3F7E66A1FA60E9C8FE132EE1D)), and that really enriched the experience. That said, the first two chapters establish necessary pre-requisites, so it seems like the only real prerequisite to understand the book is mathematical maturity (calculus, linear algebra, elementary probability).

The book is fairly "old" (its revised version is from 2003 and judging by the references inside the first edition was probably written in the 80s). Despite its age, which I was initially worried would mean learning stale information, I was happy to have gotten through it for a variety of reasons. As far as the style of the text itself is concerned:

 - The authors had amazing endurance in their derivations and exposition of examples. Where lots of textbooks would say "and then it simplifies to..." Seber and Lee actually fill in the details.
 - There are tons of problems throughout the chapters and solutions to most of them in the back.
 - The authors take computation seriously, with pointers on how to actually compute quantities in practice rather than only manipulating mathematical objects for which a naive translation to code would be ill-conditioned.
 
To summarize the conceptual takeaways I got out of the book. Of course, by the end of the book you develop a familiarity with the general linear model for normal noise, which unifies ANOVA, F-tests, t-tests, z-tests, goodness-of-fit, nuisance parameters, etc. all under a single umbrella. The relationship of quadratic forms, projection, and statistical testing in this setting becomes clear, as do the BLUE guarantees. This is what you payed for, after all. Ordinary least squares may be a "classical, low dimensional" setting, but in practice it still appears with great frequency today, so the core value is there.

The above is sort-of expected. However, I was lucky to learn about a lot of things I simply never would've had exposure to. If human knowledge is a graph of concepts which, then academic discourse is, well, a distributed gossiping graph search algorithm. What I'm exposed to is a function of what I'm aware of, so even with Google many nodes are essentially inaccessible, since I don't know to search for them. I'm thankful that I can dip my toes into the pool of ideas from another time, which is bound to be in another place.

OK, enough with the metaphors. You can access my notes [here](https://github.com/vlad17/ml-notes). Cool things that I think fit into the category of unexpected takeaways are methods to do the following:

 - inference under linear constraints (if you measure the three angles of a triangle with error, how can you incorporate prior information about their sum being \\(180^{\circ}\\).
 - method of moments for inferring parameters when your _explanatory_ variables may have error
 - testing multiple hypothesis, choosing between Bonferroni, maximum modulus, and S-method
 - construct diagnostics for detecting outliers or bad linear fits
 - choose between different notions of robustness are and enforce them
 - trade off between speed and accuracy for fitting linear models in high-data settings
 - efficient and stable extensions for single-variable polynomial and spline fitting
 - compute prediction error estimators (such as, but not limited to cross-validation) analytically with only a single fit
 - decide between different model selection criteria for ordinary least squares
