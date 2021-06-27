---
layout: post
title:  "An Infinite Simpson's Paradox (Simpson's part 2 of 3)"
date:   2020-05-01
categories: causal
featured_image: /assets/2020-simpsons-series/diagram.jpg
---

# An Infinite Simpsons Paradox (Simpson's part 2 of 3)

This is Problem 9.11 in Elements of Causal Inference.

_Construct a single Bayesian network on binary $X,Y$ and variables $\\{Z\_j\\}\_{j=1}^\infty$ where the difference in conditional expectation,
\\[
\Delta\_j(\vz\_{\le j}) = \CE{Y}{X=1, Z\_{\le j}=\vz\_{\le j}}-\CE{Y}{X=0, Z\_{\le j}=\vz\_{\le j}}\,\,,
\\]
satisfies $\DeclareMathOperator\sgn{sgn}\sgn \Delta\_j=(-1)^{j}$ and $\abs{\Delta\_j}\ge \epsilon\_j$ for some fixed $\epsilon\_j>0$. $\Delta\_0$ is unconstrained._

### Proof overview

We will do this by induction, constructing a sequence of Bayes nets $\mcC\_d$ for $d\in\N$ with variables $X,Y,Z\_1,\cdots,Z\_d$, such that $\mcC\_d\subset\mcC\_{d+1}$, in a strict sense. In particular, our nets will be nested so that they have the same structure on common variables. This means that for their entailed respective joints $p\_d,p\_{d+1}$,
\\[
p\_d(x, y, z\_{1:d})=\int\d{z\_{d+1}}p\_{d+1}(x, y, z\_{1:d+1})\,\,.
\\]

Intuitively, this seems to lead us to a limiting structure $\mcC\_\infty$ over the infinite set of nodes, but it's not clear that this necessarily exists. Our ability to generate larger probability spaces by adding independent variables doesn't help here, since those are finite tools.

For simplicity, we'll construct $X$ such that its marginal has mass $b(x)=0.5$ for $x=0,1$. We'll also take $Z\_j$ to be binary. Nonetheless, even in this simple setting, the set of realizations of $\\{Z\_j\\}\_{j=1}^\infty$ is uncountable, $2^\N$. Assigning probabilities to every subset of this set isn't easy.

So first we'll have to tackle well-definedness. What does $\mcC\_\infty$ even mean, mathematically? Equiped with this, we can specify more details about the specific $\mcC\_\infty$ we want to have that'll satisfy properties about $\Delta\_j$.

### Well-definedness of $\mcC\_\infty$

_Suppose that we have open unit interval-valued functions $\alpha\_x,\beta\_x$ for $x\in\\{0,1\\}$ on binary strings that satsify, for any $j\in\N$ and binary $j$-length string $\vz$, that
\\[
\beta\_x(\vz)=(1-\alpha\_x(\vz))\beta\_i(\vz:0)+\alpha\_x(\vz)\beta\_x(\vz:1)\,\,,
\\]
where $\vz:i$ is the concatenation operation (at the end). We construct an object $\mcC\_\infty$ defined by finite kernels $p(x|\vz\_J),p(y|x,\vz\_J),p(\vz\_J)$ (that is, for any $J\subset \N$, $\mcC\_\infty$ provides us with these functions) that induce a joint distribution over $(X, Y, Z\_J)$. Moreover, there exists unique law $\P$ Markov wrt $\mcC\_\infty$ (it is consistent with the kernels), which adheres to the following equality over binary strings $\vz$ of length $j$ with $Z=(Z\_1,\cdots, Z\_j)$:_
{% raw %}
$$
\begin{align}
\beta_i(\vz)&=\CP{Y=1}{X=i,Z=\vz}\\
\alpha_i(\vz)&=\CP{Z_j=1}{X=i,Z=\vz}\,\,.
\end{align}
$$
{% endraw %}

#### Proof

The proof amounts to specifying Markov kernels inductively in a way that respects the invariant promised, and then applying the Kolmogorov extension theorem. While this would be the way to formally go about it, an inverse order, starting from Kolmogorov, is more illustritive.

The theorem states that given joint distributions $p\_N$ over arbitrary finite tuples $N\subset \\{X,Y,Z\_1, Z\_2,\cdots\\}$ which match the consistency property $p\_K=\int\d{\vv}p\_N$, where $\vv$ is the realization of variables $N\setminus K$ for $K\subset N$, there exists a unique law $\P$ matching all the joint distributions on all tuples (even infinite ones). That is, you need be consistent under marginalization.



Before diving in, we make a few simplifications.

1. Since the variables are all binary it's easy enough to make sure all our kernels are valid conditional probability distributions; just specify a valid probability for one of the outcomes, the other being the complement.
2. We'll focus only on kernels $p(z\_j)$, $p(x|\vz\_{\le j})$, and $p(y|x, \vz\_{\le j})$ for $j\in\N$. It's easy enough to derive the other ones; for any finite $J\subset \N$, with $m=\max J$, just let
\\[
p(x|\vz\_J)=\int\d{\vz\_{[m]\setminus J}}p(x|\vz\_{\le m})p(\vz\_{[m]\setminus J})\,\,,
\\]
and analogously for $p(y|x, \vz\_{\le j})$. Thanks to independence structure, $p(\vz\_J)=\prod\_{j\in J}p(z\_j)$.

![bayes net](/assets/2020-simpsons-series/diagram.jpg){: .center-image }

##### Extension

This simplification in (2) means when checking the Kolmogorov extension condition, we needn't worry about differences in $N$ and $K$ by $Z\_j$ nodes.

Consider tuples $K\subset N$ over our variables from $\mcC\_\infty$, and denote their intersections with $\\{Z\_j\\}\_{j\in\N}$ as $Z\_{J\_K},Z\_{J\_N}$. Letting $m=\max J\_N$, 

{% raw %}
$$
\begin{align}
p_K(x, y, \vz_{J_K})&= p_K(y|x, \vz_{J_K})p_K(x|\vz_{J_K})p(\vz_{J_K})\\
&=\int\d{\vz_{[m]\setminus J_K}}p(y|x, \vz_{\le m})p(x|\vz_{\le m})p(\vz_{\le m})\\
&=\int\d{\vz_{J_N\setminus J_K}}\int\d{\vz_{[m]\setminus J_N}}p(y|x, \vz_{\le m})p(x|\vz_{\le m})p(\vz_{\le m})\\
&=\int\d{\vz_{J_N\setminus J_K}}p_N(y|x, \vz_{J_N})p_N(x|\vz_{J_N})p_N(\vz_{J_N})\\
&=\int\d{\vz_{J_N\setminus J_K}}p_N(x, y, \vz_{J_N})\,\,.
\end{align}
$$
{% endraw %}
The subscripts are important here. Of course, $p\_K(\vz\_L)=p\_N(\vz\_L)=p(\vz\_L)=\prod\_{j\in L}p(z\_j)$ for any $L\subset K\subset N$ by independence and kernel specification. Otherwise, the steps above rely on joint decomposition, then simplification (2) applied to $K$, Fubini, then simplification (2) applied to $N$ now in reverse, and finally joint distribution composition.

The above presumes $X,Y\in K$, but it's clear that we can simply add in the corresponding integrals on the right hand side to recover them if they're in $N$ after performing the steps above.

The above finishes our use of the extension theorem, relying only on the fact that we constructed valid Markov kernels to provde us a law $\P$ consistent with them. But to actually apply this reasoning, we have to explicitly construct these kernels, which we'll do with the help of simplification (1).

##### Kernel Specification

We show that there exist marginals $p(z\_j)$ such that $p(x)$ is the Bernoulli pmf $b(x)$ and $p(z\_j|x,\vz\_{<j})$ is defined by $\alpha\_x(\vz\_{<j})$. In particular, we first inductively define $p(x|\vz\_{\le j})$ and $p(z\_j)$ simultaneously. For $j=0$ set $p(x)=b(x)$. Then for $j>0$ define
\\[
p(Z\_j=1)=\int\d{(x,\vz\_{<j})}\alpha\_x(\vz\_{<j})p(x|\vz\_{<j})p(\vz\_{<j})\,\,,
\\]
which for $j=1$ simplifies to $p(Z\_1=1)=\int \d{x}p(x)\alpha\_x(\emptyset)=b(0)\alpha\_0(\emptyset)+b(1)\alpha\_1(\emptyset)$, and then use that to define
\\[
p(X=1|\vz\_{\le j})=\frac{\alpha\_x(\vz\_{\le j})p(x|\vz\_{<j})}{p(z\_j)}\,\,,
\\]
which induces $p(x|\vz\_{\le j})$. For the case of $j=1$ the above centered equation simplifies to $\alpha\_x(z\_1)b(x)/p(z\_1)$. It is evident from Bayes' rule applied to $p(x|z\_j,\vz\_{< j})$ that this is the unique distribution $p(x|\vz\_{\le j})$ matching the semantic constraint on $\alpha\_x$, assuming $Z\_j$ are independent.

Uniqueness of $p(x|\vz\_{\le j})$
follows inductively, as does
$\int \d{\vz\_{\le m}}p(x|\vz\_{\le m})p(\vz\_{\le m})=b(x)$.

While above we could _construct_ conditional pmfs and marginals pmfs to suit our needs, where they formed valid measures simply by construction (i.e., specifying any open unit interval valued $\alpha$ constructs valid pmfs above), we must now use our assumption to validate that the free function $\beta$ induces a valid measure on $Y$.

It must be the case that for any kernel we define that for all $j\in \N$,
\\[
p(y|x, \vz\_{\le j})=\int\d{z\_{j+1}}p(z\_{j+1}|x, \vz\_{\le j})p(y|x, \vz\_{\le j+1})\,\,,
\\]
which by the assumption $\beta\_x(\vz)=(1-\alpha\_x(\vz))\beta\_i(\vz:0)+\alpha\_x(\vz)\beta\_x(\vz:1)$ holds precisely when
\\[
p(Y=1|x, \vz\_{\le j})=\beta\_x(\vz\_{\le j})\,\,,
\\]
by our definition of $p(z\_{j+1})$ above. Then such a specification of kernels is valid.

### Configuring $\Delta\_j$

Having done all the work in constructing $\mcC\_\infty$, we now just need to specify $\alpha, \beta$ meeting our constraints.

To do this, it's helpful to work through some examples. We first note a simple equality, which is
\\[
\Delta\_d(\vz)=\beta\_1(\vz)-\beta\_0(\vz)\,\,.
\\]

For $d=0$, we can just take $\beta\_0(\emptyset)=\beta\_1(\emptyset)=0.5$


![contingency table](/assets/2020-simpsons-series/contingency.jpg){: .center-image }

For $d=1$, we introduce $Z\_1$. Notice we now are bound to our constraints,
{% raw %}
$$
\begin{align}
    \beta_0(\emptyset)&=(1-\alpha_0(\emptyset))\beta_i(0)+\alpha_0(\emptyset)\beta_0(1)\\
\beta_1(\emptyset)&=(1-\alpha_1(\emptyset))\beta_i(0)+\alpha_1(\emptyset)\beta_1(1)\,\,.
\end{align}
$$
{% endraw %}
At the same time, we're looking to find settings such that $\forall z,\,\,\beta\_1(z)-\beta\_0(z)\le -\epsilon$.

Luckily, our constraints amount to convexity constraints; algebraicly, this means that $\beta\_0(0),\beta\_0(1)$ must be on either side of $\beta\_0(\emptyset)$ and similarly for $\beta\_1(\emptyset)$. At the same time, we'd like to make sure that $\beta\_0(1)-\beta\_1(1)\ge \epsilon\_1$. This works out! See the picture below, which sets
\\[
(\beta\_1(0), \beta\_0(0),\beta\_1(1),\beta\_0(1))=0.5+\pa{-2\epsilon\_1,-\epsilon\_1,\epsilon\_1,2\epsilon\_1}\,\,.
\\]


![number line](/assets/2020-simpsons-series/numberline.jpg){: .center-image }

Then, choice of $\alpha\_i(\emptyset)$ meets the constraints.

For the recursive case, recall we need
\\[
\beta\_x(\vz)=(1-\alpha\_x(\vz))\beta\_x(\vz:0)+\alpha\_x(\vz)\beta\_x(\vz:1)\,\,,
\\]
so we'll choose $\beta\_x(\vz:1)>\beta\_x(\vz)>\beta\_x(\vz:0)$, which always admits solutions on the open unit interval, but to ensure that $\beta\_1(\vz:z\_j)-\beta\_0(\vz:z\_j)=(-1)^{j}\epsilon\_j$, we need another construction similar to the above with the number line. Here's the next step.


![recursive number line](/assets/2020-simpsons-series/recursive-numberline.jpg){: .center-image }

Where we can frame the above recursively. Suppose $m$ is the minimum distance from $\beta\_1(\vz),\beta\_0(\vz)$ to $0,1$. Without loss of generality assume the parity of $\vz$ is such that we're interested in having $\beta\_1(\vz:z\_j)>\beta\_0(\vz:z\_j)$, which implies by parity as well that in the previous step $\beta\_1(\vz)\le \beta\_0(\vz)$.

Then for $j=\card{\vz}+1$ set
\\[
\mat{\beta\_0(\vz:0)\\\\ \beta\_1(\vz:0)\\\\\beta\_0(\vz:1)\\\\\beta\_1(\vz:1) }=
\mat{\beta\_1(\vz)-\frac{m+\epsilon\_{j}}{2} \\\\ \beta\_1(\vz)-\frac{m-\epsilon\_{j}}{2} \\\\ \beta\_0(\vz)+\frac{m-\epsilon\_{j}}{2} \\\\ \beta\_0(\vz)+\frac{m+\epsilon\_{j}}{2} }\,\,.
\\]

This recursion keeps the equations solvable, and by choice of $\epsilon\_{j}$ sufficiently small all quantities are within $(0,1)$.
