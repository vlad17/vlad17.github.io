---
layout: post
title:  FIEFdom Dualization
date:   2020-12-13
categories: tools
---
# FIEFdom Dualization

[AAA](http://aaa.princeton.edu/orf523) once taught me a generic dualization technique, which I remember with a mnemonic FIEFdom. Being able to take duals on the fly has been really useful in practical optimization scenarios for me, because they:

* may provide computationally faster ways to solve primal problems. For instance, [SVM](http://localhost:4000/2020/12/13/fiefdom-dualization.html) optimization can avoid runtime costs that scale in feature dimensionality by using dual optimization.
* provide certificates for lower bounds of optimization problems. For global optimization problems we can use random search and know when to stop, because if you're within  \\(\epsilon \\) of a dual problem solution in objective value, you're at most that far away from your optimum in the primal.

FIEFdom lets you derive Lagrangian duals for mathematical programs. In particular, given a primal problem of the form (everything except  \\(p^\*,f,d^\*,f' \\) below being vector-valued, inequalities holding entrywise)

\\[
p^*=\min\_x f(x)\,\text{s.t.}\,g(x)=0,h(x)\le 0\,,
\\]

we'd like to derive a dual problem

\\[
d^*=\max\_y f'(y)\,\text{s.t.}\,g'(y)=0,h'(y)\le 0
\\]

where  \\(d^\*\le p^\* \\) (weak duality) and whenever possible  \\(d^\*=p^\* \\) (strong duality).

## LP Example

When I first learned linear programming, I rotely commited standard forms to memory. For LPs, we could just wedge them into the form of one of these primal-dual pairs by reshaping the problem with additional equality constraints, where something of the form

\\[
p^* = \min\_x c^\top x\,\text{s.t.}\,Ax=b,x\ge 0
\\]

has dual

\\[
d^* = \min\_y b^\top y\,\text{s.t.}\,A^\top y\le c\,.
\\]

But in more generic settings or when we don't remember the above, we need FIEFdom!

## FIEFdom

FIEFdom stands for

1. **F**lip the optimization direction
2. **I**mply to rid original direction
3. **E**quate instead of imply
4. **F**ree bound quantifiers, lifting to optimization **dom**ain.

Let's see how it works for LPs.

#### LP FIEFdom

We start off with our original LP,

\\[
p^* = \min\_x c^\top x\,\text{s.t.}\,Ax=b,x\ge 0\,.
\\]

Introduce a trivial lower bound such that  \\(\gamma\le c^\top x \\), so  \\(\max\_\gamma \gamma=p^* \\). This **flips** the optimization direction to go in the direction of the dual.

\\[
p^* = \max\_\gamma\min\_x\gamma \,\text{s.t.}\,Ax=b,x\ge 0,\gamma\le c^\top x\,.
\\]

We reformulate the minimization where  \\(x\in \\{x': Ax=b,x\ge0\\} \\) as an equivalent **implication**:

\\[
p^* = \max\_\gamma\gamma \,\text{s.t.}\,\forall x\,Ax=b,x\ge 0\implies \gamma\le c^\top x\,.
\\]

The **equation** replacement for the implication is the most sophisticated step. So far, we've been modifying our primal to be exactly equal to the original problem. What we'll do now is re-write our predicate that defines the domain of optimization. Right now our single-variable program is just the maximum of the set  \\(S=\\{\gamma: \forall x\,Ax=b,x\ge 0\implies \gamma\le c^\top x\\} \\). We'll come up with a predicate  \\(\exists\lambda\,\exists\mu\ge 0\,\forall x\,\mathcal{L}(x, \lambda, \mu)=\gamma \\) that implies _the implication_  \\(\forall x\,Ax=b,x\ge 0\implies \gamma\le c^\top x \\). This means that the set  \\(L=\\{\gamma:\exists\lambda\exists\mu\ge 0\forall x\, \mathcal{L}(x, \lambda, \mu)=\gamma\\} \\) is smaller, i.e.,  \\(L\subset S \\), and thus  \\(d^*=\max L\le \max S = p^* \\). We haven't defined  \\(\mathcal{L} \\) yet but after we do we'll achieve weak duality by construction.

Define  \\(\mathcal{L}(x,\lambda,\mu)=c^\top x - (Ax-b)^\top \eta-x^\top \mu \\) (i.e., the Lagrangian). This makes our statement defining  \\(L \\) into

\\[
(*)\, \exists\lambda\,\exists\mu\ge 0\,\forall x\, (Ax-b)^\top \eta+x^\top \mu = c^\top x -\gamma\,.
\\]

Now, assuming this holds, then does the implication  \\(\forall x\, Ax=b,x\ge 0\implies 0\le c^\top x-\gamma \\)? Indeed! Assuming the antecedent the left hand side of the equality  \\((*) \\) simplifies to  \\(x^\top\mu\ge 0 \\) since both vectors are positive, proving the conclusion. It's worth noting that we'll always be able to do this, even for nonlinear nonconvex programs, when  \\(\lambda \\) is associated with the equality constraints and  \\(\mu \\) is associated with the less-than-equal-to constraints.

After making this substitution, we know the objective value may go down, so we have now

\\[
p^\*\ge d^\*=\max\_\gamma \gamma\,\text{s.t.}\,\exists\lambda\,\exists\mu\ge 0\,\forall x\,\mathcal{L}(x, \lambda, \mu)=\gamma\,.
\\]

To **free** ourselves of these quantifiers, let's take a closer look at  \\((*) \\):

\\[
\exists\lambda\,\exists\mu\ge 0\,\forall x\,x^\top (A^\top\eta+\mu)-b^\top \eta = x^\top c -\gamma\,.
\\]

For both sides of a polynomial equation to be equal, we must have identical coefficients. Thus we have an identical predicate (and thus identical maximization program):

\\[
\exists\lambda\,\exists\mu\ge 0\,A^\top\eta+\mu=c,b^\top \eta =\gamma\,,
\\]

which by positivity of  \\(\mu \\) is true iff

\\[
\exists\lambda\,A^\top\eta\le c,b^\top \eta =\gamma\,,
\\]

and thus by replacing  \\(\gamma \\) in the original optimization problem **domain** yields

\\[
p^\*\ge d^\*=\max\_\eta b^\top \eta\,\text{s.t.}\, A^\top\eta\le c\,,
\\]

which is indeed the LP dual!

#### Strong Duality

What I like about this dualization algorithm is that it sheds a new light on strong duality. The duality gap appears when  \\(L\subsetneq S \\) from the equation step.

So we have strong duality precisely when  \\(S\subset L \\), i.e., in the case of an LP, the implication

\\[
\forall x\,Ax=b,x\ge 0\implies \gamma\le c^\top x
\\]

itself implies 

\\[
\exists\lambda\,\exists\mu\ge 0\,\forall x\, (Ax-b)^\top \eta+x^\top \mu = c^\top x -\gamma\,.
\\]

In the case of LPs, this implication is well-known as [Farkas' Lemma](https://en.wikipedia.org/wiki/Farkas%27_lemma), showing that for the above \\(d^\*=p^\* \\). In general, for arbitrary, possibly nonconvex programs, showing these linear equations essentially amounts to strong alternatives ([Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/), Chapter 5).


