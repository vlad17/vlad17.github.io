---
layout: post
title:  Complex Hash Collisions
date:   2020-07-04
categories: tools
featured_image:  https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTrPLsXA0X_gaVwIUsBlpjNddQ8ego1KNODsw&usqp=CAU
---
# Complex Hash Collisions

This article investigates a fun thought experiment about the [Poisson-Binomial distribution](https://en.wikipedia.org/wiki/Poisson_binomial_distribution).

Let's imagine we're designing a large hash table. We know up-front we're going to get  \\(n \\) distinct keys, so let's number them  \\(j\in [n] \\).

Ahead of time, we're allowed to see the set of hashes the  \\(j \\)-th key will belong to. It is allowed to take on one of  \\(c\_j \\) distinct hashes,  \\(S\_j=\\{h\_1, \cdots, h\_{c\_j}\\}\subset\mathbb{N} \\), where at run time the real hash is sampled uniformly (and independently) from  \\(S\_j \\) for each  \\(j \\). The only catch is these  \\(S\_j \\) values could be really large, so maybe they're stored in a file or something, and we'd like to think about only reading them in a stream, as  \\(m \\) goes from  \\(1 \\) to  \\(n \\), we get  \\(S\_m \\), and hopefully we do some processing that takes constant memory with respect to  \\(n \\) and  \\(\left\|S\_m\right\| \\).

We get to choose the table size  \\(T \\), so the buckets key  \\(j \\) could belong in are now  \\(S\_j\pmod{T} \\) (with constituent hashes modded). A reasonable approach might be to look at an exponential table of primes  \\(T \\) and compute the expected number of collisions such a table would incur. If we could get this number of expected collisions, then we'd be able to make some space/time tradeoff to find our best  \\(T \\).

For starters, let's assume  \\(h\in S\_j \\) are all distinct modulo  \\(T \\). We'll revisit this later.

### The Poisson Binomial

Let's fix  \\(T \\), and per our assumption let's pretend  \\( S\_j\pmod{T}\subset[T] \\) is of size  \\(c\_j \\) after the modulo operation. By our assumption, each key  \\(j \\) can take on one of these hash values uniformly.

Now let's consider the number  \\(b\_i \\) of keys that fall into a particular bucket  \\(i\in[T] \\).

\\[
b\_i=\sum\_{j=1}^n1\left\\{\mathrm{hash}(j)=i\right\\}
\\]

This  \\(b\_i \\) is precisely distributed as a Poisson Binomial: each key  \\(j \\) falls in bucket  \\(i \\) with probability  \\(c\_j^{-1} \\) independently among all  \\(j \\) such that  \\(i\in S\_j\pmod{T} \\). Label this distribution  \\(\mathrm{PB}\_i \\) (note that its parameter is the full indexed set  \\(\left\\{c\_j^{-1}\,\middle\|\, j\in M\_i\right\\} \\) where  \\(M\_i=\left\\{j\,\middle\|\,i\in S\_j\pmod{T}\right\\} \\)).

So we want to know the number expected collisions,

\\[
\mathbb{E}\sum\_i 1\\{ b\_i> 1\\}= \sum\_i\mathbb{P}\\{b\_i>1\\}=\sum\_i 1-\mathbb{P}\\{b\_i=1\\}-\mathbb{P}\\{b\_i=0\\}\,\,,
\\]

and easy peasy, the probability of a  \\(\mathrm{PB}\_i \\) variable vanishing is

\\[
\prod\_{j\in M\_i}(1-c\_j^{-1})
\\]

while being exactly 1 happens with probability

\\[
\sum\_{k\in M\_i}c\_k^{-1}\prod\_{j\in M\_i\setminus k}(1-c\_j^{-1})\,\,.
\\]

### Easy Peasy?

Not so fast. How would you actually compute the last probability,  \\(\mathbb{P}\\{\mathrm{PB}\_i=1\\} \\)? You'd need to compute the set  \\(M\_i \\) entirely. There's something devious about the formula for probability of unity that isn't there in the probability of vanishing  \\(\prod\_{j\in M\_i}(1-c\_j^{-1}) \\). The latter is a commutative, associative reduction (a product) over the set  \\(M\_j \\). So we can iterate through  \\(m\in[n] \\) in a streaming manner, maintaining only one value per bucket ( \\(T \\) items total), which would be the running product  \\(\prod\_{j\in M\_i,j\le m}(1-c\_j^{-1}) \\). When we process  \\(S\_m \\), we just need to multiply each term in-place by  \\((1-c\_m^{-1}) \\) for all  \\(j\in S\_m \\).

The above works great for  \\(\mathbb{P}\\{\mathrm{PB}\_i=0\\} \\), but what about  \\(\mathbb{P}\\{\mathrm{PB}\_i=1\\} \\)? For that, we need to compute  \\(M\_i \\) for each  \\(i\in[T] \\), requiring  \\(\sum\_i\left\| M\_i\right\| \\) memory, which can be arbitrarily larger than  \\(T \\)!

What to do? One approach is to rely on approximations to the  \\(\mathrm{PB}\_i \\) distribution: Wikipedia links [Le Cam's theorem](https://en.wikipedia.org/wiki/Le_Cam%27s_theorem). Even better approximations exist, courtesy of the [Chen-Stein bound](https://www.jstor.org/stable/2325124?seq=1).

Really, give it a go, how would you compute the expression  \\(\mathbb{P}\\{\mathrm{PB}\_i=1\\} \\) with only constant memory per bucket  \\(i \\)?

### An Exact Approach

We get a head start by looking at this SO [answer by user wolfies](https://stats.stackexchange.com/a/78200/37308). In particular, it pulls in a concept called the probability generating function. The PGF  \\(G\_X \\) of a non-negative, discrete random variable  \\(X \\) is  \\(G\_X(t)=\mathbb{E}t^X \\), which, by inspection of the resulting series, satisfies  \\(\mathbb{P}\\{X=k\\}=\frac{G\_X^{(k)}(0)}{k!} \\), similar to the MGF, where  \\(G\_X^{(k)} \\) is the  \\(k \\)-th derivative of  \\(G\_X \\).

For independent discrete random variables  \\(X\_1,\cdots X\_n \\),

\\[
G\_{\sum\_j X\_j}(t)=\prod\_j G\_{X\_j}(t)\,\,.
\\]

Since  \\(\mathrm{PB}\_i \\) is the sum of  \\(\left\|M\_i\right\| \\) independent Bernoulli random variables, and a Bernoulli- \\(p \\) PGF is  \\(1-p(1-t) \\),

\\[
G\_{\mathrm{PB}\_i}(t)=\prod\_{j\in M\_i}(1-c\_j^{-1}(1-t))\,\,.
\\]

The SO answer then goes on: all we need to do is expand the polynomial  \\(G\_{\mathrm{PB}\_i}(t) \\) above into its coefficient list, and then its  \\(k \\)-th derivative at 0 divided by  \\(k! \\) is exactly the  \\(k \\)-th term's coefficient, because the  \\(k! \\) term cancels out from the falling polynomial powers as you take derivatives!

This is _extremely_ neat, and it certainly saves a bunch of computation time if we were looking for  \\(\mathbb{P}\\{\mathrm{PB}\_i=k\\} \\), since a naive computation would require  \\(\binom{n}{k} \\) terms, now reduced to expanding the  \\(n \\) coefficients, and what's more  \\(G\_{\mathrm{PB}\_i}(t) \\) is a product of terms, we can compute it by keeping a single coefficient list and updating its terms by processing  \\(S\_m \\) in a streaming manner.

Unfortunately, each update would require  \\(O(\left\|M\_i\right\|) \\) time, and we don't really get around our memory requirements. We're solving for  \\(\mathbb{P}\\{\mathrm{PB}\_i=1\\} \\) in particular, does that give us anything?

That being said, this is a _great_ jumping-off point. All we have to notice is that

\\[
\mathbb{P}\\{\mathrm{PB}\_i=1\\}=G\_{\mathrm{PB}\_i}'(0)
\\]

As we stream through  \\(m\in[n] \\), we see  \\(S\_m \\) and can imagine maintaining finite differences  \\(\prod\_{j\in M\_i,j\le m}(1-c\_j^{-1}(1-t))\bigg\|\_{t=\pm h} \\) for small  \\(h \\) for each bucket  \\(i \\). Each finite difference is updated for a new  \\(j\in S\_m \\) by multiplying a simple term  \\((1-c\_j^{-1}(1-t)) \\) for one of  \\(t\in\pm h \\), and at the end we compute the quotient.

Unfortunately, round-off error is brutal: we don't know  \\(\left\|M\_i\right\| \\) ahead of time, and it can vary for different  \\(i \\), so we have different optimal  \\(h \\) for each  \\(i \\), and the error guarantees are different...

Luckily, [complex-step differentiation](https://epubs.siam.org/doi/abs/10.1137/S003614459631241X?journalCode=siread) has none of these problems. Since  \\(G\_{\mathrm{PB}\_i} \\) is analytic,

\\[
G\_{\mathrm{PB}\_i}'(t)=\mathrm{Im}\,\frac{G\_{\mathrm{PB}\_i}(t+ ih)}{h}+O(h^2)\,\,.
\\]

So we can set  \\(h=10^{-100} \\) and then compute the complex product  \\(G\_{\mathrm{PB}\_i}(t+ ih) \\) incrementally, term-by-term, using 1 complex number of memory, and the divide by  \\(h \\) at the end without worry about numerical precision. This works out because  \\(G\_{\mathrm{PB}\_i}(t) \\) is itself a product, and thus a commutative, associative reduction. This really comes in handy when the asymptotic approximations don't hold.

On the other hand, we can approach this from the coefficient list standpoint, too: if we only care about the first  \\(k \\) terms of  \\(G\_{\mathrm{PB}}(t) \\), then we need only to update that part of the coefficient list! We look at both approaches below.

```python
# instead of specifying S_j directly, we'll focus on a single Poisson Binomial term b_i
# and see how well we can approximate.

import numpy as np

n = 1000
base = 2
p = np.logspace(np.log(1/n) / np.log(base), np.log(1/2) / np.log(base), n)
print('n =', n)
print('probabilities min {:.1g} median {:.1g} max {:.1g}'.format(*np.percentile(p, (0, 50, 100))))

ps = p.sum()
eb = (p * p).sum()/ps
print('E[X] = {:.2f} Chen-Stein total variation error bound {:.2f}'.format(ps, eb))
print('estimates for P{X=1}')
h = 1e-100
print('complex step    ', np.imag(np.prod(1 - p * (1 - 1j * h))) / h)

# recall Bernoulli PGF (1 - p) + pt
# start with coefficient list [1, 0]
def reduce_pgf1(ab, p):
    # (a + bt + O(t^2))(c + dt) = ac + (cb + ad)t + O(t^2)
    a, b = ab
    c, d = 1 - p, p
    return a * c, c * b + a * d

from functools import reduce
print('coefficient list', reduce(reduce_pgf1, p, (1, 0))[1])

cs = ps * np.exp(-ps)
print('chen-stein      ', cs, '--- guarantee is {:.4f}-{:.4f}'.format(max(cs - eb, 0), min(cs + eb, 1)))
```

    n = 1000
    probabilities min 1e-10 median 3e-06 max 0.1
    E[X] = 4.89 Chen-Stein total variation error bound 0.05
    estimates for P{X=1}
    complex step     0.03408848559058106
    coefficient list 0.03408848559058105
    chen-stein       0.03680206141293496 --- guarantee is 0.0000-0.0873

### Revisiting More General Cases

Remember how we assumed  \\(\left\|S\_j\pmod{T}\right\|=\left\|S\_j\right\| \\)? This was really to simplify notation. We can instead have the sets  \\(S\_j \\) specified as a list of arbitrary probabilities for each possible hash (so in the case of  \\(h\_1\equiv h\_2\pmod{T} \\) the probability that the  \\(j \\)-th key lands in the  \\((h\_1\bmod T) \\)-th bucket is  \\(2 c\_j^{-1} \\)); this just changes the parameters for  \\(\mathrm{PB}\_i \\).

Then, to solve the original problem, we can try out various  \\(T \\), compute their expected collisions, and decide how large we want the hash table to be, based on the space/time tradeoff we want to make.

There are, of course, other contexts where this distribution comes up (see [this robotics paper](http://asl.stanford.edu/wp-content/papercite-data/pdf/Jorgensen.Chen.Milam.Pavone.AURO2017.pdf) and [this SO question](https://stats.stackexchange.com/q/41247/37308)), and I'll admit the hashing metaphor was mostly a vehicle to give a picture of what the Poisson Binomial looks like.

As a fun note for the reader, there's an efficient two-pass streaming algorithm for computing the probability of unity that's different from both of the above approaches. It may be numerically unstable, though.

There's an interesting connection we made here between complex numbers and discrete random variables. Let's see how far we can take this in the next post.

[Try the notebook out yourself.](/assets/2020-07-04-complex-hash-collisions.ipynb)

