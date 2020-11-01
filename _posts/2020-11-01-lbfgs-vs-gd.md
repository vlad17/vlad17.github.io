---
layout: post
title:  L-BFGS vs GD
date:   2020-11-01
categories: tools
---
# L-BFGS vs GD

Curiously, the original L-BFGS convergence proof essentially reduces the L-BFGS iteration to GD. This establishes L-BFGS converges globally for sufficiently regular functions and also that it has local linear convergence, just like GD, for smooth and strongly convex functions.

But if you look carefully at the proof, the construction is very strange: the more memory L-BFGS uses the less it looks like GD, the worse the smoothness constants are for the actual local rate of convergence. I go to into more detail on this in my [SO question](https://scicomp.stackexchange.com/questions/36166/when-does-l-bfgs-outperform-gd) on the topic, but I was curious about some empirical assessments of how these compare.

I found a [study](http://math.furman.edu/~mwoodard/fuejum/content/2007/paper2_2007.pdf) which confirms high-level intuition: L-BFGS interpolates between CG and BFGS as you increase memory. This relationship is true in a limiting sense: when  \\(L=0 \\) L-BFGS is equal to a flavor of CG (with exact line search) and when  \\(L=\infty \\) it's BFGS. BFGS, in turn, iteratively constructs approximations  \\(B\_k \\) to the Hessian which eventually satisfy a directional inequality  \\(\\|(B\_k-\nabla\_k^2)\mathbf{p}\_k\\|=o(\\|\mathbf{p}\_k\\|) \\) where  \\(\mathbf{p}\_k=-B\_k^{-1}\nabla\_k \\) is the descent direction, which it turns out is enough to be "close enough" to Newton that you can achieve superlinear convergence rates.

So, to what extent does agreement between  \\(\mathbf{p}\_k,\nabla\_k \\) (measured as  \\(\cos^2 \theta\_k \\), the square of the cosine of the angle between the two) explain fast L-BFGS convergence? How about the magnitude of the Hessian-approximate-BFGS-Hessian agreement along the descent direction  \\(\\|(B\_k-\nabla\_k^2)\mathbf{p}\_k\\| \\)? What about the secant equation difference?

One interesting hypothesis is that the low-rank view L-BFGS has into the Hessian means that it can't approximate the Hessian well if its eigenvalues are spread far apart (since you need to "spend" rank to explore parts of the eigenspace).

Let's take some simple overdetermined least squares systems with varying eigenspectra and see how all the metrics above respond.

```python
from numpy_ringbuffer import RingBuffer
import numpy as np
from scipy.stats import special_ortho_group
from scipy import linalg as sla
%matplotlib inline
from matplotlib import pyplot as plt
from scipy.optimize import line_search

class LBFGS:
    def __init__(self, m, d, x0, g0):
        self.s = RingBuffer(capacity=m, dtype=(float, d))
        self.y = RingBuffer(capacity=m, dtype=(float, d))
        self.x = x0.copy()
        self.g = g0.copy()
    
    def mvm(self, q):
        q = q.copy()
        m = len(self.s)
        alphas = np.zeros(m, dtype=float)
        for i, (s, y) in enumerate(zip(reversed(self.s), reversed(self.y))):
            inv_rho = s.dot(y)
            alphas[m - i - 1] = s.dot(q) / inv_rho
            q -= alphas[m - i - 1] * y
        if m > 0:
            s = next(reversed(self.s))
            y = next(reversed(self.y))
            gamma = s.dot(y) / y.dot(y)
        else:
            gamma = 1
        z = gamma * q
        for (alpha, s, y) in zip(alphas, self.s, self.y):
            inv_rho = s.dot(y)
            beta = y.dot(z) / inv_rho
            z += s * (alpha - beta)
        return -z
    
    # mvm(self, self.g) gives current lbfgs direction
    # - H g
    
    def update(self, x, g):
        s = x - self.x
        y = g - self.g
        if self.s.is_full:
            assert self.y.is_full
            self.s.popleft()
            self.y.popleft()
        self.s.append(s)
        self.y.append(y)
        self.x = x.copy()
        self.g = g.copy()
```

```python
def haar(n, d):
    # https://nhigham.com/2020/04/22/what-is-a-random-orthogonal-matrix/
    assert n >= d
    z = np.random.normal(size=(n, d))
    if n > d:
        q, r = sla.qr(z, mode='economic')
    else:
        q, r = sla.qr(z, mode='full')
    assert q.shape[1] == d, (q.shape[1], d)
    return q

np.random.seed(1234)
d = 100
n = 1000
vt = haar(d, d)
u = haar(n, d)

# bottom singular value we'll keep at 1
# so top determines the condition number
# for a vector s of singular values
# A = u diag(s) vt
# objective = 1/2 ||Ax - 1||_2^2

x0 = np.zeros(d)
b = np.ones(n)

def xopt(s):
    return vt.T.dot(u.T.dot(b) / s)

def objective(s, x):
    vtx = vt.dot(x)
    Ax = u.dot(s * vtx)
    diff = Ax - b
    f = diff.dot(diff) / 2
    g = vt.T.dot(s * (s * vtx - u.T.dot(b)))
    return f, g

def hessian_mvm(s, q):
    return vt.T.dot(s * (s * vt.dot(q)))

def gd(s, max_iter=1000, tol=1e-12):
    x = x0.copy()
    xsol = xopt(s)
    fsol = objective(s, xsol)[0]
    gaps = []
    for _ in range(max_iter):
        f, g = objective(s, x)
        gaps.append(abs(f - fsol))
        if gaps[-1] < tol:
            break
        alpha = line_search(
            lambda x: objective(s, x)[0],
            lambda x: objective(s, x)[1],
            x, -g, maxiter=1000,
            c1=0.2, c2=0.8)
        if alpha[0] is None:
            raise RuntimeError(alpha)
        x -= alpha[0] * g
    return gaps

def lbfgs(s, m, max_iter=1000, tol=1e-12, extras=False):
    L = s.max()
    x = x0.copy()
    xsol = xopt(s)
    fsol = objective(s, xsol)[0]
    gaps = []
    if extras:
        newton = []
        cosine = []
    
    f, g = objective(s, x)
    opt = LBFGS(m, d, x, g)

    for i in range(max_iter):
        gaps.append(abs(f - fsol))
        p = opt.mvm(opt.g)
        
        if extras:
            newton.append(np.linalg.norm(
                hessian_mvm(s, p) - opt.mvm(p)
            ) / np.linalg.norm(p))
            cosine.append(1 - p.dot(-g) / np.linalg.norm(p) / np.linalg.norm(g))

        
        if gaps[-1] < tol:
            break        
        alpha = line_search(
            lambda x: objective(s, x)[0],
            lambda x: objective(s, x)[1],
            x, p, maxiter=1000,
            c1=0.2, c2=0.8)
        if alpha[0] is None:
            raise RuntimeError(alpha)
        x += alpha[0] * p
        f, g = objective(s, x)
        opt.update(x, g)
    if extras:
        return gaps, newton, cosine
    return gaps
```

```python
for kappa, ls in [(10, '-'), (50, '--')]:
    s = np.linspace(1, kappa, d)
    gds = gd(s)
    memory = 10
    lbs = lbfgs(s, memory)
    matrix_name = 'linspace eigenvals'
    plt.semilogy(gds, c='b', label=r'GD ( \\(\kappa = {kappa} \\))'.format(kappa=kappa), ls=ls)
    plt.semilogy(lbs, c='r', ls=ls,
                 label=r'L-BFGS ( \\(m = {memory}, \kappa = {kappa} \\))'.format(
                     kappa=kappa, memory=memory))
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.xlabel('iterations')
plt.ylabel('function optimality gap')
plt.title(matrix_name)
plt.show()
```

![png](/assets/2020-11-01-lbfgs-vs-gd_files/2020-11-01-lbfgs-vs-gd_3_0.png)

OK, so pretty interestingly, L-BFGS is still fundamentally linear in terms of its convergence rate (which translates to  \\(\log \epsilon^{-1} \\) speed for quadratic problems like ours). But clearly it gets better bang for its buck in the rate itself. And this is obviously important---even though the  \\(\kappa = 50 \\) GD is still "exponentially fast", it's clear that the small slope means it'll still take a practically long time to converge. We know from [explicit analysis](https://calculus.subwiki.org/wiki/Gradient_descent_with_exact_line_search_for_a_quadratic_function_of_multiple_variables) that the GD linear rate will be something like  \\(((\kappa - 1)/\kappa)^2 \\). If you squint really hard, that's basically  \\(((1-\kappa^{-1})^\kappa)^{2/\kappa}\approx e^{-2/\kappa} \\) for large  \\(\kappa \\), which is why our "exponential rates" look not-so-exponential, especially for  \\(\kappa \\) near the number of iterations (because then the suboptimality gap looks like  \\(r^{T/\kappa} \\) for  \\(T \\) iterations and  \\(r=e^{-2} \\)).

It's interesting to compare how sensitive L-BFGS and GD are to the condition number increase. Yes, we're fixing the eigenvalue pattern to be a linear spread, but let's save inspecting that to the end.

The linear rate is effectively the slope that the log plot has at the end. While we're at it, what's the effect of more memory?

```python
for memory, ls in [(10, '-'), (25, '--'), (50, ':')]:
    for kappa, color in [(10, 'r'), (25, 'b'), (50, 'g')]:
        s = np.linspace(1, kappa, d)
        lbs = lbfgs(s, memory)
        plt.semilogy(lbs, c=color, ls=ls,
                     label=r' \\(\kappa = {kappa}, m = {memory} \\)'.format(
                        memory=memory, kappa=kappa))

plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.xlabel('iterations')
plt.ylabel('function optimality gap')
plt.title(matrix_name)
plt.show('linspace eigenvals (L-BFGS)')
```

![png](/assets/2020-11-01-lbfgs-vs-gd_files/2020-11-01-lbfgs-vs-gd_5_0.png)

Now that's pretty cool, it looks like the limiting behavior is still ultimately linear (as expected, it takes about as many iterations as the memory size for the limiting behavior to "kick in"), but as memory increases, the rate gets better. What if we make the eigenvalues clumpy?

```python
for memory, ls in [(10, '-'), (25, '--'), (50, ':')]:
    for kappa, color in [(10, 'r'), (25, 'b'), (50, 'g')]:
        bot, mid, top = d // 3, d // 3, d - 2 * d // 3
        s = [1] * bot + [kappa / 2] * mid + [kappa] * top
        s = np.array(s)
        lbs = lbfgs(s, memory)
        plt.semilogy(lbs, c=color, ls=ls,
                     label=r' \\(\kappa = {kappa}, m = {memory} \\)'.format(
                        memory=memory, kappa=kappa))

plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.xlabel('iterations')
plt.ylabel('function optimality gap')
plt.title('tri-cluster eigenvals (L-BFGS)')
plt.show()

for memory, ls in [(10, '-'), (25, '--'), (50, ':')]:
    for kappa, color in [(10, 'r'), (25, 'b'), (50, 'g')]:
        s = np.logspace(0, np.log10(kappa), d)
        lbs = lbfgs(s, memory)
        plt.semilogy(lbs, c=color, ls=ls,
                     label=r' \\(\kappa = {kappa}, m = {memory} \\)'.format(
                        memory=memory, kappa=kappa))

plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.xlabel('iterations')
plt.ylabel('function optimality gap')
plt.title('logspace eigenvals (L-BFGS)')
plt.show()
```

![png](/assets/2020-11-01-lbfgs-vs-gd_files/2020-11-01-lbfgs-vs-gd_7_0.png)

![png](/assets/2020-11-01-lbfgs-vs-gd_files/2020-11-01-lbfgs-vs-gd_7_1.png)

Wow, done in 12 iterations for the clustered eigenvalues. It looks like the hardest spectrum for L-BFGS (and coincedentally the one with the cleanest descent curves) is evenly log-spaced spectra.

Let's try to map out the relationship between memory, kappa, and the linear convergence rate.

```python
from scipy.stats import linregress

kappa = 30
memory = list(range(5, 100 + 1, 5))
for kappa, color in [(10, 'r'), (30, 'b'), (50, 'g')]:
    rates = []
    for m in memory:
        s = np.logspace(0, np.log10(kappa), d)
        lbs = lbfgs(s, m)
        y = np.log(lbs)
        x = np.arange(len(lbs)) + 1
        slope, *_ = linregress(x, y)
        rates.append(np.exp(slope))
    plt.plot(memory, rates,
             c=color, label=r' \\(\kappa = {kappa} \\)'.format(kappa=kappa))

plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.xlabel('memory (dimension = {d})'.format(d=d))
plt.ylabel('linear convergence rate')
plt.title(r'logspace eigenvals, L-BFGS'
          .format(kappa=kappa))
plt.show()

kappas = list(range(5, 50 + 1, 5))
# interestingly, large memory becomes unstable
for memory, color in [(10, 'r'), (15, 'b'), (20, 'g')]:
    rates = []
    for kappa in kappas:
        s = np.logspace(0, np.log10(kappa), d)
        lbs = lbfgs(s, memory)
        y = np.log(lbs)
        x = np.arange(len(lbs)) + 1
        slope, *_ = linregress(x, y)
        rates.append(np.exp(slope))
    
    plt.plot(kappas, rates,
            c = color, label=r' \\(m = {memory} \\)'.format(memory=memory))
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.xlabel(r' \\(\kappa \\)')
plt.ylabel('linear convergence rate')
plt.title(r'logspace eigenvals, L-BFGS'
          .format(memory=memory))
plt.show()
```

![png](/assets/2020-11-01-lbfgs-vs-gd_files/2020-11-01-lbfgs-vs-gd_9_0.png)

![png](/assets/2020-11-01-lbfgs-vs-gd_files/2020-11-01-lbfgs-vs-gd_9_1.png)

OK, so the dependence in  \\(\kappa \\) still smells like  \\(1-\kappa^{-1} \\), but at least there's a very interesting linear trend between the linear convergence rate and memory (which does seem to bottom out for well-conditioned problems, but those don't matter so much). What's cool is that it's preserved across different  \\(\kappa \\).

To finish off, just out of curiosity, do any of the BFGS diagnostics tell us much about the convergence rate?

```python
kappa = 30
s = np.logspace(0, np.log10(kappa), d)

newtons, cosines = [], []
memories = [5, 50]

gds = gd(s, max_iter=len(lbs))
plt.semilogy(gds, c='b', label='GD')
for color, memory in zip(['r', 'g'], memories):
    lbs, newton, cosine = lbfgs(s, memory, extras=True)
    matrix_name = r'logspace eigenvals,  \\(\kappa = {kappa} \\)'.format(kappa=kappa)
    plt.semilogy(lbs, c=color,
                 label=r'L-BFGS ( \\(m = {memory} \\))'.format(memory=memory))
    newtons.append(newton)
    secants.append(secant)
    cosines.append(cosine)

plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.xlabel('iterations')
plt.ylabel('function optimality gap')
plt.title(matrix_name)
plt.show()

for newton, memory in zip(newtons, memories):
    newton = np.array(newton)
    index = np.arange(len(newton))
    q = np.percentile(newton, 95)
    plt.semilogy(index[newton < q], newton[newton < q],
                 label=r' \\(m = {memory} \\)'.format(memory=memory))
plt.xlabel('iterations')
plt.ylabel(r' \\(\\|(B\_k -\nabla\_k^2)\mathbf{p}\_k\\|\_2/\\|\mathbf{p}\_k\\|\_2 \\)')
plt.title('Directional Newton Approximation')
plt.show()

for cosine, memory in zip(cosines, memories):
    cosine = np.array(cosine)
    plt.plot(cosine,
                 label=r' \\(m = {memory} \\)'.format(memory=memory))
plt.xlabel('iterations')
plt.ylabel(r' \\(\cos \theta\_k \\)')
plt.title('L-BFGS and GD Cosine')
plt.show()
```

![png](/assets/2020-11-01-lbfgs-vs-gd_files/2020-11-01-lbfgs-vs-gd_11_0.png)

![png](/assets/2020-11-01-lbfgs-vs-gd_files/2020-11-01-lbfgs-vs-gd_11_1.png)

![png](/assets/2020-11-01-lbfgs-vs-gd_files/2020-11-01-lbfgs-vs-gd_11_2.png)

So, as we can see above, it's not quite right to look to either  \\(\cos\theta\_k \\) nor  \\(\\|(B\_k-\nabla^2\_k)\mathbf{p}\_k\\|/\\|\mathbf{p}\_k\\| \\) to demonstrate L-BFGS convergence (the latter should tend to zero per BFGS theory as memory tends to infinity). But at least for quadratic functions, perhaps it's possible to work out the linear rate acceleration observed earlier via some matrix algebra.


[Try the notebook out yourself.](/assets/2020-11-01-lbfgs-vs-gd.ipynb)

