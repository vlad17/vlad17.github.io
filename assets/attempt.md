## A First Attempt

Let $\mathbf{x}^\top=\begin{pmatrix}\mathbf{x}_0&\cdots&\mathbf{x}_n\end{pmatrix}$ for conformal $\mathbf{x}_i$ of sizes $s_0,s,\cdots,s$. Inspecting the $i$-th block of $\mathbf{x}=\mathbf{1}+ P\mathbf{x}$ yields for $1<i<n$ that
$$
\mathbf{x}_{i}=\mathbf{1}_s+U\mathbf{x}_{i-1}+C\mathbf{x}_{i}+V^\top \mathbf{x}_{i+1}\,\,,
$$
which can be rewritten as
$$
\mathbf{x}_{i}=(I-C)^{-1}(\mathbf{1}_s+U\mathbf{x}_{i-1}+V^\top \mathbf{x}_{i+1})\,\,,
$$
since $\rho(C)<1$ is implied by $\rho(P)<1$, as $C$ is the restriction of $P$ to the subspace for the $i$-th block, so $I-C$ is also nonsingular.

By itself, expressing the $i$-th block in terms of the $(i-1)$-st and $(i+1)$-st isn't too helpful: such a recurrence doesn't bottom out (at least not obviously). Naively, if $U$ was somehow nonsingular then we could express $\mathbf{x}_{i-1}$ in terms of $\mathbf{x}_i,\mathbf{x}_{i+1}$, which does clearly terminate.

## A Second Attempt

Writing $U=L\Sigma R^\top$ as the full SVD for unitary $s\times s$ matrices $L, R$, and $\Sigma=\begin{pmatrix}
I_r & 0\\
0 & 0_{s-r}
\end{pmatrix}$ for the rank $r$ of $U$, let's multiply each block of the equation $(I-P)\mathbf{x}=\mathbf{1}$ by $L^\top$ on the left and $R$ on the right. Note we don't need to treat the edge case $r=0$ differently. Letting $\mathbf{y}_i= R^\top \mathbf{x}_i$ for every block $n\ge i >1$ we are left with for every $i$,
$$
\begin{gather*}
&-L^\top  L\Sigma R^\top R\mathbf{y}_{i-1}+
L^\top(I-C)R\mathbf{y}_i-
L^\top V^\top R\mathbf{y}_{i+1}\\
=&\begin{pmatrix}
-I_r & 0\\
0 & 0
\end{pmatrix}\mathbf{y}_{i-1} + A\mathbf{y}_i+B\mathbf{y}_{i+1}\\
=&L^\top\mathbf{1}\,\,.
\end{gather*}
$$
Above, $A,B$ are suitably defined, where we note that since $L,R$ are unitary, $A$ is nonsingular, and we let $\mathbf{y}_{n+1}=\mathbf{0}$ for the last equation. 

Now, to consider fully determined systems, we need to think about re-indexing. Our current block indexing is by $s_0,s,\cdots,s$:
$$
\mathbf{y}^\top=\begin{pmatrix}\mathbf{x}_0&\mathbf{y}_1&\cdots&\mathbf{y}_{n-1}&\mathbf{x}_n & \mathbf{0}_s\end{pmatrix}\,\,,
$$
but consider instead blocking by $s_0 + r, s, \cdots s$ by extending with more zeros so that
$$
(\mathbf{y}')^\top=\begin{pmatrix}\mathbf{y}_0'&\mathbf{y}_1'&\cdots&\mathbf{y}_{n-1}'&\mathbf{y}_n' & \mathbf{0}_s\end{pmatrix}\,\,,
$$
where $(\mathbf{y}_0')^\top =\mathbf{x}_0^\top|(\mathbf{y}_1)_{1:r}^\top$, $(\mathbf{y}_{n-1}')^\top =(\mathbf{y}_{n-1})_{r+1:s}^\top|(\mathbf{x}_n)_{1:r}^\top$, $(\mathbf{y}_{n}')^\top =(\mathbf{x}_{n})_{r+1:s}^\top|\mathbf{0}_r^\top$. All this hard work yields the shifted equations
$$
\begin{pmatrix}
A_{s-r,r} & A_{s-r}\\
 -I_r & 0
\end{pmatrix}\mathbf{y}_{i-1}' + \begin{pmatrix}
B_{s-r,r} & B_{s-r}\\
A_{r} & A_{r,s-r}
\end{pmatrix}\mathbf{y}_i'+\begin{pmatrix}
0 & 0\\
B_{r} & A_{r,s-r}
\end{pmatrix}\mathbf{y}_{i+1}'
=\begin{pmatrix}
(L^\top)_{s-r,s}\\(L^\top)_{r, s}
\end{pmatrix}\mathbf{1}\,\,,
$$
where we conformably break the $s\times s$ matrix $A=\begin{pmatrix}
A_r & A_{r,s-r}\\
A_{s-r,r} & A_{s-r}
\end{pmatrix}$, and similarly for others, e.g. $L^\top=\begin{pmatrix}
(L^\top)_{r, s}\\
(L^\top)_{s-r,s}
\end{pmatrix}$.

For all the marbles, we ask whether $\begin{pmatrix}
A_{s-r,r} & A_{s-r}\\
 -I_r & 0
\end{pmatrix}$ is invertible. I haven't figured this out yet (TODO). But, assuming it is, then we can solve the linear system above for $\mathbf{y}_{i-1}'=T(\mathbf{y}_i,\mathbf{y}_{i+1})$.

TODO make it square

$$
(\mathbf{y}_0,\mathbf{x}_1,\mathbf{x}_2)=T_0(\mathbf{x}_2,\mathbf{x}_3, \mathbf{x}_4,\mathbf{1}_s)=T_0T^{n-1}(\mathbf{x}_{n-1},\mathbf{x}_n,\mathbf{1}_s)\,\,,
$$

TODO exponentiate as above

Backsolve x


------------------


## A More Generic Solution

While the message-passing algorithm from the previous post works, it turns out it's also possible to use a much more generic technique for solving block-Toeplitz (not necessarily tridiagonal) systems: circulant completion.

A block circulant matrix $\mathrm{BC}(X)$ is defined by a three-dimensional array $X$ of shape $m\times s\times s$, yielding a matrix of size $ms\times ms$ which is both block-Toeplitz and the $i$-th and $(m+i)$-th blocks are identical (modulo $m$).

Following [De Mazancourt](https://ieeexplore.ieee.org/document/1143132), block circulant matrices have an explicit inverse. Namely, if $A=\mathrm{BC}(X)$ then $A^{-1}=\mathrm{BC}(\mathcal{F}^{-1}(\mathcal{F} X)^{-1})$, where $\mathcal{F}$ is the DFT along the first axis (i.e., we apply $s^2$ independent DFTs entrywise along $X$) but the inverse operation is applied to each matrix defined by the last two axes. This holds assuming $A$ is invertible.

For a block-Toeplitz $B$, $A=\begin{pmatrix}
B & F \\
E & B
\end{pmatrix}$ is an embedding into a block-circulant matrix of twice $B$'s size, with $E$, $F$ chosen with null diagonals and Toeplitz such that the resulting matrix is indeed Toeplitz. Then we simply pad the original system with zeros and enjoy a simple solution to block-Toeplitz systems!

import numpy as np

# rows[i] defines the (-i-1)-th block diagonal
# columns[i] defines the (i+1)-th block diagonal
def block_toeplitz(rows, center, cols):
    assert len(rows) == len(cols)
    res = []
    for i in range(len(rows) + 1):
        res.append(list(reversed(rows[:i])) + [center] + cols[:len(cols)-i])
    return np.block(res)

np.random.seed(1234)
r = [np.random.randint(10, size=(2, 2)), np.zeros((2, 2)), np.zeros((2, 2))]
c = [np.zeros((2, 2)), np.random.randint(10, size=(2, 2)), np.zeros((2, 2))]

%matplotlib inline
from matplotlib import pyplot as plt

T = block_toeplitz(
    r,
    10 * np.eye(2),
    c)
plt.imshow(T)
plt.axis('off')
plt.show()

def block_circulant(c):
    res = []
    for i in range(len(c)):
        res.append(c[len(c) - i:] + c[:len(c)-i])
    return np.block(res)

plt.imshow(block_circulant([np.eye(2), 2 * np.eye(2), 3 * np.eye(2)[[1, 0]]]), resample=False)
plt.axis('off')
plt.show()

def t2c(rows, center, cols):
    return [center] + cols + [np.zeros_like(center)] + list(reversed(rows))

C = block_circulant(t2c(
    r,
    10 * np.eye(2),
    c))
plt.imshow(C, resample=False)
plt.axis('off')
plt.show()

T = block_toeplitz(
    r,
    10 * np.eye(2),
    c)
C = block_circulant(t2c(
    r,
    10 * np.eye(2),
    c))

x = np.random.rand(T.shape[0])
s1 = np.linalg.solve(T, x)
s2 = np.linalg.solve(C, np.concatenate([x, np.zeros_like(x)]))
np.linalg.norm(s1 - s2[:len(x)])

# doesn't work, clearly b/c inverse can't be embedded
# same sol Tx = y
# when embedded Cx != [y, 0], Cx = [y, masked(T)x]
