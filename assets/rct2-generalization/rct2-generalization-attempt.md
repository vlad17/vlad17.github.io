A First Attempt

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
