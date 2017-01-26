---
layout: post
title:  "The Semaphore Barrier (Solution)"
date:   2017-01-25
categories: interview-question parallel
---

# The Semaphore Barrier

This is the answer post to the question [posed here]({{ site.baseurl }}{% post_url 2017-01-24-semaphore-barrier %}).

## A Useful Formalism

Reasoning about parallel systems is tough, so to make sure that our solution is correct we'll have to introduce a formalism for parallel execution.

The notion is the following. Given some instructions for threads \\(\\{t\_i\\}\_{i=0}^{n-1}\\), we expect each thread's individual instructions to execute in sequence, but instructions between threads can be interleaved arbitrarily. 

In our simplified execution model without recursive functions, it suffices to assume each thread has a fixed set of instructions it will execute. Let this be the sequence \\(t_i\\), with \\(k\\)-th instruction \\(t_{ik}\\), which must be `s(j).up` or `s(j).down` for some `j`.

### Order of Execution

Our parallel machine is free to choose a global order of operations \\(g\\) amongst all threads \\(\\{t\_i\\}\_{i}\\), where each \\(g_j=t_{ik}\\) for all \\(j\\) and some corresponding \\(i,k\\). However, the machine has to choose an ordering that is _valid_.

A valid ordering \\(g\\) satisfies two criteria.

The _sequencing constraint_ is as follows:

\\[
k<m \implies t\_{ik} <\_g t\_{im}
\\]

Above, we define an ordering over operations \\(x <\_g y\\) with respect to some ordering in the natural way: in \\(g\\), \\(x\\) comes before \\(y\\). If a statement holds for all (valid) \\(g\\), we omit the subscript: the conclusion of the sequencing constraint can be re-written \\(t\_{ik} < t\_{im}\\).

In addition, for every global ordering of operations \\(g\\), there's a corresponding sequence \\(s\\) (which differs from the un-italicized `s(i)`, the code for the `i`-th semaphore). The \\(j\\)-th element in the sequence \\(s\\) is the state of each semaphore after the \\(j\\)-th instruction \\(g\_j\\). We represent this state as a function from semaphore index to semaphore state. Letting \\(s_0=(i\mapsto 0)\\):

\\[
s_{j}(i)=s_{j-1}(i)+\begin{cases}
1 & g_j=\text{s(i).up}\\\\  - 1 & g_j=\text{s(i).down}\\\\ 0 & \text{otherwise}\\\\
\end{cases}
\\]

The above just says that after the `i`-th semaphore is upped, its value should be 1 more than before, and vice-versa for down.

The _semaphore constraint_ requires that the global order \\(g\\) is chosen such that:
\\[
\forall i,j\,\,\,\,\, s_j(i)\ge 0
\\]
Here, this constraint just makes sure that semaphores actually work as expected - it can't be that a `down` call succeeds on a semaphore that had state 0 - it should wait until a corresponding `up` call completes, first.

### Solution Criteria

A solution (which defines the particular values \\(\\{t\_i\\}\_{i}\\)) must satisfy two criteria.

(_Correctness_): No thread can finish `b.wait()` before all threads have called the method:
\\[
\forall i,j,\,\, t_{j1}<t_{i\left\vert t_i\right\vert}
\\]

(_Liveness_): Eventually, every thread must complete `b.wait()`. There must exist at least one valid ordering \\(g\\) (if there is only one, the parallel processing system is forced to choose it).

### Example

Let's apply the formalism to the warmup solution for two threads:

| t0 | | t1 |
| --- | --- | --- |
| `s(0).up` | | `s(1).up` |
| `s(1).down` | | `s(0).down` |

All the potential orderings respecting sequencing are:

```
0. s(0).up, s(1).up, s(1).down, s(0).down
1. s(0).up, s(1).up, s(0).down, s(1).down
2. s(1).up, s(0).up, s(1).down, s(0).down
3. s(1).up, s(0).up, s(0).down, s(1).down
4. s(0).up, s(1).down, s(1).up, s(0).down
5. s(1).up, s(0).down, s(0).up, s(1).down
```

Of these, we notice `4` and `5` violate the semaphore constraint. For `4`, the state function after the second step \\(s_2(0)=1, s_2(1)=-1\\), and vice-versa for `5`.

That leaves only `0,1,2,3` as the valid orderings. In turn, we satisfy liveness. Correctness is guaranteed by inspection: the last operations are only executed after the first ones.

## Solution 1

\\(O(n^2)\\) space and \\(O(n)\\) time.

This solution follows directly from reasoning about our formalism. Suppose `s(i)` was upped only once. For any \\(g\\) to be valid (no negative values), we must only down it once as well. Moreover, any down is guaranteed to occur after the up, again by the non-negativity requirement.

This could be proven formally - every state starts at 0, so if no ups occur before a down, by induction, the state of that semaphore is 0 right before the down and -1 after. This leads to a contradiction.

Suppose \\(t_{ik}\\) is `s(ij).up` and \\(t_{jm}\\) is `s(ij).down`. If we never use `s(ij)` again, the lemma above holds, in which case for every ordering \\(t_{ik} < t_{jm}\\). For any sequences \\(t_i,t_j\\), we must have \\(k\in[1, \left\vert t_i\right\vert],m\in[1, \left\vert t_j\right\vert]\\). Then by transitivity we conclude:
\\[
t_{i1}\le t_{ik} < t_{jm} \le t_{j\left\vert t_j\right\vert}
\\]

Thus, the presence of `s(ij).up` on thread `i` and `s(ij).down` on `j` guarantees correctness, if applied to all threads `i,j`. To guarantee some ordering exists, we will want to ignore the redundant case `s(ii)` and sequence our operations in a clear way:

```
def wait(thread i):
    for all j != i:
        s(ij).up
    for all j != i:
        s(ji).down
```

This solution is live: an order where all ups get executed in some order, then all downs do exists and is valid.

With 3 threads, this looks like:

| t0 | | t1 | |  t2 |
| --- | --- | --- | --- | --- |
| `s(01).up` | | `s(12).up` | | `s(20).up` |
| `s(02).up` | | `s(10).up` | | `s(21).up` |
| `s(10).down` | | `s(21).down` | | `s(02).down` |
| `s(20).down` | | `s(01).down` | | `s(12).down` |

In other words, if we represent each pairwise constraint \\(\forall i,j,T\triangleq\left\vert t_i\right\vert, t_{j1}<t_{iT}\\) explicitly, we get a solution.

## Solution 2

\\(O(n)\\) space and \\(O(n)\\) time.

This solution can be constructed by augmenting our lemma from before: for any \\(g\\) to be valid (no negative values), any semaphore must be upped more times than it has been downed right before every down.

Then, if a single thread is responsible for upping its own semaphore, and all other threads down it exactly once, _at least one_ up must've occured before each of the downs. This lets us recover the transitive inequality from before for correctness.

In other words, the following works:

| t0 | | t1 | |  t2 |
| --- | --- | --- | --- | --- |
| `s(0).up` | | `s(1).up` | | `s(2).up` |
| `s(0).up` | | `s(1).up` | | `s(2).up` |
| `s(1).down` | | `s(2).down` | | `s(0).down` |
| `s(2).down` | | `s(0).down` | | `s(1).down` |

With the same liveness argument, more generally the psuedocode is:

```
def wait(thread i):
    do n-1 times:
        s(i).up
    for all j != i:
        s(j).down
```

## Solution 3

\\(O(n)\\) space and \\(O(1)\\) average time, \\(O(n)\\) worst-case time

Now we need to start getting a little bit more clever. Previous solutions still performed a quadratic amount of work total, establishing the quadratic number of inequalities needed for correctness.

The goal here will be to get transitivity to do some of our heavy lifting.

```
def wait(thread i):
    // (1)
    if i < n-1:
        s(i).down
        s(i + 1).up
    else:
        s(0).up
        s(n-1).down
    
    // (2)
    if i < n-1:
        s(n).down
    else:
        do n-1 times:
            s(n).up
```

By the reasoning from before, block (2) guarantees that \\(t_{(n-1)k} < t_{j1}\\) for some \\(j\neq n-1\\) and some \\(k\in[3,n+2]\\). Then by sequential validity of our orderings and transitivity we have a global property saying:
\\[
\forall j, t_{(n-1)3}<t_{j\left\vert t_j\right\vert}
\\]

In other words, all threads wait on thread 3 (eq. 1).

Next, we focus on block (1). We apply the lemma from solution 1 for each \\(i\\) between \\(2\\) and \\(n-2\\), which, by virtue of `s(i)` only being used once, says that the `s(i).down` instruction on thread \\(i\\) follows the `s(j + 1).up` one on thread \\(j\\), where \\(j = i - 1\\). For \\(j<n-2\\), this statement is \\(t\_{j2}<t\_{i1}\\). Next, by the sequence property, we have \\(\forall j,t_{j1}<t_{j2}\\). Finally, chaining all these inequalities together, we get for \\(j<n-1\\) (eq.2):

\\[
t\_{j1}\le t\_{(n-2)1}< t\_{(n-2)2}
\\]

We use the lemma from solution 1 once on the semaphore `s(n-1)`, upped exactly at \\(t_{(n-2)2}\\) and downed on \\(t\_{(n-1)2}\\). In turn, we have (eq. 3):

\\[
t\_{(n-2)2} < t\_{(n-1)2}< t\_{(n-1)3}
\\]

Let's recap. All threads already wait on thread \\(n-1\\). We just need to check that all threads also wait on all threads \\(i\\) between \\(1\\) and \\(n-2\\). For all \\(i,j\\):

\\[
\begin{align} t\_{i1} &< t\_{(n-2)2} & \text{eq. 2}\\\\ &<t\_{(n-1)3} &\text{eq. 3} \\\\ &<t\_{j\left\vert t_j\right\vert} &\text{eq. 1} \\\\ \end{align}
\\]

This finishes the correctness proof. We show liveness exists by providing the ordering \\(t\_{(n-1)1}\\) followed by \\(t\_{i1}, t\_{i2}\\) for all \\(i\\) in order up to \\(n-1\\). Then we let \\(t_{n-1}\\) finish and after that order doesn't matter.

Here's what this looks like on 5 threads:

| t0 | | t1 | |  t2 | | t3 | |  t4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `s(0).down` | | `s(1).down` | | `s(2).down` | | `s(3).down` | | `s(0).up` |
| `s(1).up` | | `s(2).up` | | `s(3).up` | | `s(4).up` | | `s(4).down` |
| `s(5).down` | | `s(5).down` | | `s(5).down` | | `s(5).down` | | `s(5).up` |
|  | |  | |  | |  | | `s(5).up` |
|  | |  | |  | |  | | `s(5).up` |
|  | |  | |  | |  | | `s(5).up` |


## Solution 4

\\(O(n)\\) space and \\(O(1)\\) worst-case time

```
def wait(thread i):
    // (1)
    if i < n-1:
        s(i).down
        s(i + 1).up
    else:
        s(0).up
        s(n-1).down
    
    // (2)
    if i > 0:
        s(i).down
        s(i - 1).up
    else:
        s(n-1).up
        s(0).down
```

The proof is left as an exercise to the reader :)
