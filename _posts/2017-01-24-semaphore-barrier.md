---
layout: post
title:  "The Semaphore Barrier"
date:   2017-01-24
categories: interview-question parallel
---

# The Semaphore Barrier

I wanted to share an interview question I came up with. The idea came from my operating and distributed systems classes, where we were expected to implement synchronization primitives and reason about parallelism, respectively.

Synchronization primitives can be used to coordinate across multiple threads working on a task in parallel.

Most primitives can be implemented through the use of a condition variable and lock, but I was wondering about implementing other primitives in terms of semaphores.

## Introduction to the Primitives

### Semaphores

Semaphores are a type of synchronization primitive that encapsulate the idea of "thresholding".

A semaphore `s` has two operations: `s.up()` and `s.down()`. A semaphore also has an internal non-negative number representing its state. A thread calling `s.down()` is allowed to continue only if this number is positive, in which case the number is atomically decremented and the thread goes on with its work.

`s.up()` doesn't guarantee any blocking either, and raises the number.

If we wanted to make sure that only 5 threads executing `f()` (a function that we implement) ever printed `hello`, and we somehow pre-emptively set the state of semaphore `s` to `5`, then the following code would work:

    def f():
        s.down()
        print("hello\n", end='')

Regardless of how many threads call `f` at the same time, because of the atomic guarantees on `s`'s state, only 5 threads will be let through to print `hello`.

If at any time in the above we had at least 6 threads call `f` and any thread also call `s.up()` at some point, eventually 6 `hello`s would be printed.

In the following, we'll assume the OS provides a magic semaphore implementation.

### Barriers

A barrier is similar to a semaphore, but it's meant to be a one-off, well, barrier. A barrier is preconfigured to accept `n` threads. Its API is defined by `b.wait()`, where a thread waits until `n-1` other threads are *also* waiting on `b`, an only then are the threads allowed to continue.

Barriers are useful when we want to coordinate some work. Suppose we have 2 threads, who want to draw a picture together. Say thread 0 can only draw red and thread 1 can only draw blue. But no two threads can draw on the same half of the screen at the same time.

Assuming `b` has been initialized with `n=2`, the following would work:

| thread 0 | | thread 1 |
| --- | --- | --- |
| draw left half | | draw right half |
| `b.wait()` | | `b.wait()` |
| draw right half  | |  draw left half |

Now, no matter which thread is faster, we'll never violate the condition that 2 threads write on the same half of the screen.

The only way thread 0 can be on the left half is if it hasn't crossed the barrier `wait` yet. The only way thread 1 can be on the left half is if it crossed the barrier, but since `b=2`, it can only cross the barrier if thread `1` is waiting, in which case it must have finished drawing on the left half!

Similar logic can be applied to the right side; in other words, no side of the screen is ever shared by two threads at any given time, regardless of how fast one thread is compared to the other.

## The Challenge

### Warm-up

Our goal will be to implement a barrier (namely, fill in what `b.wait()` does for a given `n`). Let's focus on the case where we only have `n=2` threads.

This can be done with two semaphores.

### Solution to the Warm-up

As you may have guessed, the only nontrivial semaphore arrangement works. From here on, we let `s(i)` be the `i`-th semaphore, initialized with state 0. Similarly, \\(t_i\\) will refer to the \\(i\\)-th thread. Here's what we would want `b.wait()` to do on each thread.

| t0 | | t1 |
| --- | --- | --- |
| `s(0).up` | | `s(1).up` |
| `s(1).down` | | `s(0).down` |


Indeed - \\(t_0\\) can't advance past `b.wait()` unless `s(1)` is `up`ped, which only happens if \\(t_1\\) call `b.wait()`. Symmetric logic shows that our barrier, if implemented to execute those instructions on each thread, will similarly stop `s(2)` from advancing without `s(1)` being ready.

### The General Problem

Now, here's the main question:

**Can we implement an arbitrary barrier, capable of blocking `n` threads, with semaphores and no control flow? With control flow?** 

Now, can we do so _efficiently_, using as few semaphores as possible? In as little time per thread as possible?

#### Attempt: Extending the 2-thread Case

Let's try extending our approach from the 2-thread case. Maybe we can just use 3 semaphores now, but using the "cycle" that seems to be built in the 2-thread example?

| t0 | | t1 | |  t2 |
| --- | --- | --- | --- | --- |
| `s(0).up` | | `s(1).up` | | `s(2).up` |
| `s(1).down` | | `s(2).down` | | `s(0).down` |

But this won't work, unfortunately: suppose \\(t_0\\) is running slow. Both \\(t_1\\) and \\(t_2\\) finish well ahead of time, and each calls `b.wait()`. Then \\(t_2\\) ups `s(2)`, after which \\(t_1\\) can pass through without waiting for \\(t_0\\) to call `b.wait()`, a violation of our barrier behavior.

#### Answer

Not so fast! Try it yourself! How efficient is your solution? There's a couple of them, in increasing order of difficulty. The following list describes the asymptotic space complexity (number of semaphores used) and time complexity (**per thread**).

0. \\(O(n^2)\\) space and \\(O(n)\\) time
0. \\(O(n)\\) space and \\(O(n)\\) time
0. \\(O(n)\\) space and \\(O(1)\\) average time, \\(O(n)\\) worst-case time
0. \\(O(n)\\) space and \\(O(1)\\) worst-case time
0. \\(O(1)\\) space and \\(O(1)\\) worst-case time

[Link to answer]({{ site.baseurl }}{% post_url 2017-01-25-semaphore-answer %})

### A Note on Thread IDs

The fact that we can write different code for each of the threads to execute in the above examples might seem a bit questionable. However, we can get around this by assuming that we have access to thread IDs. As long as we can procure a thread's procedure given just its ID (and the function procuring such a procedure doesn't take \\(O(n^2)\\) space), we should be fine.

Even if the thread ID isn't available, we can use an atomic counter, which assigns effective thread IDs based on which thread called `b.wait()` first:

```
atomic = AtomicInteger(0)

def wait():
    tid = atomic.increment_and_get()
    ...
```
