---
layout: post
title:  "Duplicate Finding"
date:   2018-11-03
categories: interview-question
---

# Duplicate Finding

I've been getting pretty good signal from a CS-fundamentals-type interview question lately. It's got quite a lot of solutions, and it's pretty hard, while still admitting decent naive solutions, so it scales well with a variety of candidates.

I'll leave the merits of brain-teaser--though I think that's a mis-characterization--interview questions to HN commenters or another post.

## The Question

You're given a mutable array of \\(n\\) integers. Each one is valued in the range \\([0,n-2]\\), so there must be at least one duplicate. Return a duplicate.

## The Basics

What's nice here is we can build up a memory/runtime frontier even with some crude asymptotic analysis. Let's work on an idealized machine where we can process pointers in constant time, but bits still aren't free (i.e., we can't treat the integers as being arbitrarily wide).

One immediate solution is "bubble". Let the input array be `xs`

    for x in xs:
      for y in (xs skipping x):
        if x == y:
          return y

Nice, \\(O(n^2)\\) runtime \\(O(1)\\) extra space overhead down. Let's get less naive. We might notice bubble linearly compares equality with every element; but a set can compare equality with multiple elements in constant time.

    vals = set()
    for x in xs:
      if x in vals:
        return x
      vals.add(x)

Great, for a hashset we now have \\(O(n)\\) extra space overhead for a linear-time algorithm. One might ask is this "linear time" worst-case or average? A hash set with a doubling strategy that picks prime sizes and an identity hash function would actually avoid the worst-case separate chaining performance bad hash functions induce. But if we're being pedantic, and we are, you'd need to build-in an algorithm to get arbitrarily large primes during resizes. Luckily we can avoid all this nonsense by switching the `set` to a `bitset` above for worst-case linear performance.

One might be tempted to encode the `bitset` in the first bit (the sign bit) of the `xs` array, but this would still incur linear overhead according to our assumptions. Indeed, these assumptions are somewhat realistic (say `n` contains values at least up to \\(2^{31}\\) and we use 4-byte integers).

Now we might ask if there's a nicer trade-off giving up runtime to save on memory use than going from the hash to the bubble approach. Indeed, we know another data structure that enables equality checks on multiple numbers in sublinear time: the sorted array!

    xs.sort()
    for x, y in zip(xs, xs[1:]):
      if x == y:
        return x

Now, what sort would we use (from an interview perspective, it's less valuable to see how many sorts do you know, but do you know a sort really well, and what is it doing)? What does your chosen language do? If you're using python, then you get points for knowing Timsort is a combination of out-of-place mergesort and quicksort; inheriting worst-case \\(O(n)\\) space and \\(O(n\log n)\\) time from the former.

Did you use Java? Props to you if you know JDK8 would be applying dual-pivot quick sort on the primitive type array (bonus bonus: timsort on non-primitives).

If you choose quicksort, then you should be aware of its worst-case performance and how to mitigate it. One mitigation would be as in C++, to use a heapsort cutoff ("introsort").

Of course, this is all very extra, but someone who's aware of what's going on at all layers of the stack is very valuable.

## The Advanced

Now here's where I tell you there are at least three different ways to achieve \\(O(n)\\) run time and \\(O(1)\\) extra memory overhead for this problem.

I won't reveal them here.

They are all meaningfully different, and each one has a different runtime profile. Two of the solutions use \\(\sim 2n\\) array accesses or sets in the worst case, and one of those has the bonus of being cache-friendly.

[//]: # quickselect-style, cuckoo-style graph following, and radix sort
