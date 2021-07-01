---
layout: post
title:  "The Triple Staple"
date:   2019-11-30
categories: tools joke-post
meta_keywords: staple
---

# The Triple Staple

When reading, I prefer paper to electronic media. Unfortunately, a lot of my reading involves manuscripts from 8 to 100 pages in length, with the original document being an electronic PDF.

Double-sided printing works really well to resolve this issue partway. It lets me convert PDFs into paper documents, which I can focus on. This works great up to 15 pages. I print the page out and staple it. I've tried not-stapling the printed pages before, but then the individual papers frequently get out of order or generally all over the place.

**However**, for larger manuscripts I frequently found myself in a pickle:

* I don't want to manage loose leaf pages individually.
* Staplers that can handle stapling over 15 pages don't occur naturally, at least near the printers I'm around.

Attempting to use a stapler beyond its capacity does not end successfully.

![weak staplers](/assets/2019-11-30-the-triple-staple/the-problem.png){: .center-image }

For a good deal of my life I've resigned myself to dealing with a reality of mediocre staplers and even more mediocre workarounds, e.g., a packet on a single topic now needs be represented by 3 independent, separately-stapled documents, which is 2 too many.

I'm confident many others also have this problem. To wit, I'd like to introduce a life hack, for all situations where you have documents of up to \\(2X\\) pages and staplers with penetration power rated at \\(X\\) pages.

## The Problem

I want to staple this thick paper stack.

![initial conditions](/assets/2019-11-30-the-triple-staple/initial-conditions.png){: .center-image }

_Optimality criteria_.

(A) Grip strength of resulting staple.

(B) Non-obstruction of reading material.

## Solution

1. Staple pages \\(1\\) to \\(X\\).
2. Staple pages \\(X+1\\) to \\(2X\\).
3. Peel back the corner of pages \\(1\\) to \\(\lfloor X/2\rfloor\\) over the staple. Repeat for \\(\lfloor 3X/2\rfloor\\) to \\(2X\\)
4. Insert the exposed corner of pages \\(\lfloor X/2\rfloor +1\\) to \\(\lfloor 3X/2\rfloor - 1\\) into the stapler, making sure the folded-away corners of the outer pages are out of the stapler's line of fire.
5. Apply the stapler to the middle pages, then fold the outer pages' corners back up.

## Results

Step 1 and 2.

![step 1 and 2](/assets/2019-11-30-the-triple-staple/step-one.png){: .center-image }

Step 3.

![step 3](/assets/2019-11-30-the-triple-staple/step-three.png){: .center-image }

Step 4.

![step 4](/assets/2019-11-30-the-triple-staple/step-four.png){: .center-image }

Step 5.

![step 5](/assets/2019-11-30-the-triple-staple/step-five.png){: .center-image }

Additional results (skew angle, front, and back views).

![step 5 1](/assets/2019-11-30-the-triple-staple/step-five1.png){: .center-image }

![step 5 2](/assets/2019-11-30-the-triple-staple/step-five3.png){: .center-image }

![step 5 3](/assets/2019-11-30-the-triple-staple/step-five2.png){: .center-image }

## Discussion and Related Work

(A) is met due to each staple holding together at least \\(X\\) pages. Contrast this with related work which only staples two pages \\(X,X+1\\) with an intermediate staple, resulting in a single point of failure at page \\(X\\).

(B) UX is equivalent to a single-stapled page, as opposed to binder-clip methodology which frequently requires clipping past the margin.

## Future Work

There exists a straightforward alternating iteration of our method that can be shown, by induction to apply to documents of length up to \\(n X\\) for any \\(n\in\mathbb\{N\}\\). We leave evaluation to future work.
