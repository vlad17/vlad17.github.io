---
layout: post
title:  "Gemini Flash Pretraining"
date:   2025-04-24
categories: llm pretraining
featured_image: /assets/2025-04-24-gemini-talk.png
meta_keywords: LLM, large language model, gemini, pretraining, distillation
---

# Gemini Flash Pretraining

Not too long ago, I gave a public talk on Gemini Pretraining to [COS 568 by Prof. Kai Li](https://www.cs.princeton.edu/courses/archive/spring25/cos568/), Systems and Machine Learning.

In the talk, I covered what I thought would be an interesting modelling perspective for ML systems students. I mostly go through:

1. public academic papers on scaling laws
2. how scaling approaches might need to be modified in the face of inference constraints

This is a literature review and discussion of relevant external work, but I figured the collation itself, as well as commentary from an industry POV, might be pretty useful.

For the most part, (1) goes over the historical discussion of how we came to understand scaling laws, from the horses' mouths themselves (so these are explicitly Sebastian Borgeaud's and Jean-Baptiste Alayrac's slides) and (2) reviews a lot of relevant works I thought about when applying this to the Flash setting, which itself touches on quite a bit of excellent work Jacob Austin posted externally.

Seb's talk in particular is a great resource on the first half, [it's on Youtube](https://www.youtube.com/watch?v=1MqlbPsWnAA).

Please see the final slide for the original papers and references from which the presentation draws so heavily!

## Link to Slides as PDF

Slides are available [here](/assets/2025-04-24-princeton-talk.pdf).

The embeded videos are:

1. [Project Astra](https://www.youtube.com/watch?v=hIIlJt8JERI) (video courtesy of Tara Sainath)
2. [Project Mariner](https://www.youtube.com/watch?v=_uBg6syzXhk) (video courtesy of Anmol Gulati)

## Future Research Opportunities for Academia

A question I get often, both in this context and in general, is what kind of research academic labs could do in this area, given how expensive pretraining is.

I think there's actually quite a lot that could be contributed here.

I excerpted my slide on the matter inline below.

![future research](/assets/2025-04-24-future-research.png){: .center-image }

Let me expand on these here.

* Quant and kernel development are self-evident. They don't require actual extended training, but demand a lot of creative thinking to identify mathematical invariants.
* The [Funsearch](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/) direction is a nice little nugget. Funsearch used LLMs to generate candidate programs in a setting where they could be evaluated quantitatively for reaching some objective (think: defining heuristics based for a combinatorial problem like travelling salesman to minimize travel time), and applied genetic programming on top do search over such heuristics. You wouldn't know it from the paper/its appendices, but what happened is that the Funsearch team tried to use larger and smaller models in the middle of the loop; they had best results with a mid-sized candidate (that I trained with Emanuel Taropa and Rohan Anil). I always found this to be an interesting tidbit: in generative search you need to strike the right balance of proposal frequency with evaluation. Formalize. Maybe even apply it to the verified RL setting.
* Finally, one piece missing from all scaling law discussion, which is indeed pure theory (and maybe small scale validation work) is a statistical framework for law fits. Each \\((N, D)\\) point is expensive to observe, and least squares vs MLE fits for laws imply different prescriptions. Moreover, a framework for discussing the noise in LLM evaluations would equip us with more efficient proposals for how to fit scaling laws---rather than a grid over data/param size, select points iteratively by expected information gain.
