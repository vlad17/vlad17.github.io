---
layout: post
title:  "Deep Learning Learning"
date:   2017-07-09
categories: deep-learning
---

# Deep Learning Learning Plan

This is my plan to on-board myself with recent deep learning practice (as of the publishing date of this post). Comments and recommendations [via GitHub issues](https://github.com/vlad17/vlad17.github.io/issues) are welcome and appreciated! This plan presumes some probability, linear algebra, and machine learning theory already, but if you're following along [Part 1 of the Deep Learning book](http://www.deeplearningbook.org/) gives an overview of topics to cover.

1. Intro tutorials/posts.
    * [Karpathy](http://karpathy.github.io/neuralnets/)
    * Skim lectures from weeks 1-6, 9-10 of [Hinton's Coursera course](https://www.coursera.org/learn/neural-networks)
1. Scalar supervised learning theory
  * Read Chapters 6, 7, 8, 9, 11, 12 of [Dr. Goodfellow's Deep Learning Book](http://www.deeplearningbook.org/) and [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    * Keep compacted notes for yourself later [(or pool them with mine!)](https://github.com/vlad17/ml-notes)
1. Scalar supervised learning practice
    * Choose an enviornment.
        * Should be TensorFlow-based, given the wealth of ecosystem around it; stuff like [Sonnet](https://github.com/deepmind/sonnet) and [T2T](https://github.com/tensorflow/tensor2tensor).
        * Among [TF-Slim](https://github.com/tensorflow/models/blob/master/inception/inception/slim/README.md), [TFLearn](http://tflearn.org/), [Pretty Tensor](https://github.com/google/prettytensor), [Keras](https://keras.io/), and [TensorLayer](https://github.com/zsdonghao/tensorlayer), TensorLayer is my choice, providing a transparent interface to TF, integration with other frameworks, pre-implemented techniques from research (not as much as Keras, though). See [this TensorLayer how-to](https://github.com/wagamamaz/tensorlayer-tricks).
        * Most tutorials use Theano or Keras, so translating to TensorLayer is a good extra exercise. Additionally, try to extend with techniques from the theory, above.
    * Lessons 0-5 from [USF](http://course.fast.ai/index.html)
    * [Stanford CS20S1](http://web.stanford.edu/class/cs20si/syllabus.html)
    * Tutorials 1-3 from [deeplearning.net](http://deeplearning.net/tutorial/)
    * Notebooks 00-05, 09, 10 from [nlintz/TensorFlow-Tutorials](https://github.com/nlintz/TensorFlow-Tutorials).
    * Lab 1 from [MIT 6.S191](https://github.com/yala/introdeeplearning)
    * [Stanford CS231n](http://cs231n.github.io/)
    * Replicate [ResNet by He et al 2015](https://arxiv.org/abs/1512.03385)
    * Do an end-to-end application from scratch. E.g., convert an equation image to LaTeX.
1. Sequence supervised learning
    * Gentle introductions
        * Lessons 6-7 from [USF](http://course.fast.ai/index.html)
        * [Karpathy RNN post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
        * Weeks 7-8 of [Hinton's Coursera course](https://www.coursera.org/learn/neural-networks)
    * Theory
        * Chapter 10 from [Goodfellow](http://www.deeplearningbook.org/)
    * Practice
        * LSTM and RNN tutorials from [deeplearning.net](http://deeplearning.net/tutorial/)
        * Notebook 07 from [nlintz/TensorFlow-Tutorials](https://github.com/nlintz/TensorFlow-Tutorials).
        * Lab 2 from [MIT 6.S191](https://github.com/yala/introdeeplearning)
        * End-to-end application from scratch: a Swype keyboard ([Reddit tips](https://www.reddit.com/r/MachineLearning/comments/5ogbd5/d_training_lstms_in_practice_tips_and_tricks/))
    * Paper recreations
        * Machine translation [Sutskever et al 2014](https://arxiv.org/abs/1409.3215)
        * NLP [Vinyals et al 2015](https://arxiv.org/abs/1412.7449)
        * Dense captioning [Karpathy 2016](http://cs.stanford.edu/people/karpathy/densecap/)
1. Unsupervised and semi-supervised approaches
    * Theory
        * Weeks 11-16 of [Hinton's Coursera course](https://www.coursera.org/learn/neural-networks)
        * Chapters 13, 16-20 from [Goodfellow](http://www.deeplearningbook.org/)
        * See also my links for [VAE and RBM notes here](https://github.com/vlad17/ml-notes/tree/master/deep-learning)
    * Practice
        * Remaining [deeplearning.net](http://deeplearning.net/tutorial/) tutorials, based on interest.
        * Notebooks 06, 11 from [nlintz/TensorFlow-Tutorials](https://github.com/nlintz/TensorFlow-Tutorials).
    * Paper recreations
        * [WGAN](https://arxiv.org/abs/1701.07875)
        * [VAE](https://arxiv.org/abs/1312.6114)
        * [IAF VAE](https://arxiv.org/abs/1606.04934)

[//]: # (% LocalWords: TF nlintz deeplearning Coursera Reddit )
        
