---
layout: post
title:  "Deep Learning Learning"
date:   2017-07-09
categories: deep-learning
---

# Deep Learning Learning Plan

This is my plan to on-board myself with recent deep learning practice (as of the publishing date of this post). Comments and recommendations [via GitHub issues](https://github.com/vlad17/vlad17.github.io/issues) are welcome and appreciated! This plan presumes some probability, linear algebra, and machine learning theory already, but if you're following along [Part 1 of the Deep Learning book](http://www.deeplearningbook.org/) gives an overview of topics to cover.

My notes on these sources are [publicly available](https://github.com/vlad17/ml-notes), as are my [experiments](https://github.com/vlad17/learning-to-deep-learn).

1. Intro tutorials/posts.
    * [Karpathy](http://karpathy.github.io/neuralnets/)
    * Skim lectures from weeks 1-6, 9-10 of [Hinton's Coursera course](https://www.coursera.org/learn/neural-networks)
1. Scalar supervised learning theory
  * Read Chapters 6, 7, 8, 9, 11, 12 of [Dr. Goodfellow's Deep Learning Book](http://www.deeplearningbook.org/) and [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
1. Scalar supervised learning practice
    * Choose an enviornment.
        * Should be TensorFlow-based, given the wealth of ecosystem around it; stuff like [Sonnet](https://github.com/deepmind/sonnet) and [T2T](https://github.com/tensorflow/tensor2tensor).
        * I tried [TF-Slim](https://github.com/tensorflow/models/blob/master/inception/inception/slim/README.md) and and [TensorLayer](https://github.com/zsdonghao/tensorlayer), but I still found [Keras](https://keras.io/) easiest to rapidly prototype in (and expand). TensorFlow is still pretty easy to [drop down into](https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html) from the Keras models.
    * Google [MNIST](https://www.tensorflow.org/get_started/mnist/pros)
    * Lessons 0-4 from [USF](http://course.fast.ai/index.html)
    * Assignments 1-4 from [Udacity](https://www.udacity.com/course/deep-learning--ud730)
    * [CIFAR-10](https://www.tensorflow.org/tutorials/deep_cnn)
      * Extend to multiple GPUs
      * Visualizations (with Tensorboard): histogram summary for weights/biases/activations and layer-by-layer gradient norm recordings (+ how does batch norm affect them), graph visualization, cost over time
      * Visualizations for trained kernels: most-activating image from input set as viz, direct kernel image visualizations + maximizing image from input set as the viz [per maximizing inputs](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html), activations direct image viz (per [Yosinki et al 2015](http://yosinski.com/media/papers/Yosinski__2015__ICML_DL__Understanding_Neural_Networks_Through_Deep_Visualization__.pdf)). For maximizing inputs use regularization from Yosinki paper.
      * Faster input pipeline and timing metrics for each stage of operation [input pipeline notes](http://web.stanford.edu/class/cs20si/lectures/notes_09.pdf).
    * Assignment 2 from [Stanford CS20S1](http://web.stanford.edu/class/cs20si/syllabus.html)
    * Lab 1 from [MIT 6.S191](https://github.com/yala/introdeeplearning)
    * [Stanford CS231n](http://cs231n.github.io/)
    * Try out slightly less common techniques: compare initialization (orthogonal vs LSUV vs uniform), weight normalization vs batch normalization, Bayesian-inspired weight decay vs early stopping vs proximal regularization
    * Replicate [ResNet by He et al 2015](https://arxiv.org/abs/1512.03385), [Dropconnect](http://cs.nyu.edu/~wanli/dropc/), [Maxout](https://arxiv.org/abs/1302.4389), [Inception](https://github.com/tensorflow/models/tree/master/inception) (do a fine-tuning example with Inception per [this paper](http://proceedings.mlr.press/v32/donahue14.pdf)).
    * Do an end-to-end application from scratch. E.g., convert an equation image to LaTeX.
1. Sequence supervised learning
    * Gentle introductions
        * Lessons 5-7 from [USF](http://course.fast.ai/index.html)    
        * Assignments 5-6 from [Udacity](https://www.udacity.com/course/deep-learning--ud730)
        * [Karpathy RNN post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
        * Weeks 7-8 of [Hinton's Coursera course](https://www.coursera.org/learn/neural-networks)
    * Theory
        * Chapter 10 from [Goodfellow](http://www.deeplearningbook.org/)
    * Practice
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
        
