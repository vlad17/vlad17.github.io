---
layout: post
title:  "Beating TensorFlow Training in-VRAM"
date:   2017-12-23
categories: hardware-acceleration machine-learning tools
meta_keywords: deep learning, tensorflow, GPU, data preprocessing, pipelining
---

# Beating TensorFlow Training in-VRAM

In this post, I'd like to introduce a technique that I've found helps accelerate mini-batch SGD training in my use case. I suppose this post could also be read as a public grievance directed towards the TensorFlow Dataset API optimizing for the large vision deep learning use-case, but maybe I'm just not hitting the right incantation to get `tf.Dataset` working (in which case, [drop me a line](https://github.com/vlad17/vlad17.github.io/issues/new)). The solution is to TensorFlow _harder_ anyway, so this shouldn't really be read as a complaint.

Nonetheless, if you are working with a new-ish GPU that has enough memory to hold a decent portion of your data alongside your neural network, you may find the final training approach I present here useful. The experiments I've run fall exactly in line with this "in-VRAM" use case (in particular, I'm training deep reinforcement learning value and policy networks on semi-toy environments, whose training profile is many iterations of training on a small replay buffer of examples). For some more context, you may want to check out an article on the [TensorForce blog](https://reinforce.io/blog/end-to-end-computation-graphs-for-reinforcement-learning/), which suggests that RL people should be building more of their TF graphs like this.

Briefly, if you have a dataset that fits into a GPU's memory, you're giving away a lot of speed with the usual TensorFlow pipelining or data-feeding approach, where the CPU delivers mini-batches whose forward/backward passes are computed on GPUs. This gets worse as you move to pricier GPUs, whose relative CPU-GPU bandwidth-to-GPU-speed ratio drops. Pretty easy change for a 2x.

## Punchline

Let's get to it. With numbers similar to my use case, 5 epochs of training take about **16 seconds** with the standard `feed_dict` approach, **12-20 seconds** with the TensorFlow Dataset API, and **8 seconds** with a custom TensorFlow control-flow construct.

This was tested on an Nvidia Tesla P100 with a compiled TensorFlow 1.4.1 (CUDA 9, cuDNN 7), Python 3.5. Here is the [test script](https://gist.github.com/vlad17/5d67eef9fb06c6a679aeac6d07b4dc9c). I didn't test it too many times ([exec trace](https://gist.github.com/vlad17/f43dba5783adfc21b1abab520dd2a8f1)). Feel free to change the data sizes to see if the proposed approach would still help in your setting.

Let's fix the toy benchmark supervised task we're looking at:

{% highlight python %}
import numpy as np
import tensorflow as tf

# pretend we don't have X, Y available until we're about
# to train the network, so we have to use placeholders. This is the case
# in, e.g., RL.
np.random.seed(1234)
# suffix tensors with their shape
# n = number of data points, x = x dim, y = y dim
X_nx = np.random.uniform(size=(1000 * 1024, 64))
Y_ny = np.column_stack([X_nx.sum(axis=1),
                        np.log(X_nx).sum(axis=1),
                        np.exp(X_nx).sum(axis=1),
                        np.sin(X_nx).sum(axis=1)])
nbatches = 10000 # == 20 epochs at 512 batch
batch_size = 512
{% endhighlight %}

### Vanilla Approach

This is the (docs-discouraged) approach that everyone really uses for training. Prepare a mini-batch on the CPU, ship it off to the GPU. _Note code here and below is excerpted (see the test script link above for the full code). It won't work if you just copy it._

{% highlight python %}
# b = batch size
input_ph_bx = tf.placeholder(tf.float32, shape=[None, X_nx.shape[1]])
output_ph_by = tf.placeholder(tf.float32, shape=[None, Y_ny.shape[1]])

# mlp = a depth 5 width 32 MLP net
pred_by = mlp(input_ph_bx)
tot_loss = tf.losses.mean_squared_error(output_ph_by, pred_by)
update = adam.minimize(tot_loss)

with tf.Session() as sess:
    for _ in range(nbatches):
        batch_ix_b = np.random.randint(X_nx.shape[0], size=(batch_size,))
        sess.run(update, feed_dict={
            input_ph_nx: X_nx[batch_ix_b],
            output_ph_ny: Y_ny[batch_ix_b]})
{% endhighlight %}

This drops whole-dataset loss from around 4500 to around 4, taking around **16 seconds** for training. You might worry that random-number generation might be taking a while, but excluding that doesn't drop the time more than **0.5 seconds**.

### Dataset API Approach

With the dataset API, we set up a pipeline where TensorFlow orchestrates some dataflow by synergizing more buzzwords on its worker threads. This should constantly feed the GPU by staging the next mini-batch while the current one is sitting on the GPU. This might be the case when there's a lot of data, but it doesn't seem to work very well when the data is small and GPU-CPU latency, not throughput, is the bottleneck.

Another unpleasant thing to deal with is that all those orchestrated workers and staging areas and buffers and shuffle queues need magic constants to work well. I tried my best, but it seems like performance is very sensitive with this use case. This could be fixed if Dataset detected (or could be told) it could be placed onto the GPU, and then it did so.

{% highlight python %}
# make training dataset, which should swallow the entire dataset once
# up-front and then feed it in mini-batches to the GPU
# presumably since we only need to feed stuff in once it'll be faster
ds = tf.data.Dataset.from_tensor_slices((input_ph_nx, output_ph_ny))
ds = ds.repeat()
ds = ds.shuffle(buffer_size=bufsize)
ds = ds.batch(batch_size)
# magic that Zongheng Yang (http://zongheng.me/) suggested I add that was
# necessary to keep this from being *worse* than feed_dict
ds = ds.prefetch(buffer_size=(batch_size * 5))
it = ds.make_initializable_iterator()
# reddit user ppwwyyxx further suggests folding training into a single call
def while_fn(t):
    with tf.control_dependencies([t]):
        next_bx, next_by = it.get_next()
        pred_by = mlp(next_bx)
        loss = tf.losses.mean_squared_error(next_by, pred_by)
        update = adam.minimize(loss)
        with tf.control_dependencies([update]):
            return t + 1
training = tf.while_loop(lambda t: t < nbatches,
                         while_fn, [0], back_prop=False)
with tf.Session() as sess:
    fd = {input_ph_nx: X_nx, output_ph_ny: Y_ny}
    sess.run(it.initializer, feed_dict=fd)
    sess.run(training)
{% endhighlight %}

For a small `bufsize`, like `1000`, this trains in around **12 seconds**. But then it's not actually shuffling the data too well (since all data points can only move by a position of 1000). Still, the loss drops from around 4500 to around 4, as in the `feed_dict` case. A large `bufsize` like `1000000`, which you'd think should effectively move the dataset onto the GPU entirely, performs _worse_ than `feed_dict` at around **20 seconds**.

I don't think I'm unfair in counting `it.initializer` time in my benchmark (which isn't that toy, either, since it's similar to my RL use case size). All the training methods need to load the data onto the GPU, and the data isn't available until run time.

### Using a TensorFlow Loop

This post isn't a tutorial on `tf.while_loop` and friends, but this code does what was promised: just feed everything once into the GPU and do all your epochs without asking for permission to continue from the CPU.

{% highlight python %}
# generate random batches up front
# i = iterations
n = tf.shape(input_ph_nx)[0]
batches_ib = tf.random_uniform((nbatches, batch_size), 0, n, dtype=tf.int32)

# use a fold + control deps to make sure we only train on the next batch
# after we're done with the first
def fold_fn(prev, batch_ix_b):
    X_bx = tf.gather(input_ph_nx, batch_ix_b)
    Y_by = tf.gather(output_ph_ny, batch_ix_b)
    # removing control deps here probably gives you Hogwild!
    with tf.control_dependencies([prev]):
        pred_by = mlp(X_bx)
        loss = tf.losses.mean_squared_error(Y_by, pred_by)
        with tf.control_dependencies([opt.minimize(loss)]):
            return tf.constant(0)

training = tf.foldl(fold_fn, batches_ib, 0, back_prop=False)

with tf.Session() as sess:
    fd = {input_ph_nx: X_nx, output_ph_ny: Y_ny}
    sess.run(training, feed_dict=fd)
{% endhighlight %}

This one crushes at around **8 seconds**, dropping loss again from around 4500 to around 4.

## Discussion

It's pretty clear Dataset isn't feeding as aggressively as it can, and its many widgets and knobs don't help (well, they do, but only after making me do more work). But, if TF wants to invalidate this blog post, I suppose it could add yet another option that plops the dataset into the GPU.
