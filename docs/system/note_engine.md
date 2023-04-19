Dependency Engine for Deep Learning
===================================
One always important theme of deep learning libraries is to run faster and scale to larger
datasets. In order to do so, one natural direction is to always go beyond using one device (GPU),
and make use of more computation resources.

When library designer started to think about this problem, one natural theme will occur.
How can we ***parallelize*** the computation across devices? More importantly,
how do we ***synchronize*** the computation when we introduce multi-threading?

Runtime dependency engine is a generic solution to such problems. This article discusses
the runtime dependency scheduling problem in deep learning. We will introduce the dependency
scheduling problem, how it can help make multi-device deep learning easier and faster, and
discuss possible designs of a generic dependency engine that is library and operation independent.

Most design details of this article inspires the dependency engine of mxnet, with the dependency tracking algorithm majorly contributed by [Yutian Li](https://github.com/hotpxl) and [Mingjie Wang](https://github.com/jermainewang).

Dependency Scheduling Problem
-----------------------------
While most of the users want to take advantage of parallel computation,
most of us are more used to serial programs. So it is interesting to ask
if we can write serial programs, and build a library to automatically parallelize
operations for you in an asynchronized way.

For example, in the following code snippet. We can actually run ```B = A + 1```
and ```C = A + 2``` in any order, or in parallel.
```python
A = 2
B = A + 1
C = A + 2
D = B * C
```

However, it is quite hard to code the sequence manually, as the last operation,
```D = B * C```, needs to wait for both the above operations to complete before it starts running.
We can represent the computation as the following dependency graph.

![Dep Simple](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/engine/dep_simple.png)

In this specific case, the graph is also called data-flow graph, as it represents the dependency
in terms of data and computation.

A dependency engine is a library that takes some sequence of operations, and schedules them
correctly according to the dependency pattern, and potentially in parallel. So in the toy example,
a dependency library could run ```B = A + 1``` and ```C = A + 2``` in parallel, and run ```D = B * C```
after both operations complete.

Problems in Dependency Scheduling
---------------------------------
In last section we introduced what is dependency engine mean in this article. It seems a quite interesting
thing to use as it relieves our burden from writing concurrent programs. However, as things go parallel,
there are new (dependency tracking)problems that arises which need to be solved in order to make the running
program correct and efficient. In this section, let us discuss the problems we will encounter in deep
learning libraries when things go parallel.

### Data Flow Dependency
The central thing that almost every dependency engine will have to solve, is the dataflow dependency problem.

![Dep Simple](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/engine/dep_simple.png)

Data Flow dependency describes how the outcome of one computation can be used in other computations.
As we have elaborated this in last section, we will only put the same figure here. Libraries that have
data flow tracking engines include Minerva and Purine2.

### Correct Memory Recycle
One problem that we will encounter is when to recycle memory we allocated to the arrays.
This is simple in the serial case. Because we can simply recycle the memory after the variable
go out of scope. However, things becomes a bit harder in parallel case. Consider the following
example

![Dep Del](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/engine/dep_del.png)

In the above example, because both computation needs to use values from A. We cannot perform
the memory recycling before these computation completes. So a correct engine
need to schedule the memory recycle operations according to the dependency, and make sure it
is executed after both ```B = A + 1``` and ```C = A + 2``` completes.


### Random Number Generation
Random number generators are commonly used in machine learning. However, they also bring
interesting challenges for dependency engine. Consider the following example

![Dep Rand](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/engine/dep_rand.png)

Here we are generating random numbers in a sequence. While it seems that the two random number
generations can be parallelized. This is usually not the case. Because usually a pseudorandom
number generator (PRNG) is not thread-safe because it might contain some internal state to mutate
when generating a new number. Even if the PRNG is thread-safe, it is still desirable to
run the generation in the a serialized way, so we can get reproducible random numbers.

So in this case, actually what we might want to do is to serialize the operations that uses the same PRNG.

Case Study on Multi-GPU Neural Net
----------------------------------
In the last section, we have discussed the problem we may facing in dependency engine.
But before thinking about how we can design a generic engine to solve this problems.
Let us first talk about how can dependency engine help in multi GPU training of neural net.
The following is a pseudo python program that describes one batch training of two layer
neural net.

```python
# Example of one iteration Two GPU neural Net
data = next_batch()
data[gpu0].copyfrom(data[0:50])
data[gpu1].copyfrom(data[50:100])
# forward, backprop on GPU 0
fc1[gpu0] = FullcForward(data[gpu0], fc1_weight[gpu0])
fc2[gpu0] = FullcForward(fc1[gpu0], fc2_weight[gpu0])
fc2_ograd[gpu0] = LossGrad(fc2[gpu0], label[0:50])
fc1_ograd[gpu0], fc2_wgrad[gpu0] =
     FullcBackward(fc2_ograd[gpu0] , fc2_weight[gpu0])
_, fc1_wgrad[gpu0] = FullcBackward(fc1_ograd[gpu0] , fc1_weight[gpu0])
# forward, backprop on GPU 1
fc1[gpu1] = FullcForward(data[gpu1], fc1_weight[gpu1])
fc2[gpu1] = FullcForward(fc1[gpu1], fc2_weight[gpu1])
fc2_ograd[gpu1] = LossGrad(fc2[gpu1], label[50:100])
fc1_ograd[gpu1], fc2_wgrad[gpu1] =
     FullcBackward(fc2_ograd[gpu1] , fc2_weight[gpu1])
_, fc1_wgrad[gpu1] = FullcBackward(fc1_ograd[gpu1] , fc1_weight[gpu1])
# aggregate gradient and update
fc1_wgrad[cpu]  = fc1_wgrad[gpu0] + fc1_wgrad[gpu1]
fc2_wgrad[cpu]  = fc2_wgrad[gpu0] + fc2_wgrad[gpu1]
fc1_weight[cpu] -= lr *  fc1_wgrad[gpu0]
fc2_weight[cpu] -= lr *  fc2_wgrad[gpu0]
fc1_weight[cpu].copyto(fc1_weight[gpu0] , fc1_weight[gpu1])
fc2_weight[cpu].copyto(fc2_weight[gpu0] , fc2_weight[gpu1])
```
In this program, the example 0 to 50  are copied to GPU0 and example 50 to 100
are copied to GPU1. The calculated gradient are aggregated in CPU, which then performs
a simple SGD update, and copies the updated weight back to each GPU.
This is a common data parallel program written in a serial manner.
The following dependency graph shows how it can be parallelized:

![Dep Net](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/engine/dep_net.png)

Few important notes:
- The copy of gradient to CPU, can happen as soon as we get gradient of that layer.
- The copy back of weight, can also be done as soon as the weight get updated.
- In the forward pass, actually we have a dependency to ```fc1_weight[cpu].copyto(fc1_weight[gpu0] , fc1_weight[gpu1])```
  from previous iteration.
- There is a lag of computation between last backward to layer k to next forward call to layer k.
	- We can do the weight synchronization of layer k ***in parallel*** with other computation in this lag.

The points mentioned in above list is the exact optimization used by multi GPU deep learning libaries such as cxxnet.
The idea is to overlap the weight synchronization(communication) with the computation.
However, as you may find out it is really not easy to do that, as the copy need to be triggered as soon as backward of
that layer completes, which then triggers the reduction, updates etc.

Having a dependency engine to schedule these operations makes our life much easier, by pushing the task of multi-threading
and dependency tracking to the engine.

Design a Generic Dependency Engine
----------------------------------
Now hopefully you are convinced that a dependency engine is useful for scaling deep learning programs to multiple devices.
Let us now discuss how we can actually design a generic interface for dependency engine, and how we can implement one.
We need to emphasize that solution discussed in this section is not the only possible design for dependency engine,
but rather an example that we think is useful to most cases.

One of the most g