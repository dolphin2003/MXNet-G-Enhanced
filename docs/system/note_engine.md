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

A dependency engine 