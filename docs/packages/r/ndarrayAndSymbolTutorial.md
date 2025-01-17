
MXNet R Tutorial on NDArray and Symbol
======================================

This vignette gives a general overview of MXNet"s R package.  MXNet contains a
mixed flavor of elements to bake flexible and efficient
applications. There are two major concepts introduced in this tutorial.

* [NDArray](#ndarray-numpy-style-tensor-computations-on-cpus-and-gpus)
  offers matrix and tensor computations on both CPU and GPU, with automatic
  parallelization
* [Symbol](#symbol-and-automatic-differentiation) makes defining a neural
  network extremely easy, and provides automatic differentiation.

## NDArray: Vectorized tensor computations on CPUs and GPUs

`NDArray` is the basic vectorized operation unit in MXNet for matrix and tensor computations.
Users can perform usual calculations as on R"s array, but with two additional features:

1.  **multiple devices**: all operations can be run on various devices including
CPU and GPU
2. **automatic parallelization**: all operations are automatically executed in
   parallel with each other

### Create and Initialization

Let"s create `NDArray` on either GPU or CPU


```r
require(mxnet)
```

```
## Loading required package: mxnet
## Loading required package: methods
```

```r
a <- mx.nd.zeros(c(2, 3)) # create a 2-by-3 matrix on cpu
b <- mx.nd.zeros(c(2, 3), mx.cpu()) # create a 2-by-3 matrix on cpu
# c <- mx.nd.zeros(c(2, 3), mx.gpu(0)) # create a 2-by-3 matrix on gpu 0, if you have CUA enabled.
```

As a side note, normally for CUDA enabled devices, the device id of GPU starts from 0.
So that is why we passed in 0 to GPU id. We can also initialize an `NDArray` object in various ways:


```r
a <- mx.nd.ones(c(4, 4))
b <- mx.rnorm(c(4, 5))
c <- mx.nd.array(1:5)
```

To check the numbers in an `NDArray`, we can simply run


```r
a <- mx.nd.ones(c(2, 3))
b <- as.array(a)
class(b)
```

```
## [1] "matrix"
```

```r
b
```

```
##      [,1] [,2] [,3]
## [1,]    1    1    1
## [2,]    1    1    1
```

### Basic Operations

#### Elemental-wise operations

You can perform elemental-wise operations on `NDArray` objects:


```r
a <- mx.nd.ones(c(2, 4)) * 2
b <- mx.nd.ones(c(2, 4)) / 8
as.array(a)
```

```
##      [,1] [,2] [,3] [,4]
## [1,]    2    2    2    2
## [2,]    2    2    2    2
```

```r
as.array(b)
```

```
##       [,1]  [,2]  [,3]  [,4]
## [1,] 0.125 0.125 0.125 0.125
## [2,] 0.125 0.125 0.125 0.125
```

```r
c <- a + b
as.array(c)
```

```
##       [,1]  [,2]  [,3]  [,4]
## [1,] 2.125 2.125 2.125 2.125
## [2,] 2.125 2.125 2.125 2.125
```

```r
d <- c / a - 5
as.array(d)
```

```
##         [,1]    [,2]    [,3]    [,4]
## [1,] -3.9375 -3.9375 -3.9375 -3.9375
## [2,] -3.9375 -3.9375 -3.9375 -3.9375
```

If two `NDArray`s sit on different divices, we need to explicitly move them
into the same one. For instance:


```r
a <- mx.nd.ones(c(2, 3)) * 2
b <- mx.nd.ones(c(2, 3), mx.gpu()) / 8
c <- mx.nd.copyto(a, mx.gpu()) * b
as.array(c)
```

#### Load and Save

You can save a list of `NDArray` object to your disk with `mx.nd.save`:


```r
a <- mx.nd.ones(c(2, 3))
mx.nd.save(list(a), "temp.ndarray")
```

You can also load it back easily:


```r
a <- mx.nd.load("temp.ndarray")
as.array(a[[1]])
```

```
##      [,1] [,2] [,3]
## [1,]    1    1    1
## [2,]    1    1    1
```

In case you want to save data to the distributed file system such as S3 and HDFS,
we can directly save to and load from them. For example:


```r
mx.nd.save(list(a), "s3://mybucket/mydata.bin")
mx.nd.save(list(a), "hdfs///users/myname/mydata.bin")
```

### Automatic Parallelization

`NDArray` can automatically execute operations in parallel. It is desirable when we
use multiple resources such as CPU, GPU cards, and CPU-to-GPU memory bandwidth.

For example, if we write `a <- a + 1` followed by `b <- b + 1`, and `a` is on CPU while
`b` is on GPU, then want to execute them in parallel to improve the
efficiency. Furthermore, data copy between CPU and GPU are also expensive, we
hope to run it parallel with other computations as well.

However, finding the codes can be executed in parallel by eye is hard. In the
following example, `a <- a + 1` and `c <- c * 3` can be executed in parallel, but `a <- a + 1` and
`b <- b * 3` should be in sequential.


```r
a <- mx.nd.ones(c(2,3))
b <- a
c <- mx.nd.copyto(a, mx.cpu())
a <- a + 1
b <- b * 3
c <- c * 3
```

Luckily, MXNet can automatically resolve the dependencies and
execute operations in parallel with correctness guaranteed. In other words, we
can write program as by assuming there is only a single thread, while MXNet will
automatically dispatch it into multi-devices, such as multi GPU cards or multi
machines.

It is achieved by lazy evaluation. Any operation we write down is issued into a
internal engine, and then returned. For example, if we run `a <- a + 1`, it
returns immediately after pushing the plus operator to the engine. This
asynchronous allows us to push more operators to the engine, so it can determine
the read and write dependency and find a best way to execute them in
parallel.

The actual computations are finished if we want to copy the results into some
other place, such as `as.array(a)` or `mx.nd.save(a, "temp.dat")`. Therefore, if we
want to write highly parallelized codes, we only need to postpone when we need
the results.

## Symbol and Automatic Differentiation

WIth the computational unit `NDArray`, we need a way to construct neural networks. MXNet provides a symbolic interface named Symbol to do so. The symbol combines both flexibility and efficiency.

### Basic Composition of Symbols

The following codes create a two layer perceptrons network:


```r
require(mxnet)
net <- mx.symbol.Variable("data")
net <- mx.symbol.FullyConnected(data=net, name="fc1", num_hidden=128)
net <- mx.symbol.Activation(data=net, name="relu1", act_type="relu")
net <- mx.symbol.FullyConnected(data=net, name="fc2", num_hidden=64)
net <- mx.symbol.Softmax(data=net, name="out")
class(net)
```

```
## [1] "Rcpp_MXSymbol"
## attr(,"package")
## [1] "mxnet"
```

Each symbol takes a (unique) string name. *Variable* often defines the inputs,
or free variables. Other symbols take a symbol as the input (*data*),
and may accept other hyper-parameters such as the number of hidden neurons (*num_hidden*)
or the activation type (*act_type*).

The symbol can be simply viewed as a function taking several arguments, whose
names are automatically generated and can be get by


```r
arguments(net)
```

```
## [1] "data"       "fc1_weight" "fc1_bias"   "fc2_weight" "fc2_bias"
## [6] "out_label"
```

As can be seen, these arguments are the parameters need by each symbol:

- *data* : input data needed by the variable *data*
- *fc1_weight* and *fc1_bias* : the weight and bias for the first fully connected layer *fc1*
- *fc2_weight* and *fc2_bias* : the weight and bias for the second fully connected layer *fc2*
- *out_label* : the label needed by the loss

We can also specify the automatic generated names explicitly:


```r
data <- mx.symbol.Variable("data")
w <- mx.symbol.Variable("myweight")
net <- mx.symbol.FullyConnected(data=data, weight=w, name="fc1", num_hidden=128)
arguments(net)
```

```
## [1] "data"     "myweight" "fc1_bias"
```

### More Complicated Composition

MXNet provides well-optimized symbols for
commonly used layers in deep learning. We can also easily define new operators
in python.  The following example first performs an elementwise add between two
symbols, then feed them to the fully connected operator.


```r
lhs <- mx.symbol.Variable("data1")
rhs <- mx.symbol.Variable("data2")
net <- mx.symbol.FullyConnected(data=lhs + rhs, name="fc1", num_hidden=128)
arguments(net)
```

```
## [1] "data1"      "data2"      "fc1_weight" "fc1_bias"
```

We can also construct symbol in a more flexible way rather than the single
forward composition we addressed before.


```r
net <- mx.symbol.Variable("data")
net <- mx.symbol.FullyConnected(data=net, name="fc1", num_hidden=128)
net2 <- mx.symbol.Variable("data2")
net2 <- mx.symbol.FullyConnected(data=net2, name="net2", num_hidden=128)
composed.net <- mx.apply(net, data=net2, name="compose")
arguments(composed.net)
```

```
## [1] "data2"       "net2_weight" "net2_bias"   "fc1_weight"  "fc1_bias"
```

In the above example, *net* is used a function to apply to an existing symbol
*net*, the resulting *composed.net* will replace the original argument *data* by
*net2* instead.

### Training a Neural Net.

The [model API](../../../R-package/R/model.R) is a thin wrapper around the symbolic executors to support neural net training.

You are also highly encouraged to read [Symbolic Configuration and Execution in Pictures for python package](../python/symbol_in_pictures.md),
which provides a detailed explanation of concepts in pictures.

### How Efficient is Symbolic API

In short, they design to be very efficienct in both memory and runtime.

The major reason for us to introduce Symbolic API, is to bring the efficient C++
operations in powerful toolkits such as cxxnet and caffe together with the
flexible dynamic NArray operations. All the memory and computation resources are
allocated statically during Bind, to maximize the runtime performance and memory
utilization.

The coarse grained operators are equivalent to cxxnet layers, which are
extremely efficient.  We also provide fine grained operators for more flexible
composition. Because we are also doing more inplace memory allocation, mxnet can
be ***more memory efficient*** than cxxnet, and gets to same runtime, with
greater flexiblity.