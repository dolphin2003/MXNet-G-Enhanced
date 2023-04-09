# Run MXNet on Multiple CPU/GPUs with Data Parallel

MXNet supports trainig with multiple CPUs and GPUs since the very
beginning. Almost any program using MXNet's provided training modules, such as
[python/mxnet.model](https://github.com/dmlc/mxnet/blob/master/python/mxnet/model.py),
can be efficiently run over multiple devices.

## Data Parallelism

In default MXNet uses data parallelism to partition the workload over multiple
devices. Assume there are *n* devices, then each one will get the complete model
and train it on *1/n* of the data. The results such as the gradient and
updated model are communicated cross these devices.

## Multiple GPUs within a Single Machine

### Workload Partitioning

If using data parallelism, MXNet will evenly partition a minbatch in each
GPUs. Assume we train with batch size *b* and *k* GPUs, then in one iteration
each GPU will perform forward and backward on a batch with size *b/k*. The
gradients are then summed over all GPUs before updating the model.

In ideal case, *k* GPUs will provide *k* time speedup comparing to the single
GPU. In addition, assume the model has size *m* and the temporal workspace is
*t*, then the memory footprint of each GPU will be *m+t/k*. In other words, we
can use a large batch size for multiple GPUs.

### How to Use

> To use GPUs, we need to compiled MXNet with GPU support. For
> example, set `USE_CUDA=1` in `config.mk` before `make`. (see
> [MXNet installation guide](build.html) for more options).

If a machine has one or more than one GPU cards installed, then each card is
labeled by a number starting from 0. To use a particular GPU, one can often
either specify the context `ctx` in codes or pass `--gpus` in commandlines. For
example, to use GPU 0 and 2 in python one can often create a model with
```python
import mxnet as mx
model = mx.model.FeedForward(ctx=[mx.gpu(0), mx.gpu(2)], ...)
```
while if the program accepts a `--gpus` flag such as
[example/image-classification](https://github.com/dmlc/mxnet/tree/master/example/image-classification),
then we can try
```bash
python train_mnist.py --gpus 0,2 ...
```

### Advanced Usage

If the GPUs are have different computation power, we can partition the workload
according to their powers. For example, if GPU 0 is 3 times faster than GPU 2,
then we provide an additional workload option `work_load_list=[3, 1]`, see
[model.fit](../packages/python/model.html#mxnet.model.FeedForward.fit) for more
details.

Training with multiple GPUs should have the same results as a single GPU if all
other hyper-parameters are the same. But in practice the results vary mainly due
to the randomness of I/O (random order or other augmentations), weight
initialization with different seeds, and CUDNN.

We can control where the gradient is aggregated and model updating if performed
by creating different `kvstore`, which is the module for data
communication. There are three options,
which vary on speed and memory consumption:

```eval_rst
==========================  ====================  ================
kvstore type                gradient aggregation  weight updating
==========================  ====================  ================
``local_update_cpu``        CPU                   CPU
``local_allreduce_cpu``     CPU                   all GPUs
``local_allreduce_device``  one GPU               all GPUs
==========================  ====================  ================
```

Here
- `local_update_cpu`: gradients are first copied to CPU memory, and aggregated
  on CPU. Then we update the weight on CPU and copy back the updated weight to
  GPUs. It is suitable when the layer model size is not large, such as
  convolution layers.

- `local_allreduce_cpu` is similar to `local_update_cpu` except that the
  aggregated gradients are copied back to each GPUs, and the weight is updated
  there. Note that, comparing to `local_update_cpu`, each weight is updated by
  *k* times if there are *k* GPUs. But it might be still faster when the model
  size is large, such as fully connected layers, in which GPUs is much faster
  than CPU. Also note that, it may use more GPU memory because we need to store
  the variables needed by the updater in GPU memory.

- `local_allreduce_device`, or simplified as `device`, is similar to
   `local_allreduce_cpu` except that the we use a particular GPU to aggregated
   the gradients. It may be faster than `local_allreduce_cpu` if the gradient
   size is huge, where the gradient summation operation could be the
   bottleneck. However, it uses even more GPU memory since we need to store the
   aggregated gradient on GPU.

The `kvstore` type is `local` in default. It will choose `local_update_cpu` if the
weight size of each layer is less than 1Mb, which can be changed by
the environment varialbe `MXNET_KVSTORE_BIGARRAY_BOUND`, and
`local_allreduce_cpu` otherwise.

## Distributed Training with Multiple Machines

### Data Consistency 