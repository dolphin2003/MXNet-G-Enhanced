Handwritten Digits Classification Competition
=============================================

[MNIST](http://yann.lecun.com/exdb/mnist/) is a handwritten digits image data set created by Yann LeCun. Every digit is represented by a 28x28 image. It has become a standard data set to test classifiers on simple image input. Neural network is no doubt a strong model for image classification tasks. There's a [long-term hosted competition](https://www.kaggle.com/c/digit-recognizer) on Kaggle using this data set.
We will present the basic usage of [mxnet](https://github.com/dmlc/mxnet/tree/master/R-package) to compete in this challenge.

This tutorial is written in Rmarkdown. You can download the source [here](https://github.com/dmlc/mxnet/blob/master/R-package/vignettes/mnistCompetition.Rmd) and view a
hosted version of tutorial [here](http://mxnet.readthedocs.io/en/latest/packages/r/mnistCompetition.html).

## Data Loading

First, let us download the data from [here](https://www.kaggle.com/c/digit-recognizer/data), and put them under the `data/` folder in your working directory.

Then we can read them in R and convert to matrices.


```r
require(mxnet)
```

```
## Loading required package: mxnet
## Loading required package: methods
```

```r
train <- read.csv('data/train.csv', header=TRUE)
test <- read.csv('data/test.csv', header=TRUE)
train <- data.matrix(train)
test <- data.matrix(test)

train.x <- train[,-1]
train.y <- train[,1]
```

Here every image is represented as a single row in train/test. The greyscale of each image falls in the range [0, 255], we can linearly transform it into [0,1] by


```r
train.x <- t(train.x/255)
test <- t(test/255)
```
We also transpose the input matrix to npixel x nexamples, which is the column major format accepted by mxnet (and the convention of R).

In the label part, we see the number of each digit is fairly even:


```r
table(train.y)
```

```
## train.y
##    0    1    2    3    4    5    6    7    8    9
## 4132 4684 4177 4351 4072 3795 4137 4401 4063 4188
```

## Network Configuration

Now we have the data. The next step is to configure the structure of our network.


```r
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")
```

1. In `mxnet`, we use its own data type `symbol` to configure the network. `data <- mx.symbol.Variable("data")` use `data` to represent the input data, i.e. the input layer.
2. Then we set the first hidden layer by `fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)`. This layer has `data` as the input, its name and the number of hidden neurons.
3. The activation is set by `act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")`. The activation function takes the output from the first hidden layer `fc1`.
4. The second hidden layer takes the result from `act1` as the input, with its name as "fc2" and the number of hidden neurons as 64.
5. the second activation is almost the same as `act1`, except we have a different input source and name.
6. Here comes the output layer. Since there's only 10 digits, we set the number of neurons to 10.
7. Finally we set the activation to softmax to get a probabilistic prediction.

## Training

We are almost ready for the training process. Before we start the computation, let's decide what device should we use.


```r
devices <- mx.cpu()
```

Here we assign CPU to `mxnet`. After all these preparation, you can run the following command to train the neural network! Note that `mx.set.seed` is the correct function to control the random process in `mxnet`.


```r
mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y,
                                     ctx=devices, num.round=10, array.batch.size=100,
                                     learning.rate=0.07, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))
```

```
## Start training with 1 devices
## Batch [100] Train-accuracy=0.6563
## Batch [200] Train-accuracy=0.777999999999999
## Batch [300] Train-accuracy=0.827466666666665
## Batch [400] Train-accuracy=0.855499999999999
## [1] Train-accuracy=0.859832935560859
## Batch [100] Train-accuracy=0.9529
## Batch [200] Train-accuracy=0.953049999999999
## Batch [300] Train-accuracy=0.955866666666666
## Batch [400] Train-accuracy=0.957525000000001
## [2] Train-accuracy=0.958309523809525
## Batch [100] Train-accuracy=0.968
## Batch [200] Train-accuracy=0.9677
## Batch [300] Train-accuracy=0.9696
## Batch [400] Train-accuracy=0.970650000000002
## [3] Train-accuracy=0.970809523809526
## Batch [100] Train-accuracy=0.973
## Batch [200] Train-accuracy=0.974249999999999
## Batch [300] Train-accuracy=0.976
## Batch [400] Train-accuracy=0.977100000000003
## [4] Train-accuracy=0.977452380952384
## Batch [100] Train-accuracy=0.9834
## Batch [200] Train-accuracy=0.981949999999999
## Batch [300] Train-accuracy=0.981900000000001
## Batch [400] Train-accuracy=0.982600000000003
## [5] Train-accuracy=0.983000000000003
## Batch [100] Train-accuracy=0.983399999999999
## Batch [200] Train-accuracy=0.98405
## Batch [300] Train-accuracy=0.985000000000001
## Batch [400] Train-accuracy=0.985725000000003
## [6] Train-accuracy=0.985952380952384
## Batch [100] Train-accuracy=0.988999999999999
## Batch [200] Train-accuracy=0.9876
## Batch [300] Train-accuracy=0.988100000000001
## Batch [400] Train-accuracy=0.988750000000003
## [7] Train-accuracy=0.988880952380955
## Batch [100] Train-accuracy=0.991999999999999
## Batch [200] Train-accuracy=0.9912
## Batch [300] Train-accuracy=0.990066666666668
## Batch [400] Train-accuracy=0.990275000000003
## [8] Train-accuracy=0.990452380952384
## Batch [100] Train-accuracy=0.9937
## Batch [200] Train-accuracy=0.99235
## Batch [300] Train-accuracy=0.991966666666668
## Batch [400] Train-accuracy=0.991425000000003
## [9] Train-accuracy=0.991500000000003
## Batch [100] Train-accuracy=0.9942
## Batch [200] Train-accuracy=0.99245
## Batch [300] Train-accuracy=0.992433333333334
## Batch [400] Train-accuracy=0.992275000000002
## [10] Train-accuracy=0.992380952380955
```

## Prediction and Submission

To make prediction, we can simply write


```r
preds <- predict(model, test)
dim(preds)
```

```
## [1]    10 28000
```

It is a matrix with 28000 rows and 10 cols, containing the desired classification probabilities from the output layer. To extract the maximum label for each row, we can