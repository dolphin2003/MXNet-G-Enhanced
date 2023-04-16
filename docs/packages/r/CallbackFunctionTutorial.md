MXNet R Tutorial on Callback Function
======================================

This vignette gives users a guideline for using and writing callback functions,
which can very useful in model training.

This tutorial is written in Rmarkdown.

- You can directly view the hosted version of the tutorial from [MXNet R Document](http://mxnet.readthedocs.io/en/latest/packages/r/CallbackFunctionTutorial.html)

- You can find the Rmarkdown source from [here](https://github.com/dmlc/mxnet/blob/master/R-package/vignettes/CallbackFunctionTutorial.Rmd)

Model training example
----------

Let's begin from a small example. We can build and train a model using the following code:


```r
library(mxnet)
data(BostonHousing, package="mlbench")
train.ind = seq(1, 506, 3)
train.x = data.matrix(BostonHousing[train.ind, -14])
train.y = BostonHousing[train.ind, 14]
test.x = data.matrix(BostonHousing[-train.ind, -14])
test.y = BostonHousing[-train.ind, 14]
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, num_hidden=1)
lro <- mx.symbol.LinearRegressionOutput(fc1)
mx.set.seed(0)
model <- mx.model.FeedForward.create(
  lro, X=train.x, y=train.y,
  eval.data=list(data=test.x, label=test.y),
  ctx=mx.cpu(), num.round=10, array.batch.size=20,
  learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse)
```

```
## Auto detect layout of input matrix, use rowmajor..
## Start training with 1 devices
## [1] Train-rmse=16.063282524034
## [1] Validation-rmse=10.1766446093622
## [2] Train-rmse=12.2792375712573
## [2] Validation-rmse=12.4331776190813
## [3] Train-rmse=11.1984634005885
## [3] Validation-rmse=10.3303041888193
## [4] Train-rmse=10.2645236892904
## [4] Validation-rmse=8.42760407903415
## [5] Train-rmse=9.49711005504284
## [5] Validation-rmse=8.44557808483234
## [6] Train-rmse=9.07733734175182
## [6] Validation-rmse=8.33225500266177
## [7] Train-rmse=9.07884450847991
## [7] Validation-rmse=8.38827833418459
## [8] Train-rmse=9.10463850277417
## [8] Validation-rmse=8.37394452365264
## [9] Train-rmse=9.03977049028532
## [9] Validation-rmse=8.25927979725672
## [10] Train-rmse=8.96870685004475
## [10] Validation-rmse=8.19509291481822
```

Besides, we provide two optional parameters, `batch.end.callback` and `epoch.end.callback`, which can provide great flexibility in model training.

How to use callback functions
---------

Two callback functions are provided in this package:

- `mx.callback.save.checkpoint` is used to save checkpoint to files each period iteration.


```r
model <- mx.