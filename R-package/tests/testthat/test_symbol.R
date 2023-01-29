require(mxnet)

context("symbol")

test_that("basic symbol operation", {
  data = mx.symbol.Variable('data')
  net1 = mx.symbol.FullyConnected(data=