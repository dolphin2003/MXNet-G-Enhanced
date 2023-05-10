library(mxnet)

conv_factory <- function(data, num_filter, kernel, stride,
                         pad, act_type = 'relu', conv_type = 0) {
    i