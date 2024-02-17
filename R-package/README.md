
<img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/mxnetR.png width=155/> Deep Learning for R
==========================
[![Build Status](https://travis-ci.org/dmlc/mxnet.svg?branch=master)](https://travis-ci.org/dmlc/mxnet)
[![Documentation Status](https://readthedocs.org/projects/mxnet/badge/?version=latest)](http://mxnet.readthedocs.org/en/latest/packages/r/index.html)

You have reached the MXNet R Package! The MXNet R package brings efficiency and state-of-art deep learning to R.

- This package allows writing seamless tensor/matrix computation with multiple GPUs in R.
- You can also construct and customize top-notch deep learning models in R, and apply them to image classification and data science challenges.

This page contains links to all related documents on R package.

Resources
---------
* [MXNet R Package Document](http://mxnet.readthedocs.org/en/latest/packages/r/index.html)
  - Check this for detailed documents, examples, installation guides.

Installation
------------

For Windows/Mac users, we provide a pre-built binary package that uses CPU. Install the weekly updated package directly in R console:

```r
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
```

To use the GPU version or install on Linux, please follow [Installation Guide](http://mxnet.readthedocs.org/en/latest/how_to/build.html)

License
-------
MXNet R-package is licensed under [BSD](./LICENSE) license.