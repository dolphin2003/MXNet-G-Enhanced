FCN-xs EXAMPLES
---------------
This folder contains the examples of image segmentation in MXNet.

## Sample results
![fcn-xs pasval_voc result](https://github.com/dmlc/web-data/blob/master/mxnet/image/fcnxs-example-result.jpg)

we have trained a simple fcn-xs model, the parameter is below:

| model | lr (fixed) | epoch |
| ---- | ----: | ---------: |
| fcn-32s | 1e-10 | 31 |
| fcn-16s | 1e-12 | 27 |
| fcn-8s | 1e-14 | 19 |
(```when using the newest mxnet, you'd better using larger learning rate, such as 1e-4, 1e-5, 1e-6 instead, because the newest mxnet will do gradient normalization in SoftmaxOutput```)

the training image number is only : 2027, and the Validation image number is: 462  

## How to train fcn-xs in mxnet
#### step1: download the vgg16fc model and experiment data
* vgg16fc model : you can download the ```VGG_FC_ILSVRC_16_layers-symbol.json``` and ```VGG_FC_ILSVRC_16_layers-0074.params```   [baidu yun](http://pan.baidu.com/s/1bgz4PC), [dropbox](https://www.dropbox.com/sh/578n5cxej7ofd6m/AACuSeSYGcKQDi1GoB72R5lya?dl=0).  
this is the fully convolution style of the origin
[VGG_ILSVRC_16_layers.caffemodel](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel), and the corresponding [VGG_ILSVRC_16_layers_deploy.prototxt](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-vgg_ilsvrc_16_layers_deploy-prototxt), the vgg16 model has [license](http://creativecommons.org/licenses/by-nc/4.0/) for 