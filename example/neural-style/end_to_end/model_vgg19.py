
import mxnet as mx
import os, sys
from collections import namedtuple

ConvExecutor = namedtuple('ConvExecutor', ['executor', 'data', 'data_grad', 'style', 'content', 'arg_dict'])

def get_vgg_symbol(prefix, content_only=False):
    # declare symbol
    data = mx.sym.Variable("%s_data" % prefix)
    conv1_1 = mx.symbol.Convolution(name='%s_conv1_1' % prefix, data=data , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), workspace=1024)
    relu1_1 = mx.symbol.Activation(data=conv1_1 , act_type='relu')
    conv1_2 = mx.symbol.Convolution(name='%s_conv1_2' % prefix, data=relu1_1 , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), workspace=1024)
    relu1_2 = mx.symbol.Activation(data=conv1_2 , act_type='relu')
    pool1 = mx.symbol.Pooling(data=relu1_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv2_1 = mx.symbol.Convolution(name='%s_conv2_1' % prefix, data=pool1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), workspace=1024)
    relu2_1 = mx.symbol.Activation(data=conv2_1 , act_type='relu')
    conv2_2 = mx.symbol.Convolution(name='%s_conv2_2' % prefix, data=relu2_1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), workspace=1024)
    relu2_2 = mx.symbol.Activation(data=conv2_2 , act_type='relu')
    pool2 = mx.symbol.Pooling(data=relu2_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv3_1 = mx.symbol.Convolution(name='%s_conv3_1' % prefix, data=pool2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), workspace=1024)
    relu3_1 = mx.symbol.Activation(data=conv3_1 , act_type='relu')
    conv3_2 = mx.symbol.Convolution(name='%s_conv3_2' % prefix, data=relu3_1 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), workspace=1024)
    relu3_2 = mx.symbol.Activation(data=conv3_2 , act_type='relu')
    conv3_3 = mx.symbol.Convolution(name='%s_conv3_3' % prefix, data=relu3_2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), workspace=1024)
    relu3_3 = mx.symbol.Activation(data=conv3_3 , act_type='relu')
    conv3_4 = mx.symbol.Convolution(name='%s_conv3_4' % prefix, data=relu3_3 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), workspace=1024)
    relu3_4 = mx.symbol.Activation(data=conv3_4 , act_type='relu')
    pool3 = mx.symbol.Pooling(data=relu3_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv4_1 = mx.symbol.Convolution(name='%s_conv4_1' % prefix, data=pool3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), workspace=1024)
    relu4_1 = mx.symbol.Activation(data=conv4_1 , act_type='relu')
    conv4_2 = mx.symbol.Convolution(name='%s_conv4_2' % prefix, data=relu4_1 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), workspace=1024)
    relu4_2 = mx.symbol.Activation(data=conv4_2 , act_type='relu')
    conv4_3 = mx.symbol.Convolution(name='%s_conv4_3' % prefix, data=relu4_2 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), workspace=1024)
    relu4_3 = mx.symbol.Activation(data=conv4_3 , act_type='relu')
    conv4_4 = mx.symbol.Convolution(name='%s_conv4_4' % prefix, data=relu4_3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), workspace=1024)
    relu4_4 = mx.symbol.Activation(data=conv4_4 , act_type='relu')
    pool4 = mx.symbol.Pooling(data=relu4_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv5_1 = mx.symbol.Convolution(name='%s_conv5_1' % prefix, data=pool4 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), workspace=1024)
    relu5_1 = mx.symbol.Activation(data=conv5_1 , act_type='relu')


    if content_only:
        return relu4_2
    # style and content layers
    style = mx.sym.Group([relu1_1, relu2_1, relu3_1, relu4_1, relu5_1])
    content = mx.sym.Group([relu4_2])
    return style, content


def get_executor_with_style(style, content, input_size, ctx):
    out = mx.sym.Group([style, content])
    # make executor
    arg_shapes, output_shapes, aux_shapes = out.infer_shape(data=(1, 3, input_size[0], input_size[1]))
    arg_names = out.list_arguments()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes]))
    grad_dict = {"data": arg_dict["data"].copyto(ctx)}
    # init with pretrained weight
    pretrained = mx.nd.load("./model/vgg19.params")
    for name in arg_names:
        if name == "data":
            continue
        key = "arg:" + name
        if key in pretrained:
            pretrained[key].copyto(arg_dict[name])
        else:
            print("Skip argument %s" % name)
    executor = out.bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, grad_req="write")
    return ConvExecutor(executor=executor,
                        data=arg_dict["data"],
                        data_grad=grad_dict["data"],
                        style=executor.outputs[:-1],
                        content=executor.outputs[-1],
                        arg_dict=arg_dict)

def get_executor_content(content, input_size, ctx):
    arg_shapes, output_shapes, aux_shapes = content.infer_shape(data=(1, 3, input_size[0], input_size[1]))
    arg_names = out.list_arguments()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes]))
    pretrained = mx.nd.load("./model/vgg19.params")
    for name in arg_names:
        if name == "data":
            continue
        key = "arg:" + name
        if key in pretrained:
            pretrained[key].copyto(arg_dict[name])
        else:
            print("Skip argument %s" % name)
    executor = out.bind(ctx=ctx, args=arg_dict, args_grad=[], grad_req="null")
    return ConvExecutor(executor=executor,
                        data=arg_dict["data"],
                        data_grad=None,
                        style=None,
                        content=executor.outputs[0],
                        arg_dict=arg_dict)

