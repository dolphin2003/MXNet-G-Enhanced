import numpy as np
import mxnet as mx


def reldiff(a, b):
    diff = np.sum(np.abs(a - b))
    norm = np.sum(np.abs(a))
    reldiff = diff  / norm
    return reldiff


def check_bind_with_uniform(uf, gf, dim, sf=None, lshape=None, rshape=None):
    """check function consistency with uniform random numbers"""
    shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
    lhs = mx.symbol.Variable('lhs')
    rhs = mx.symbol.Variable('rhs')
    if sf is not None:
        ret = sf(lhs, rhs)
    else:
        ret = uf(lhs, rhs)

    assert ret.list_arguments() == ['lhs', 'rhs']
    lshape = shape if lshape is None else lshape
    rshape = shape if rshape is None else rshape

    lhs_arr = mx.nd.array(np.random.uniform(-1, 1, lshape))
    rhs_arr = mx.nd.array(np.random.uniform(-1, 1, rshape))
    lhs_grad = mx.nd.empty(lshape)
    rhs_grad = mx.nd.empty(rshape)
    executor = ret.bind(mx.Context('cpu'),
                        args=[lhs_arr, rhs_arr],
                        args_grad=[lhs_grad, rhs_grad])

    exec3 = ret.bind(mx.Context('cpu'),
                     args=[lhs_arr, rhs_arr])


    exec4 = ret.bind(mx.Context('cpu'),
                     args={'rhs': rhs_arr, 'lhs': lhs_arr},
                     args_grad={'lhs': lhs_grad, 'rhs': rhs_grad})

    executor.forward()
    exec3.forward()
    exec4.forward()
    out2 = executor.outputs[0].asnumpy()
    out1 = uf(lhs_arr.asnumpy(), rhs_arr.asnumpy())
    out3 = exec3.outputs[0].asnumpy()
    out4 = exec4.outputs[0].asnumpy()
    assert reldiff(out1, out2) < 1e-6
    assert reldiff(out1, out3) < 1e-6
    assert reldiff(out1, out4) < 1e-6
    # test gradient
    out_grad = mx.nd.array(np.ones(out2.shape))
    lhs_grad2, rhs_grad2 = gf(out_grad.asnumpy(),
                              lhs_arr.asnumpy(),
                              rhs_arr.asnumpy())
    executor.backward([out_grad])

    assert reldiff(lhs_grad.asnumpy(), lhs_grad2) < 1e-6
    assert reldiff(rhs_grad.asnumpy(), rhs_grad2) < 1e-6


def test_bind():
    np.random.seed(0)
    nrepeat = 10
    maxdim = 4
    for repeat in range(nrepeat):
        for dim in range(1, maxdim):
            check_bind_with_uniform(lambda x, y: x + y,
                                    lambda g, x, y: (g, g),
                                    dim)
            check_bind_with_uniform(lambda x, y: x - y,
                                    lambda g, x, y: (g, -g),
                                    dim)
            check_bind_with_uniform(lambda x, y: x * y,
                                    lambda g, x, y: (y * g, x * g),
                                    dim)
            check_bind_with_uniform(lambda x, y: x / y,
                                    lambda g, x, y: (g / y, -x * g/ (y**2)),
                                    dim)

            check_bind_with_uniform(lambda x, y: np.maximum(x, y),
                                    lambda g, x, y: (g * (x>y), g * (y>x)),
                                    dim,
                                    sf=mx.symbol.maximum)
            check_bind_with_uniform(lambda x, y: np.minimum(x, y),
                                    lambda g, x, y: (g * (x<y), g * (y<x)),
                                    dim,
                                    sf=mx.symbol.minimum)

def test_dot():
    np.random.seed(0)
    nrepeat = 10
    maxdim = 4
    for repeat in range(nrepeat):
        s =tuple(np.random.randint(1, 500, size=3))
        check_bind_with_uniform(lambda x, y: np.dot(x, y),
                                lambda g, x, y: (np.dot(g, y.T), np.dot(x.T, g)),
                                2,
                                lshape=(s[0], s[1]),
                                rshape=(s[1], s[2]),
                                sf = mx.symbol.dot)
    for repeat in range(nrepeat):
        s =tuple(np.random.randint(1, 500, size=1))
        check_bind_with_uniform(lambda x, y: np.dot(x, y),
                                lambda g, x, y: (g * y, g * x),
                                2,
     