require(mxnet)

# lstm cell symbol
lstm <- function(num.hidden, indata, prev.state, param, seqidx, layeridx, dropout=0) {
    if (dropout > 0)
        indata <- mx.symbol.Dropout(data=indata, p=dropout)
    i2h <- mx.symbol.FullyConnected(data=indata,
                                    weight=param$i2h.weight,
                                    bias=param$i2h.bias,
                                    num.hidden=num.hidden * 4,
                                    name=paste0("t", seqidx, ".l", layeridx, ".i2h"))
    h2h <- mx.symbol.FullyConnected(data=prev.state$h,
                                    weight=param$h2h.weight,
                                    bias=param$h2h.bias,
                                    num.hidden=num.hidden * 4,
                                    name=paste0("t", seqidx, ".l", layeridx, ".h2h"))
    gates <- i2h + h2h
    slice.gates <- mx.symbol.SliceChannel(gates, num.outputs=4,
                                          name=paste0("t", seqidx, ".l", layeridx, ".slice"))

    in.gate <- mx.symbol.Activation(slice.gates[[1]], act.type="sigmoid")
    in.transform <- mx.symbol.Activation(slice.gates[[2]], act.type="tanh")
    forget.gate <- mx.symbol.Activation(slice.gates[[3]], act.type="sigmoid")
    out.gate <- mx.symbol.Activation(slice.gates[[4]], act.type="sigmoid")
    next.c <- (forget.gate * prev.state$c) + (in.gate * in.transform)
    next.h <- out.gate * mx.symbol.Activation(next.c, act.type="tanh")
    
    return (list(c=next.c, h=next.h))
}

# unrolled lstm network
lstm.unroll <- function(num.lstm.layer, seq.len, input.size,
                        num.hidden, num.embed, num.label, dropout=0.) {

    embed.weight <- mx.symbol.Variable("embed.weight")
    cls.weight <- mx.symbol.Variable("cls.weight")
    cls.bias <- mx.symbol.Variable("cls.bias")
    param.cells <- list()
    last.states <- list()
    for (i in 1:num.lstm.layer) {
        param.cells[[i]] <- list(i2h.weight = mx.symbol.Variable(paste0("l", i, ".i2h.weight")),
                                 i2h.bias = mx.symbol.Variable(paste0("l", i, ".i2h.bias")),
                                 h2h.weight = mx.symbol.Variable(paste0("l", i, ".h2h.weight")),
                                 h2h.bias = mx.symbol.Variable(paste0("l", i, ".h2h.bias")))
        state <- list(c=mx.symbol.Variable(paste0("l", i, ".init.c")),
                      h=mx.symbol.Variable(paste0("l", i, ".init.h")))
        last.states[[i]] <- state
    }

    last.hidden <- list()
    label <- mx.symbol.Variable("label")
    for (seqidx in 1:seq.len) {
        # embeding layer
        data <- mx.symbol.Variable(paste0("t", seqidx, ".data"))

        hidden <- mx.symbol.Embedding(data=data, weight=embed.weight,
                                      input.dim=input.size,
                                      output.dim=num.embed,
                                      name=paste0("t", seqidx, ".embed"))
        
        # stack lstm
        for (i in 1:num.lstm.layer) {
            if (i==0) {
                dp <- 0
            }
            else {
                dp <- dropout
            }
            next.state <- lstm(num.hidden, indata=hidden,
                               prev.state=last.states[[i]],
                               param=param.cells[[i]],
                               seqidx=seqidx, layeridx=i, 
                             