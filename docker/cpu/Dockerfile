FROM ubuntu:14.04
MAINTAINER Mu Li <muli@cs.cmu.edu>

# install the core library
RUN apt-get update && apt-get install -y build-essential git libopenblas-dev libopencv-dev
RUN git clone --recursive https://github.com/dmlc/mxnet/ && cd mxnet && \
    cp ma