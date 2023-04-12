# Deep Learning in a Single File for Smart Devices

Deep learning (DL) systems are complex and often have a few of dependencies. It is often painful to port a DL library into different platforms, especially for smart devices. There is one fun way to solve this problem:  provide a light interface and ***put all required codes into a single file*** with minimal dependencies. In this tutorial we will give details on how to do the amalgamation. In addition, we will show a demo to run image object recognition on mobile devices.

## Amalgamation: Make the Whole System into a Single File

The idea of amalgamation comes from SQLite and other projects, which packs all the codes into a single source file. Then it is only needed to compile that single file to create the library, which makes porting to various platforms much easier. MXNet provides an [amalgamation](https://github.com/dmlc/mxnet/tree/master/amalgamation) script, thanks to [Jack Deng](https://github.com/jdeng), to combiles all codes needed for prediction using trained DL models into a single `.cc` file, which has around 30K lines of codes. The only dependency required is just a BLAS library.

We also have a minimal version removed BLAS dependency, and the single file can be compiled into JavaScript by using [enscripten](https://github.com/kripken/emscripten).

The compiled library can be used by any other programming language easily. The `.h` file contains a light prediction API, porting to another language with a C foreign function interface needs little effect. For example

- Go: [https://github.com/jdeng/gomxnet](https://github.com/jdeng/gomxnet)
- Java: [https://github.com/dmlc/mxnet/tree/master/amalgamation/jni](https://github.com/dmlc/mxnet/tree/master/amalgamation/jni)
- Python: [https://github.com/dmlc/mxnet/tree/master/amalgamation/python](https://github.com/dmlc/mxnet/tree/master/amalgamation/python)


To do amalgamation, there are a few things we need to be careful about when building the project:

- Minimize the dependency to other libraries and do.
- Use namespace to encapsulate the types and operators.
- Avoid do commands such as ```using namespace xyz``` on the global scope.
- Avoid cyclic include dependencies.


## Image Recognition Demo on Mobile

With amalgamation, deploying the system on smart devices (such as Android or iOS) is simple. But there are two additional things we need to consider:

1. The model should be small enough to fit into the device’s memory
2. The model should not be too expensive to run given the relative low computational power of these devices

Next we will use the image recognition as an example to show how we try to get such a model. We start with the s