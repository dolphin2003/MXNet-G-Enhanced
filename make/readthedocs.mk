
#--------------------------------------------------------
# Configuration for document generation with less deps
# The library may not run, but doc generation could work
#--------------------------------------------------------

# choice of compiler
export CC = gcc
export CXX = g++
export NVCC = nvcc

# whether use CUDA during compile
USE_CUDA = 0

# add the path to CUDA library to link and compile flag
# if you have already add them to environment variable, leave it as NONE
USE_CUDA_PATH = NONE

# whether use opencv during compilation
# you can disable it, however, you will not able to use
# imbin iterator
USE_OPENCV = 0

# whether use CUDNN R3 library
USE_CUDNN = 0


# use openmp for parallelization
USE_OPENMP = 0

#
# choose the version of blas you want to use
# can be: mkl, blas, atlas, openblas
USE_STATIC_MKL = NONE
USE_BLAS = NONE
#
# add path to intel library, you may need it
# for MKL, if you did not add the path to environment variable
#
USE_INTEL_PATH = NONE


# the additional link flags you want to add
ADD_LDFLAGS = -lgomp

# the additional compile flags you want to add
ADD_CFLAGS = -DMSHADOW_STAND_ALONE=1 -DMSHADOW_USE_SSE=0
#
# If use MKL, choose static link automatically to fix python wrapper
#
ifeq ($(USE_BLAS), mkl)
	USE_STATIC_MKL = 1
endif

#------------------------
# configuration for DMLC
#------------------------
# whether use HDFS support during compile
# this will allow cxxnet to directly save/load model from hdfs
USE_HDFS = 0

# whether use AWS S3 support during compile
# this will allow cxxnet to directly save/load model from s3
USE_S3 = 0

# path to libjvm.so
LIBJVM=$(JAVA_HOME)/jre/lib/amd64/server

# uses O0 instead of O3 for better performance
DEBUG = 1