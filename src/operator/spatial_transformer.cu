/*!
 * Copyright (c) 2016 by Contributors
 * \file spatial_transformer.cu
 * \brief
 * \author Wei Wu
*/

#include "./spatial_transformer-inl.h"
#include <algorithm>
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR == 5
#include "./cudnn_spatial_transformer-inl.h"
#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR

namespace mshadow {
template<typename DType>
__global__ void BilinearSamplingForwardKernel(const int i_c, const int i_h,
                                              const int i_w, const DType* data,
                                              const DType* grid, const int o_n,
                                              const int o_c, const int o_h,
                                              const int o_w, DType* out) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < o_n * o_c * o_h * o_w;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, h, w) is the element in out
    int w = index % o_w;
    int h = (index / o_w) % o_h;
    int c = (index / o_w / o_h) % o_c;
    int n = index / o_w / o_h / o_c;
    index_t out_index = n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w;
    index_t grid_index = n * o_h * o_w * 2 + h * o_w + w;
    DType y