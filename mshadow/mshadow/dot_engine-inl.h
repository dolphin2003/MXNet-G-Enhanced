/*!
 *  Copyright (c) 2014 by Contributors
 * \file dot_engine-inl.h
 * \brief definitions of how Matrix Multiplications can be evaluated
 * \author Tianqi Chen
 */
#ifndef MSHADOW_DOT_ENGINE_INL_H_
#define MSHADOW_DOT_ENGINE_INL_H_

#include "./base.h"
#include "./extension/implicit_gemm.h"

#ifdef __CUDACC__
#include "./cuda/tensor_gpu-inl.cuh"
#endif  // #ifdef __CUDACC__

namespace mshadow {
 /*!
* \brief CPU/GPU: Get a batched view of the src array. dst[i] = src + i * stride
* \param dst 2D pointer
* \param src 1D pointer
* \param num number of batches
* \param stride size of each batch
* \param stream
*/
template<typename Device, typename DType>
inline void GetBatchedView(DType **dst, DType *src, int num, int stride,
                           Stream<Device> *stream);
template<typename DType>
inline void GetBatchedView(DType **dst, DType *src, int num, int stride,
                           Stream<cpu> *stream) {
  for (int i = 0; i < num; i++) {
    dst[i] = src + i * stride;
  }
}
#ifdef __CUDACC__
template<typename DType>
inline void GetBatchedView(DType **dst, DType *src, int num, int stride,
                    