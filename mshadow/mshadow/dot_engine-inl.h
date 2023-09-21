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
                           Stream<gpu> *stream) {
  cuda::GetBatchedView(dst, src, num, stride, stream);
}
#endif  // #ifdef __CUDACC__

namespace expr {
//---------------------------------------------------------------------
// Matrix Multiplications, depends on BLAS Engine
//---------------------------------------------------------------------
template<typename SV, typename Device, int ddim, int ldim,
         int rdim, bool ltrans, bool rtrans, typename DType>
struct DotEngine {
  inline static void Eval(Tensor<Device, ddim, DType> *p_dst,
                          const Tensor<Device, ldim, DType> &lhs,
                          const Tensor<Device, rdim, DType> &rhs,
                          DType scale);
};
// handles the dot, use CblasColMajor
template<typename Device, typename DType = default_real_t>
struct BLASEngine {
  inline static bool GetT(bool t) {
    return t ? true : false;
  }
  inline static void SetStream(Stream<Device> *stream) {
  }
  inline static void gemm(Stream<Device> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, DType alpha,
                          const DType *A, int lda, const DType *B, int ldb,
                          DType beta, DType *C, int ldc) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_gemm(Stream<Device> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, DType alpha,
                                  const DType *A, int lda, const DType *B, int ldb,
                                  DType beta, DType *C, int ldc, int batch_count,
                                  DType **workspace) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void gemv(Stream<Device> *stream,
                          bool trans, int m, int n,
                          DType alpha, const DType *A, int lda,
                          const DType *X, int incX,
                          DType beta, DType *Y, int incY) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_gemv(Stream<Device> *stream,
                                  bool trans, int m, int n,
                                  DType alpha, const DType *A, int lda,
                                  const DType *X, int incX,
                                  DType beta, DType *Y, int incY, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void ger(Stream<Device> *stream,
                         int m, int n, DType alpha,
                         const DType *X, int incX,
                         const DType *Y, int incY, DType *A, int lda) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_ger(Stream<Device> *stream,
                         int m, int n, DType alpha,
                         const DType *X, int incX,
                         const DType *Y, int incY, DType *A, int lda, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void dot(Stream<Device> *stream,
                         int n,
                         const DType* X, int incX,
                         const DType* Y, int incY,
                         DType* ret) {
    LOG(FATAL) << "Not implmented!";
  }
};

#if MSHADOW_STAND_ALONE
template<>
struct BLASEngine<cpu, float> {
  inline static bool GetT(bool t) {
    return t ? true : false;
  }
  inline static void SetStream(Stream<cpu> *stream) {
  }
  inline static void gemm(Stream<cpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, float alpha,
                          const float *A, int lda, const float *B, int ldb,
                          float beta, float *C, int ldc) {
    if (alpha == 1.0f && beta == 0.0f) {
      bool transpose_left = transb;
      bool transpose_right = transa;
      Tensor<cpu, 2, float> lhs((float*)B, Shape2(transpose_left ? k : n, transpose_left ? n : k));  // NOLINT(*)
      Tensor<cpu, 2, float> rhs((float*)A, Shape2(transpose_right ? m : k, transpose_right ? k : m));  // NOLINT(*)
      Tensor<cpu, 2, float> dst(C, Shape2(m, n));
      if (!transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs, rhs); return;
      } else if (!transpose_left && transpose_right) {
        dst = expr::implicit_dot(lhs, rhs.T()); return;
      } else if (transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs.T(), rhs); return;
      } else {
        LOG(FATAL) << "Not implmented!";
      }
    } else {
      LOG(FATAL) << "Not implmented!";
    }
  }
  inline static void batched_gemm(Stream<cpu> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, float alpha,
                                  const float *A, int lda, const float *B, int ldb,
                                  float beta, float *C, int ldc, int batch_count,
                                  float **workspace) {
    for (int i = 0; i < batch_count; ++i) {
      gemm(stream, transa, transb, m, n, k, alpha,
           A + i * m * k, lda, B + i * k * n, ldb,
           beta, C + i * m * n, ldc);
    }
  }
  inline static void gemv(Stream<cpu> *stream,
                          bool trans, int m, int n,
                          float alpha, const float *A, int lda,
                          const float *X, int incX,
                          float beta, float *Y, int incY) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_gemv(Stream<cpu> *stream,
                                  bool trans, int m, int n,
                                  float alpha, const float *A, int lda,
                                  const float *X, int incX,
                                  float beta, float *Y, int incY, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void ger(Stream<cpu> *stream,
                         int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_ger(Stream<cpu> *stream,
                         int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void dot(Stream<cpu> *stream,
                         int n,
                         const float* X, int incX,
                         const float* Y, int incY,
      