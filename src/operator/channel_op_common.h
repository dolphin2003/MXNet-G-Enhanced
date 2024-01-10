/*!
 * Copyright (c) 2015 by Contributors
 * \file channel_op_common.h
 * \brief common function used for concat and split channel
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_CHANNEL_OP_COMMON_H_
#define MXNET_OPERATOR_CHANNEL_OP_COMMON_H_
#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <vector>
#include "./operator_common.h"

namespace mxnet {
namespace op {

template<typename xpu, int dim, int cdim, typename DType>
inline void concatenate_helper(const std::vector<mshadow::Tensor<xpu, dim, DType> > &input,
                               mshadow::Tensor<xpu, dim, DType> *output, const int dimension,
                               const OpReqType req) {
  using mshadow::expr::concat;
  using mshadow::expr::slice;

  if (dimension == cdim) {
    mshadow::Tensor<xpu, dim, DType> out = *output;
    size_t size = input.size();
    index_t begin = 0;
    for (index_t i = 0; i < size; ++i) {
      index_t end = begin + input[i].size(cdim);
      Assign(slice<cdim>(out, begin, end), req, input[i]);
      begin = end;
    }
  } else {
    concatenate_helper<xpu, dim, (cdim > 0 ? cdim - 1 : 0)>(input, output, dimension, req);
  }
}

template<typename xpu, int dim, typename DType>
inline void Concatenate(const std::vector<mshadow::Tensor<xpu, dim, DType> > &input,
                        mshadow::Tensor<xpu, dim, DType> *output, const int dimension,
                        const OpReqType req) {
  if (dimension < 0) {
    LOG(FATAL) << "dimension (" << dimension << ") must be greater than 0";
  } 