/*!
 *  Copyright (c) 2015 by Contributors
 * \file elementwise_binary_op-inl.h
 * \brief Function defintion of elementwise binary operators
 */
#ifndef MXNET_OPERATOR_ELEMENTWISE_BINARY_OP_INL_H_
#define MXNET_OPERATOR_ELEMENTWISE_BINARY_OP_INL_H_

#include <mxnet/operator_util.h>
#include "./mshadow_op.h"

#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {

template<typename xpu, typename OP>
void BinaryForward_(const TBlob& lhs,
                    const TBlob& rhs,
                    const EnvArguments& env,
                    TBlob *ret,
                    OpReqType req,
                    RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
    << "Binary function only support input/output with the same type";
  CHECK_EQ(ret->type_flag_, rhs.type_flag_)
    << "Binary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    mshadow::Tensor<xpu, 1, DType> out = ret->FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req,
                    F<OP>(lhs.FlatTo1D<xpu, DType>(s),
                          rhs.FlatTo1D<xpu, DType>(s)));
  });
}


template<typename xpu>
void PlusBackward_(const OutputGrad& out_grad,
                   const EnvArguments& env,
                   TBlob* lhs_grad,
                   TBlob* rhs_grad,
                   OpReqType req_lhs_grad,
                   OpReqType req_rhs_grad,
                   RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      mshadow::Tensor<xpu, 1, DType> mout_grad = out_grad.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad = lhs_g