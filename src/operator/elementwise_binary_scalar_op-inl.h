/*!
 *  Copyright (c) 2015 by Contributors
 * \file elementwise_binary_scalar_op-inl.h
 * \brief Function defintion of elementwise binary operators
 */
#ifndef MXNET_OPERATOR_ELEMENTWISE_BINARY_SCALAR_OP_INL_H_
#define MXNET_OPERATOR_ELEMENTWISE_BINARY_SCALAR_OP_INL_H_

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
void BinaryScalarLForward_(const TBlob& lhs,
                           const EnvArguments& env,
                           TBlob *ret,
                           OpReqType req,
                           RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
    << "Binary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    mshadow::Tensor<xpu, 1, DType> out = ret->FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req,
                    F<OP>(lhs.FlatTo1D<xpu, DType>(s),
                          scalar<DType>(env.scalar)));
  });
}

template<typename xpu, typename OP>
void BinaryScalarRForward_(const TBlob& rhs,
                           const EnvArguments& env,
                           TBlob *ret,
                           OpReqType req,
                           RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, rhs.type_flag_)
    << "Binary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    mshadow::Tensor<xpu, 1, DType> out = ret->FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req,
                    F<OP>(scalar<DType>(env.scalar),
                          rhs.FlatTo1D<xpu, DType>(s)));
  });
}

template<typename xpu, typename BackwardOp>
void BinaryScalarBackwardT0_(const OutputGrad& out_grad,
                             const EnvArguments& env,
                             TBlob *in_grad,
                             OpReqType req,
                             RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(in_grad->type_flag_, out_grad.data.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
    mshadow::Tensor<xpu, 1, DType> igrad = in_grad->FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(igrad, req,
                    F<BackwardOp>(out_grad.data.FlatTo1D<xpu, DType>()));
    });
}

template<typename xpu, typename BackwardOp>
void BinaryScalarBackwardT1_(const OutputGrad& out_grad,
                             const EnvArguments& env,
                             TBlob *in_grad,
                             OpReqType req,
                             RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(in_grad->type_flag_, out_grad.data.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
    mshadow::Tensor<xpu, 1, DType> igrad = in_grad->FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(igrad, req,
                    F<BackwardOp>(out_grad.data.FlatTo1D<xpu, DType>(),
                                  scalar<DType>(env.scalar)));
  });
}

template<typename xpu, typename BackwardOp>
void BinaryScalarBackwardT2_(const OutputGrad& out_grad,
                             const Input0& lhs,
                             const EnvArguments& env,
                             TBlob *in_grad,
                             OpReqType req,
                             RunContext ctx) {
  