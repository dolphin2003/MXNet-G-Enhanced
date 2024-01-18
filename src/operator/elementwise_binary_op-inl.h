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
      mshadow::Tensor<xpu, 1, DType> mlhs_grad = lhs_grad->FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad = rhs_grad->FlatTo1D<xpu, DType>(s);
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, F<mshadow_op::identity>(mout_grad));
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, F<mshadow_op::identity>(mout_grad));
    });
}

template<typename xpu>
void MinusBackward_(const OutputGrad& out_grad,
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
      mshadow::Tensor<xpu, 1, DType> mlhs_grad = lhs_grad->FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad = rhs_grad->FlatTo1D<xpu, DType>(s);
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, F<mshadow_op::identity>(mout_grad));
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, F<mshadow_op::negation>(mout_grad));
    });
}

template<typename xpu>
void MulBackward_(const OutputGrad& out_grad,
                  const Input0& lhs,
                  const Input1& rhs,
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
      mshadow::Tensor<xpu, 1, DType> mlhs_data = lhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_data = rhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad = lhs_grad->FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad = rhs_grad->FlatTo1D<xpu, DType>(s);
      CHECK_NE(req_rhs_grad, kWriteInplace);
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, mlhs_data * mout_grad);
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, mrhs_data * mout_grad);
    });
}

template<typename xpu>
void DivBackward_(const OutputGrad& out_grad,
                  const Input0& lhs,
                  const Input1& rhs,
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
      mshadow::Tensor<xpu, 1, DType> mlhs_data = lhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_data = rhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad = lhs_grad->FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad = rhs_grad->FlatTo1D<xpu, DType>(s);
      CHECK_NE(req_rhs_grad, kWriteInplace);
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad,
                      F<mshadow_op::negation>(mout_grad * mlhs_data)/
                      F<mshadow_op::square>(mrhs_data));
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, mout_grad /  mrhs_data);
    });
}

template<typename xpu>
void PowerBackward_(const OutputGrad& out_grad,
                    const Input0& lhs,
                    const Input1& rhs,
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
      mshadow::Tensor<xpu, 1, DType> mlhs_data = lhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_data = rhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad = lhs_grad->FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad = rhs_grad->FlatTo1D<xpu, DType>(s);
      CHECK_NE(req_rhs_grad, kWriteInplace);
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad,
                      F<mshadow_op::log>(mlhs_data) *
                      F<mshadow_op::power>(mlhs_data, mrhs_data) * mout_grad);
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad,
                      mrhs_data *
                      F<mshadow_op::power>(mlhs_data, mrhs_data - scalar<DType>(1)) *
                      mout_grad);
    });
}

template<typename xpu>
void MaximumBackward_(const OutputGrad& out_grad,
                      const Input0& lhs,
                      const Input1& rhs,
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
      mshadow::Tensor<xpu, 1, DType> mlhs_data = lhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_data = rhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad = lhs_grad->FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad = rhs_grad->FlatTo1D<xpu, DType>(s);
      CHECK_NE(req_rhs_grad, kWriteInplace);
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad,
                      mout_grad * F<mshadow_op::maximum_grad>(mrhs_data, mlhs_data));
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad,
                      mout_grad * F<mshadow_op::maximum_grad>(mlhs_data, mrhs_data));
    });
}

template<typename xpu>
void MinimumBackward_(const OutputGrad& out_grad,
                      const Input0& lhs,
                      const Input1& rhs,
                      const EnvArguments& env,
                      TBlob* lhs_grad,
                      TBlob* rhs_grad,
                      OpReqType req_lhs_grad,
                      OpReqType req_rhs_grad,
                      RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      mshadow::Tensor<xpu, 1, DType> mout_grad = out_grad.data.FlatT