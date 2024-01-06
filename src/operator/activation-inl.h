/*!
 * Copyright (c) 2015 by Contributors
 * \file activation-inl.h
 * \brief Activation operator
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_ACTIVATION_INL_H_
#define MXNET_OPERATOR_ACTIVATION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {
// Declare enumeration of input order to make code more intuitive.
// // These enums are only visible within this header
namespace activation {
enum ActivationOpInputs {kData};
enum ActivationOpOutputs {kOut};
enum ActivationOpType {kReLU, kSigmoid, kTanh, kSoftReLU};
}  // activation

struct ActivationParam : public dmlc::Parameter<ActivationParam> {
  // use int for enumeration
  int act_type;
  DMLC_DECLARE_PARAMETER(ActivationParam) {
    DMLC_DECLARE_FIELD(act_type)
    .add_enum("relu", activation::kReLU)
    .add_enum("sigmoid", activation::kSigmoid)
    .add_enum("tanh", activation::kTanh)
    .add_enum("softrelu", activation::kSoftReLU)
    .describe("Activation function to be applied.");
  }
};

/**
 * \brief This is the implementation of activation operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename ForwardOp, typename BackwardOp, typename DType>
class ActivationOp : public Operator {
 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> data = in_data[activation::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[activation::kOut].FlatTo2D<xpu, DType>(s);
    Assign(out, req[activation::kOut], F<ForwardOp>(data));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> m_out_grad = out_grad[activation::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> m_out_data = out_data[activation::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> m_in_grad = in_grad[activation::kData].FlatTo2D<xpu, DType>(s);
    Assign(m_in_grad, req[activation::kData], F<BackwardOp>(m_out_data) * m_out_grad);
  }
};  // class ActivationOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(ActivationParam type, int dtype);

#if DMLC_USE_CX