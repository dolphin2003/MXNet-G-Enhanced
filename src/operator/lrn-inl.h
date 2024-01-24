/*!
 * Copyright (c) 2015 by Contributors
 * \file lrn-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_LRN_INL_H_
#define MXNET_OPERATOR_LRN_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace lrn_enum {
enum LRNInputs {kData};
enum LRNOutputs {kOut, kTmpNorm};
}  // namespace lrn_enum

struct LRNParam : public dmlc::Parameter<LRNParam> {
  float alpha;
  float beta;
  float knorm;
  uint32_t nsize;
  DMLC_DECLARE_PARAMETER(LRNParam) {
    DMLC_DECLARE_FIELD(alpha).set_default(1e-4f)
    .describe("value of the alpha variance scaling parameter in the normalization formula");
    DMLC_DECLARE_FIELD(beta).set_default(0.75f)
    .describe("value of the beta power parameter in the normalization formula");
    DMLC_DECLARE_FIELD(knorm).set_default(2.0f)
    .describe("value of the k parameter in normalization formula");
    DMLC_DECLARE_FIELD(nsize)
    .describe("normalization window width in elements.");
  }
};  // struct LRNParam

template<typename xpu>
class LocalResponseNormOp : public Operator {
 public:
  explicit LocalResponseNormOp(LRNParam param) {
    param_ = param;
  }
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // TODO(xxx): Test with gradient chceker
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 2);
    // CHECK_EQ(req.size(), 2);
    CHECK_EQ(param_.nsize