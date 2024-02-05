/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_output-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_SOFTMAX_OUTPUT_INL_H_
#define MXNET_OPERATOR_SOFTMAX_OUTPUT_INL_H_

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

namespace softmaxout_enum {
enum SoftmaxOutputOpInputs {kData, kLabel};
enum SoftmaxOutputOpOutputs {kOut};
enum SoftmaxOutputNormType {kNull, kBatch, kValid};
enum SoftmaxOutputOpResource {kTempSpace};
}  // namespace softmaxout_enum

struct SoftmaxOutputParam : public dmlc::Parameter<SoftmaxOutputParam> {
  float grad_scale;
  float ignore_label;
  bool multi_output;
  bool use_ignore;
  int normalization;
  DMLC_DECLARE_PARAMETER(SoftmaxOutputParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scale the gradient by a float factor");
    DMLC_DECLARE_FIELD(ignore_label).set_default(-1.0f)
    .describe("the label value will be ignored during backward (only works if "
      "use_ignore is set to be true).");
    DMLC_DECLARE_FIELD(multi_output).set_default(false)
    .describe("If set to true, for a (n,k,x_1,..,x_n) dimensional "
      "input tensor, softmax will generate n*x_1*...*x_n output, each "
      "has k classes");
    DMLC_DECLARE_FIELD(use_ignore).set_default(false)
    .describe("If set to true, the ignore_label value will not contribute "
      "to the backward gradient");
    DMLC_DECLARE_FIELD(normalization)
    .add_enum("null", softmaxout_enum::kNull)
    .add_enum("batch", softmaxout_enum::kBatch)
    .add_enum("valid", softmaxout_enum::kValid)
    .set_default(softmaxout_enum::kNull)
    .describe("If set to null, op will do nothing on output gradient."
              "If set to batch, op will normalize gradient by divide batch size"
              "If set to valid, op will normalize gradient by divide sample not ignored");
  };
};

template<typename xpu, typename DType>
class SoftmaxOutputOp : public Operator {
 public:
  explicit SoftmaxOutputOp(SoftmaxOutputParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2) << "SoftmaxOutput Input: [data, label]";
    CHECK_EQ(out_data.size(), 1) << "SoftmaxOutput Output: [output]";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (param_.multi_output) {
      int n = in_data[softmaxout_enum::kData].size(0);
      int k = in_data[softmaxout_enum::kData].size(1);
      Shape<3> s3 = Shape3(n, k, static_cast<int>(in_data[softmaxout_enum::kData].Size()/n/k));
      Tensor<xpu, 3, DType> data =
          in_data[softmaxout_enum::kData].get_with_shape<xpu, 3, DType>(s3, s);
      Tensor<xpu, 3, DType> out =
          out_data[softmaxout_enum::kOut].get_with_shape<xpu, 3, DType>(