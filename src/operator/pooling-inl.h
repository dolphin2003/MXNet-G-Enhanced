/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling-inl.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_POOLING_INL_H_
#define MXNET_OPERATOR_POOLING_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace pool_enum {
enum PoolingOpInputs {kData};
enum PoolingOpOutputs {kOut};
enum PoolingOpType {kMaxPooling, kAvgPooling, kSumPooling};
}  // namespace pool_enum

struct PoolingParam : public dmlc::Parameter<PoolingParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  int pool_type;
  bool global_pool;
  DMLC_DECLARE_PARAMETER(PoolingParam) {
    DMLC_DECLARE_FIELD(global_pool).set_default(false)
    .describe("Ignore kernel size, do global pooling based on current input feature map. "
              "This is useful for input with different shape");

    DMLC_DECLARE_FIELD(kernel)
    .enforce_nonzero()
    .describe("pooling kernel size: (y, x) or (d, y, x)");

    DMLC_DECLARE_FIELD(pool_type)
    .add_enum("max", pool_enum::kMaxPooling)
    .add_enum("avg", pool_enum::kAvgPooling)
    .add_enum("sum", pool_enum::kSumPooling)
    .describe("Pooling type to be applied.");

    int stride_shape[] = {1, 1};
    DMLC_DECLARE_FIELD(stride).set_default(TShape(stride_shape, stride_shape + 2))
    .enforce_nonzero()
    .describe("stride: for pooling (y, x) or (d, y, x)");

    int pad_shape[] = {0, 0};
    DMLC_DECLARE_FIELD(pad).set_default(TShape(pad_shape, pad_shape + 2))
    .describe("pad for pooling: (y, x) or (d, y, x)");
  }
};

template<typename xpu, typename Reducer, typename DType>
class PoolingOp : public Operator {
 public:
  explicit PoolingOp(PoolingParam p) {
    this->param_ = p;
  }

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
    if (param_.kernel.ndim() == 3) {
      LOG(FATAL) << "Not implmented";
    }
    Tensor<xpu, 4, DType> data = in_data[pool_enum::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[pool_enum::kOut].get<xpu, 4, DType>(s);
    mshadow::Shape<2> out_shape = Shape2(out.shape_[2], out.shape_[3]);
    if (param_.pool_type == pool_enum::kMaxPooling || param_.pool_type == pool_enum::kSumPooling) {
      Assign(out,
             req[pool_enum::kOut],
             pool<Reducer>(pad(data, param_.pad[0], param_.pad[1]),
                           out_shape,
                           param_.global_pool ? data.shape_[2] : param_.kernel[0],
                           param_.global_pool ? data.shape_[3] : param_.kernel[1],
                           param_.global_pool ? 1 : param_.stride[0],
                           param_.global_pool ? 1 : param_.stride[1]))