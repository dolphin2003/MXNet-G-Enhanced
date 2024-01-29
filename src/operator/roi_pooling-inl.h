/*!
 * Copyright (c) 2015 by Contributors
 * \file roi_pooling-inl.h
 * \brief roi pooling operator and symbol
 * \author Kye-Hyeon Kim, Jian Guo
*/
#ifndef MXNET_OPERATOR_ROI_POOLING_INL_H_
#define MXNET_OPERATOR_ROI_POOLING_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"


namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace roipool {
enum ROIPoolingOpInputs {kData, kBox};
enum ROIPoolingOpOutputs {kOut, kMaxIdx};
}  // roipool

struct ROIPoolingParam : public dmlc::Parameter<ROIPoolingParam> {
  TShape pooled_size;
  float spatial_scale;
  DMLC_DECLARE_PARAMETER(ROIPoolingParam) {
    DMLC_DECLARE_FIELD(pooled_size)
    .set_expect_ndim(2).enforce_nonzero()
    .describe("fix pooled size: (h, w)");
    DMLC_DECLARE_FIELD(spatial_scale).set_range(0.0, 1.0)
    .describe("Ratio of input feature map height (or w) to raw image height (or w). "
    "Equals the reciprocal of total stride in convolutional layers");
  }
};

template<typename xpu, typename DType>
class ROIPoolingOp : public Operator {
 public:
  explicit ROIPoolingOp(ROIPoolingParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t expected = 2;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), expected);
    CHECK_EQ(out_data[roipool::kOut].shape_[0], in_data[roipool::kBox].shape_[0]);
    CHECK_EQ(out_data[roipool::kMaxIdx].shape_[0], in_data[roipool::kBox].shape_[0]);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[roipool::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> bbox = in_data[roipool::kBox].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[roipool::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> max_idx = out_data[roipool::kMaxIdx].get<xpu, 4, DType>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    CHECK_EQ(max_idx.CheckContiguous(), true);
    out = -FLT_MAX;
    max_idx = -1.0f;
    ROIPoolForward(out, data, bbox, max_idx, param_.spatial_scale);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t expected = 2;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), expected);
    CHECK_EQ(out_grad[roipool::kOut].shape_[0], in_data[roipool::kBox].shape_[0]);
    CHECK_EQ(out_data[roipool::kMaxIdx].shape_[0], in_data[roipool::kBox].shape_[0]);
    CHECK_NE(req[roipool::kData], kWriteInplace) <<
      "ROIPooling: Backward doesn't support kWriteInplace.";
    CHECK_NE(req[roipool::kBox], kWriteInplace) <<
      "ROIPooling: Backward doesn't support kWriteInplace.";
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> grad_out = out_grad[roipool::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> bbox = in_data[roipool::kBox].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> max_idx = out_data[roipool::kMaxIdx].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grad_in = in_grad[roipool::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> grad_roi = in_grad[roipool::kBox].get<xpu, 2, DType>(s);
    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(max_idx.CheckContiguous(), true);
    CHECK_EQ(grad_in.CheckContiguous(), true);
    if (kAddTo == req[roipool::kData] || kWriteTo == req[roipool::kData]) {
      if (kWriteTo == req[roipool::kData]) {
        grad_in = 0.0f;
      }
      ROIPoolBackwardAcc(grad_in, grad_out, bbox, max_idx, param_.spatial_scale);
    }
    if (kWriteTo == req[roipool::kBox]) {
      grad_roi = 0.0f;
    }
  }

 private:
  ROIPoolingParam param_;
};  // class ROIPoolingOp

// Decalre Factory