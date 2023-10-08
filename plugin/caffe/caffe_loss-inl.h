/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_loss-inl.h
 * \brief Caffe Operator
 * \author Haoran Wang 
*/
#ifndef PLUGIN_CAFFE_CAFFE_LOSS_INL_H_
#define PLUGIN_CAFFE_CAFFE_LOSS_INL_H_

#include <caffe/proto/caffe.pb.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include <map>
#include <vector>
#include <string>
#include <utility>

#include "../../src/operator/operator_common.h"
#include "caffe_common.h"
#include "caffe_stream.h"
#include "caffe_fieldentry.h"
#include "caffe_blob.h"

namespace mxnet {
namespace op {

struct CaffeLossParam : public dmlc::Parameter<CaffeLossParam> {
  ::caffe::LayerParameter prototxt;
  int num_data, num_out;
  float grad_scale;

  DMLC_DECLARE_PARAMETER(CaffeLossParam) { DMLC_DECLARE_FIELD(prototxt).set_default("layer{}")
    .describe("Caffe's layer parameter");
    DMLC_DECLARE_FIELD(num_data).set_range(0, 100).set_default(2)
    .describe("Operator input number");
    DMLC_DECLARE_FIELD(num_out).set_range(0, 100).set_default(1)
    .describe("Operator output number");
    DMLC_DECLARE_FIELD(grad_scale)
    .set_default(1.0f)
    .describe("Scale the gradient by a float factor (a.k.a weight of this loss).");
  }
};

/**
 * \brief this is the implementation of caffe operator in caffe.
 * \tparam xpu the device that the op will be executed on.
 */
template<typename xpu, typename Dtype>
class CaffeLoss : public Operator {
 public:
  explicit CaffeLoss(CaffeLossParam p):param_(p),
                                       setup_(false) {
    std::string type = param_.prototxt.type();
    caffeOp_ = caffe::LayerRegistry<Dtype>::CreateLayer(param_.prototxt);
    grad_scale_ = (Dtype)param_.grad_scale;

    caffe::InitCaffeBlobs<Dtype>(&bot_, param_.num_data);
    caffe::InitCaffeBlobs<Dtype>(&top_, param_.num_out);
    flags_.resize(param_.num_data);
  }

  ~CaffeLoss() {
    caffe::DelCaffeBlobs(&bot_, param_.num_data);
    caffe::DelCaffeBlobs(&top_, param_.num_out);
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    // Set mode before forward
    caffe::CaffeMode::SetMode<xpu>();
    using ::caffe::Blob;
    using std::vector;
    using namespace mshadow;
    using namespace mshadow::expr;
    for (uint32_t i = 0; i < req.size(); ++i)
      CHECK_EQ(req[i], kWriteTo);

    CHECK_EQ(in_data.size(), param_.num_data);
    CHECK_EQ(out_data.size(), param_.num_out);

#if defined(__CUDACC__)
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // TODO(Haoran): when need cublas handle in stream?
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
          << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__

    caffe::TBlob2CaffeBlob<xpu, Dtype>(caffe::Data,
                                      bot_.begin(),
                                      in_data.begin(),
                                      param_.num_data);
    caffe::TBlob2CaffeBlob<xpu, Dtype>(caffe::Data,
                                      top_.begin(),
                                      out_data.begin(),
                                      param_.num_out);
    CaffeOpSetup();
    if (ctx.is_train)
      caffeOp_->SetPhase(::caffe::TRAIN);
    else
      caffeOp_->SetPhase(::caffe::TEST);
    caffeOp_->Forward(bot_, top_);

#if defined(__CUDACC__)
    // Sync cpu data to gpu data
    for (uint32_t i = 0; i < top_.size(); ++i)
      top_[i]->gpu_data();

    CHECK_EQ(cudaStreamSynchronize(NULL), cudaSuccess);
#endif  // __CUDACC__
  }

  // Set up caffe op with real data
  void CaffeOpSetup() {
    if (!setup_) {
      setup_ = true;
      caffeOp_->SetUp(bot_, top_);
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    // Set mode before backward
    caffe::CaffeMode::SetMode<xpu>();
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), param_.num_out);
    for (int i = 0; i < param_.num_data; ++i)
      CHECK(req[i] != kAddTo) << "caffe doesn't accm diff on bottom data";
    CHECK(in_data.size() == param_.num_data);

#if defined(__CUDACC__)
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // TODO(Haoran): when need cublas handle in stream?
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
          << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__

    caffe::TBlob2CaffeBlob<xpu, Dtype>(caffe::Grad,
                                      bot_.begin(),
                                      in_grad.begin(),
                                      param_.num_data);
    // Pass grad scale to caffe blob
    top_[0]->set_cpu_diff(&grad_scale_);

    // Set BP flag
    for (int i = 0; i < param_.num_data; ++i)
      flags_[i] = req[i] != kNullOp;

    caffeOp_->Backward(top_, flags_, bot_);

#if defined(__CUDACC__)
    // Sync cpu diff to gpu diff
    for (uint32_t i = 0; i < bot_.size(); ++i)
      bot_[i]->gpu_diff();

    CHECK_EQ(cudaStreamSynchronize(NULL), cudaSuccess);
#endif  // __CUDACC__
  }

 private:
  CaffeLossParam param_;
  ::caffe::Layer<Dtype> *caffeOp_;
  Dtype grad_scale_;
  std::vector< ::ca