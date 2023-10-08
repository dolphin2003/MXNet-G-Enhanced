/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_op-inl.h
 * \brief Caffe Operator
 * \author Haoran Wang 
*/
#ifndef PLUGIN_CAFFE_CAFFE_OP_INL_H_
#define PLUGIN_CAFFE_CAFFE_OP_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <caffe/proto/caffe.pb.h>

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

struct CaffeOpParam : public dmlc::Parameter<CaffeOpParam> {
  ::caffe::LayerParameter prototxt;
  int num_data, num_weight, num_out;

  DMLC_DECLARE_PARAMETER(CaffeOpParam) { DMLC_DECLARE_FIELD(prototxt).set_default("layer{}")
    .describe("Caffe's layer parameter");
    DMLC_DECLARE_FIELD(num_data).set_default(1)
    .describe("Operator input number");
    DMLC_DECLARE_FIELD(num_weight).set_default(0)
    .describe("Weight number");
    DMLC_DECLARE_FIELD(num_out).set_default(1)
    .describe("Operator output number");
  }
};


/**
 * \brief this is the implementation of caffe operator in caffe.
 * \tparam xpu the device that the op will be executed on.
 */
template<typename xpu, typename Dtype>
class CaffeOp : public Operator {
 public:
  explicit CaffeOp(CaffeOpParam p):param_(p),
                                   init_w_(false),
                                   init_wd_(false),
                                   setup_(false) {
    std::string type = param_.prototxt.type();
    caffeOp_ = caffe::LayerRegistry<Dtype>::CreateLayer(param_.prototxt);

    caffe::InitCaffeBlobs<Dtype>(&bot_, param_.num_data);
    caffe::InitCaffeBlobs<Dtype>(&top_, param_.num_out);
    caffe::InitCaffeBlobs<Dtype>(&wei_, param_.num_weight);
    flags_.resize(param_.num_data);
  }

  ~CaffeOp() {
    caffe::DelCaffeBlobs(&bot_, param_.num_data);
    caffe::DelCaffeBlobs(&top_, param_.num_out);
    caffe::DelCaffeBlobs(&wei_, param_.num_weight);
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
    int expected_num_data = param_.num_weight + param_.num_data;
    CHECK_EQ(in_data.size(), expected_num_data);
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
                      