/*!
 * Copyright (c) 2015 by Contributors
 * \file warpctc-inl.h
 * \brief warpctc operator
 * \author Liang Xiang
*/
#ifndef PLUGIN_WARPCTC_WARPCTC_INL_H_
#define PLUGIN_WARPCTC_WARPCTC_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <stdio.h>
#include <ctc.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include "../../src/operator/operator_common.h"

namespace mxnet {
namespace op {

namespace warpctc_enum {
  enum CTCOpInputs {kData, kLabel};
  enum CTCOpOutputs {kOut};
  enum CTCTemp {kTmp};
}  // namespace warpctc_enum

struct WarpCTCParam : public dmlc::Parameter<WarpCTCParam> {
  int label_length;
  int input_length;
  DMLC_DECLARE_PARAMETER(WarpCTCParam) {
    DMLC_DECLARE_FIELD(label_length)
        .set_default(0)
        .describe("Real label length");
    DMLC_DECLARE_FIELD(input_length)
        .set_default(0)
        .describe("Input length");
  }
};

template<typename xpu>
class WarpCTCOp : public Operator {
 private:
  WarpCTCParam param_;

 public:
  explicit WarpCTCOp(WarpCTCParam p) {
    this->param_ = p;
  }

  ~WarpCTCOp() {
  }

  inline void throw_on_error(ctcStatus_t status, const char* message) {
    if (status != CTC_STATUS_SUCCESS) {
      throw std::runtime_error(message
                               + (", stat = "
                                  + std::string(ctcGetStatusString(status))));
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       