/*!
 * Copyright (c) 2016 by Contributors
 * \file sequence_reverse-inl.h
 * \brief
 * \author Sebastian Bodenstien
*/

#ifndef MXNET_OPERATOR_SEQUENCE_REVERSE_INL_H_
#define MXNET_OPERATOR_SEQUENCE_REVERSE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./sequence_op_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace seq_reverse {
enum SequenceReverseOpInputs { kData, kSequenceLength };
enum SequenceReverseOpOutputs { kOut };
}

struct SequenceReverseParam : public dmlc::Parameter<SequenceReverseParam> {
  bool use_sequence_length;
  DMLC_DECLARE_PARAMETER(SequenceReverseParam) {
    DMLC_DECLARE_FIELD(use_sequence_length)
        .set_default(false)
        .describe(
            "If set to true, this layer takes in extra input sequence_length "
            "to specify variable length sequence");
  }
};

template <typename xpu, typename DType>
class SequenceReverseOp : public Operator {
 public:
  explicit S