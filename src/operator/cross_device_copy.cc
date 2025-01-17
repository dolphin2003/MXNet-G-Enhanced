
/*!
 * Copyright (c) 2015 by Contributors
 * \file cross_device_copy.cc
 * \brief Special operator that copys NDArray
*/
#include <dmlc/logging.h>
#include <mxnet/operator.h>

namespace mxnet {
namespace op {

class CrossDeviceCopyOp : public Operator {
 public:
  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data,
               const std::vector<TBlob> &aux_args) override {
    // CrossDeviceCopy is specially handled by graph executor,
    // We still re-use things such as InferShape in OperatorProperty
    LOG(FATAL) << "Not Reached";
  }

  ExecType exec_type() const override {
    // TODO(tianqi) Think of other way to blend cross device op into operator interface.
    // declare the op as cross device,
    return kCrossDeviceCopy;
  }
};

class CrossDeviceCopyProp : public OperatorProperty {
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
  }

  std::map<std::string, std::string> GetParams() const override {
    return std::map<std::string, std::string>();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    return new CrossDeviceCopyProp();
  }

  std::string TypeString() const override {
    return "_CrossDeviceCopy";
  }

  Operator* CreateOperator(Context ctx) const override {
    return new CrossDeviceCopyOp();
  }
};


MXNET_REGISTER_OP_PROPERTY(_CrossDeviceCopy, CrossDeviceCopyProp)
.describe("Special op to copy data cross device");
}  // namespace op
}  // namespace mxnet