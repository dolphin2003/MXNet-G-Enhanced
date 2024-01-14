/*!
 *  Copyright (c) 2016 by Contributors
 * \file elementwise_binary_broadcast_op-inl.h
 * \brief Function defintion of elementwise binary operators with broadcast
 *
 * For example,
 *
 *   [1, 2] + [1,  = [2, 3;
 *             2 ]    3, 4]
 *
 * The shapes broacast of the above example
 *
 *   A      (2d tensor):  1 x 2
 *   B      (1d tensor):  2 x 1
 *   Result (2d tensor):  2 x 2
 *
 * More examples
 *
 *   A      (3d tensor):  15 x 3 x 5
 *   B      (2d tensor):   1 x 1 x 5
 *   Result (3d tensor):  15 x 3 x 5
 *
 *   A      (3d tensor):  15 x 3 x 5
 *   B      (2d tensor):   1 x 3 x 1
 *   Result (3d tensor):  15 x 3 x 5
 *
 * Here are examples of shapes that do not broadcast:
 *
 *   A      (1d tensor):  3
 *   B      (1d tensor):  4 # trailing dimensions do not match
 *
 *   A      (2d tensor):  1 x 2 x 1
 *   B      (3d tensor):  8 x 4 x 3 # second from last dimensions mismatched
 *
 * When no broadcast is need, it falls back to elementwise_binary_op-inl.h
 */
#ifndef MXNET_OPERATOR_ELEMENTWISE_BINARY_BROADCAST_OP_INL_H_
#define MXNET_OPERATOR_ELEMENTWISE_BINARY_BROADCAST_OP_INL_H_

#include <mxnet/operator_util.h>
#include <algorithm>
#include <vector>
#include "./mshadow_op.h"
#include "./broadcast_reduce_op_common.h"

#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {

inline bool IsBroadcastNeeded_(const TShape& lhs,
                              const TShape& rhs) {
  // force ndim to be equal. do not smartly padding dims with 1s, which may confuse users
  CHECK_EQ(lhs.ndim(), rhs.ndim()) << "lhs:" << lhs << " rhs:" << rhs;
  for (index_t i = 0; i < lhs.ndim(); ++i) {
    if (lhs[i] != rhs[i]) return true;
  }
  return false;
}

inline TShape BinaryBroadcastShape_(const TShape& lhs,
                                    const TShape& rhs,
                                    const EnvArguments& env) {
  if (!IsBroadcastNeeded_(lhs, rhs)) return lhs;
  std::vector<index_t> ret(lhs.ndim());
  for (size_t i = 0; i < ret.size(); ++i) {
    ret[i] = std::max(lhs[i], rhs[i]);
  }
  return TShape(ret.begin(), ret.end());
}

inline void InferBroadcastNewShapes_(bool *do_opt,
  TShape *new_lhs_shape, TShape *new_rhs_shape, TShape *new_out_shape,
  const TShape &lhs_shape, const TShape &rhs_shape, const TShape &out_shape) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK((lhs_shape.ndim() == rhs_shape.ndim()) && (rhs_shape.ndim() == out_shape.ndim())) <<
    "ndim inconsistency, lhs_shape=" << lhs_shape << ", rhs_shape=" << rhs_shape <<
    ", out_shape=" << out_shape;
  *do_opt = false;
  TShape lhs_axes = GetBroadcastingAxes_(lhs_shape, out_shape);
  TShape rhs_axes = GetBroadcastingAxes_(rhs_shape, out_shape);
  bool lhs_contiguous, rhs_contiguous;
  index_t lhs_broadcasting_size, rhs_broadcasting_size;
  CheckContiguousAxes_(&lhs_contiguous, &lhs_broadcasting_size, lhs_axes, out_shape);
  CheckContiguousAxes_(&rhs_contiguous, &rhs_broadcasting_size, rhs_axes, out_shape);
  if (lhs_contiguous && rhs_contiguous && (lhs_axes.ndim() == 0 || rhs_axes.ndim() == 0)) {
    *do_opt = true;
    if (lhs_axes.ndim() == 0) {
      index_t leading =
        rhs_shape.ProdShape(0, rhs_axes[0]);
      index_t trailing =
        rhs_shape.ProdShape(rhs_axes[rhs_axes.ndim() - 1] + 1, rhs_shape.ndim());
      *new_lhs_shape = Shape3(leading, rhs_broadcasting_size, trailing);
      *new_rhs_shape = Shape3(leading, 1, trailing);
      *new_out_shape = Shape3(leading, rhs_broadcasting_size, trailing);
    } else {
      index_t leading =
        lhs_shape.ProdShape(0, lhs_axes[0]);
      index_t trailing =
        lhs_shape.ProdShape(lhs_axes[lhs_axes.ndim() - 1] + 1, lhs_shape.ndim());
      *new_lhs_shape = Shape3(leading, 1, trailing);
      *new_rhs_shape = Shape3(leading, lhs_broadcasting_size, trailing);
      *new_out_shape = Shape3(leading, lhs_broadcasting_size, trailing);
    }
  } else {
    *do_opt = false;
    CHECK(lhs_shape.ndim() <= MXNET_SPECIAL_MAX_NDIM)
      << "Only support input dimension up to " << MXNET_SPECIAL_MAX_NDIM
      << ", lhs_shape=" << lhs_shape << ", rhs_shape=" << rhs_shape
      << ", out_shape=" << out_shape;
    *new_lhs_shape = TShape(MXNET_SPECIAL_MAX_NDIM);
    *new_rhs_shape = TShape(MXNET_SPECIAL_MAX_NDIM);
    *new_out_shape = TShape(MXNET_SPECIAL_MAX_NDIM);
    for (index_t i = 0; i < lhs_shape.ndim(); i++) {
      (*new_lhs_shape)[i] = lhs_shape[i];
      (*new_rhs_shape)[i] = rhs_shape[i];
      (*new_out_shape)[i] = out_shape[i];
    }
  }
  CHECK(((*new_lhs_shape).Size() == lhs_shape.Size())
    && ((*new_rhs_shape).Size() == rhs_shape.Size())
    && ((*new_out_shape).Size() == out_shape.Size()))
    << "new_lhs_shape:" << *new_lhs_shape << ",lhs_shape:" << lhs_shape
    << "new_rhs_shape:" << *new_rhs_shape << ",rhs_shape:" << rhs_shape
    << "new_out_shape:" << *new_out_shape << ",out_shape:" << out_shape;
}

template<typename xpu, typename OP>
void BinaryBroadcastForward_(const TBlob& lhs,
                             const TBlob& rhs,
                             const EnvArguments& env,
                             TBlob *ret,
                             OpReqType req,
                             RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
    << "Binary function only support input/output with the same type";
  CHECK_EQ(ret->type_flag_, rhs.type_flag_)
    << "Binary function only support input/output with the same type";
  CHECK_EQ(lhs.shape_.ndim(), rhs.shape_.ndim()) << "the ndim of lhs and rhs must be equal,"
    " shape of lhs=" << lhs.shape_ << " shape of rhs=" << rhs.shape_;
  if (!IsBroadcastNeeded_(lhs.shape_, rhs.shape_)) {
    // no broadcast
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      mshadow::Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
      ASSIGN_DISPATCH(out, req,
        F<OP>(lhs.FlatTo2D<xpu, DType>(s),
        rhs.FlatTo2D<xpu, DType>(s)));
    });
    return;
  }
  bool do_opt;
  TShape lhs_new_shape_, rhs_new_shape_, out_new_shape_;
  InferBroadcastNewShapes_(&do_opt, &lhs_new_shape_, &rhs_new_shape_, &out_new_shape_,
    lhs.shape_, rhs.shape_, ret->shape_);
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    if (do_opt) {
      Shape<3> lhs_new_shape, rhs_new_shape, out_new_shape;
      for (index_t i = 0; i < 3; i++) {
        lhs_new_shape[i] = lhs_new_shape_[i];
        rhs_new_shape[i] = rhs_new_shape_[i];
        out_new_shape[i] = out_new_shape_[i];
      }
      Tensor<xpu, 3, DType> out = ret->get_with_shape<xpu, 3, DType>(out_new_shape, s);
      Tensor<xpu, 3, DType> mlhs = lhs.get_with_shape<xpu, 3, DType>(lhs_new_shape, s);
      Tensor<xpu, 3, DType> mrhs = rhs.get_with_shape<xpu, 3, DType>(rhs_new_shape, s);
      ASSIGN_DISPATCH(out, req,
        F<OP>(broadcast_to(mlhs, out_new_shape_), broadcast_to(mrhs, out_new_shape_)));
    } else {
      Shape<MXNET_SPECIAL_MAX_NDIM> lhs_new_shape, rhs_new_shape, out_new_shape;
      for (index_t i = 0; i < MXNET_SPECIAL_MAX_NDIM; i++) {
        lhs_new_shape[i] = lhs_new_shape_[i];
        rhs_new_shape[i] = rhs_new_shape_[i];
        out_new_shape[i] = out_new_shape_[i];
      }
      Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> out =
        ret->get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(out_new_shape, s);
      Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> mlhs =
        lhs.get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(lhs_new_shape, s);
      Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> mrhs =
        rhs.get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(rhs_new_shape, s);
      ASSIGN_DISPATCH(out, req,
        F<OP>(broadcast_to(mlhs, ou