/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray.cc
 * \brief ndarry module of mxnet
 */
#include <dmlc/io.h>
#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/resource.h>
#include <mshadow/tensor.h>
#include "./ndarray_function.h"

#if MXNET_USE_OPENCV
#include <opencv2/opencv.hpp>
#endif  // MXNET_USE_OPENCV

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::NDArrayFunctionReg);
}  // namespace dmlc

namespace mxnet {

/*!
* \brief run a ternary operation
* \param lhs left operand
* \param mhs middle operand
* \param rhs right operand
* \param out the output ndarray
*/
template<typename OP>
void TernaryOp(const NDArray &lhs,
  const NDArray &mhs,
  const NDArray &rhs,
  NDArray *out) {
  // no check if all of them are on cpu
  if (lhs.ctx().dev_mask() != cpu::kDevMask || mhs.ctx().dev_mask() != cpu::kDevMask
                                            || rhs.ctx().dev_mask() != cpu::kDevMask) {
    CHECK((lhs.ctx() == mhs.ctx()) && (mhs.ctx() == rhs.ctx())) << "operands context mismatch";
  }
  // if out is none, allocate space
  if (out->is_none()) {
    *out = NDArray(OP::GetShape(lhs.shape(), mhs.shape(), rhs.shape()), lhs.ctx(), true);
  } else {
    // no check if both of them are on cpu
    if (lhs.ctx().dev_mask() != cpu::kDevMask ||
      out->ctx().dev_mask() != cpu::kDevMask) {
      CHECK(out->ctx() == lhs.ctx()) << "target context mismatch";
    }
    CHECK(out->shape() == OP::GetShape(lhs.shape(), mhs.shape(), rhs.shape()))
      << "target shape mismatch";
  }
  // important: callback must always capture by value
  NDArray ret = *out;
  // get the const variables
  std::vector<Engine::VarHandle> const_vars;
  if (lhs.var() != ret.var()) const_vars.push_back(lhs.var());
  if (mhs.var() != ret.var()) const_vars.push_back(mhs.var());
  if (rhs.var() != ret.var()) const_vars.push_back(rhs.var());

  // redirect everything to mshadow operations
  switch (lhs.ctx().dev_mask()) {
  case cpu::kDevMask: {
    Engine::Get()->PushSync([lhs, mhs, rhs, ret](RunContext ctx) {
      ret.CheckAndAlloc();
      TBlob tmp = ret.data();
      ndarray::Eval<cpu, OP>(lhs.data(), mhs.data(), rhs.data(), &tmp, ctx);
    }, lhs.ctx(), const_vars, { ret.var() });
    break;
  }
#if MXNET_USE_CUDA
  case gpu::kDevMask: {
    Engine::Get()->PushSync([lhs, mhs, rhs, ret](RunContext ctx) {
      ret.CheckAndAlloc();
      TBlob tmp = ret.data();
      ndarray::Eval<gpu, OP>(lhs.data(), mhs.data(), rhs.data(), &tmp, ctx);
      // Wait GPU kernel to complete
      ctx.get_stream<gpu>()->Wait();
    }, lhs.ctx(), const_vars, { ret.var() });
    break;
  }
#endif
  default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
}

/*!
 * \brief run a binary operation
 * \param lhs left operand
 * \param rhs right operand
 * \param out the output ndarray
 * \param binary_op the real
 */
template<typename OP>
void BinaryOp(const NDArray &lhs,
              const NDArray &rhs,
              NDArray *out) {
  // no check if both of them are on cpu
  if (lhs.ctx().dev_mask() != cpu::kDevMask || rhs.ctx().dev_mask() != cpu::kDevMask) {
    CHECK(lhs.ctx() == rhs.ctx()) << "operands context mismatch";
  }
  // if out is none, allocate space
  if (out->is_none()) {
    *out = NDArray(OP::GetShape(lhs.shape(), rhs.shape()), lhs.ctx(), true, lhs.dtype());
  } else {
    // no check if both of them are on cpu
    if (lhs.ctx().dev_mask() != cpu::kDevMask ||
        out->ctx().dev_mask() != cpu::kDevMask) {
      CHECK(out->ctx() == lhs.ctx()) << "target context mismatch";
    }
    CHECK(out->shape() == OP::GetShape(lhs.shape(), rhs.shape()))
        << "target shape mismatch";
  }
  // important: callback must always capture by value
  NDArray ret = *out;
  // get the const variables
  std::vector<Engine::VarHandle> const_vars;
  if (lhs.var() != ret.var()) const_vars.push_back(lhs.var());
  if (rhs.var() != ret.var()) const_vars.push_back(rhs.var());

  // redirect everything to mshadow operations
  switch (lhs.ctx().dev_mask()) {
    case cpu::kDevMask: {
      Engine::Get()->PushSync([lhs, rhs, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::Eval<cpu, OP>(lhs.data(), rhs.data(), &tmp, ctx);
        }, lhs.ctx(), const_vars, {ret.var()});
      break;
    }
#if MXNET_USE_CUDA
    case gpu::kDevMask: {
      Engine::Get()->PushSync([lhs, rhs, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::Eval<gpu, OP>(lhs.data(), rhs.data(), &tmp, ctx);
          // Wait GPU kernel to complete
          ctx.get_stream<gpu>()->Wait();
        }, lhs.ctx(), const_vars, {ret.var()});
      break;
    }
#endif
    default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
}

void SetValueOp(const real_t &rhs, NDArray *out) {
  CHECK_NE(out->is_none(), true) << "Set value target must not be empty";
  // important: callback must always capture by value
  NDArray ret = *out;
  switch (ret.ctx().dev_mask()) {
    case cpu::kDevMask: {
      Engine::Get()->PushSync([rhs, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::Eval<cpu>(rhs, &tmp, ctx);
        }, ret.ctx(), {}, {ret.var()});
      break;
    }
#if MXNET_USE_CUDA
    case gpu::kDevMask: {
      Engine::Get()->PushSync([rhs, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::Eval<gpu>(rhs, &tmp, ctx);
          // Wait GPU kernel to complete
          ctx.get_stream<gpu>()->Wait();
        }, ret.ctx(), {}, {ret.var()});
      break;
    }
#endif
    default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
}

/*!
 * \brief run a binary operation
 * \param lhs left operand
 * \param rhs right operand
 * \param out the output ndarray
 * \param binary_op the real
 */
template<typename OP, bool reverse>
void ScalarOp(const NDArray &lhs,
              const real_t &rhs,
              NDArray *out) {
  if (out->is_none()) {
    *out = NDArray(lhs.shape(), lhs.ctx(), true, lhs.dtype());
  } else {
    CHECK(out->ctx() == lhs.ctx()) << "target context mismatch";
    CHECK(out->shape() == lhs.shape()) << "target shape mismatch";
  }
  // important: callback must always capture by value
  NDArray ret = *out;
  // get the const variables
  std::vector<Engine::VarHandle> const_vars;
  if (lhs.var() != ret.var()) const_vars.push_back(lhs.var());

  // redirect everything to mshadow operations
  switch (lhs.ctx().dev_mask()) {
    case cpu::kDevMask: {
      Engine::Get()->PushSync([lhs, rhs, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::Eval<cpu, OP, reverse>(lhs.data(), rhs, &tmp, ctx);
        }, lhs.ctx(), const_vars, {ret.var()});
      break;
    }
#if MXNET_USE_CUDA
    case gpu::kDevMask: {
      Engine::Get()->PushSync([lhs, rhs, ret](RunContext ctx) {
          ret.CheckAndAlloc();
          TBlob tmp = ret.data();
          ndarray::Eval<gpu, OP, reverse>(lhs.data(), rhs, &tmp, ctx);
          // Wait GPU kernel to complete
          ctx.get_stream<gpu>()->Wait();
        }, lhs.ctx(), const_vars, {ret.var()});
      break;
    }
#endif
    default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
}

void CopySliceTo(const NDArray &from, int slice_dim, index_t start, index_t end,
                 NDArray *to, int priority) {
  CHECK(from.shape().ndim() == to->shape().ndim())
      << "from and to must have the same number of dimensions";
  CHECK(slice_dim < from.shape().ndim())
      << "slice dimension out of bounds";
  CHECK(start < end)
      << "slice is empty";
  CHECK(end < from.shape()[slice_dim])
      << "slice out of bounds";

  mshadow::Shape<3> from_shape = from.shape().FlatTo3D(slice_dim);
  mshadow::Shape<3> to_shape = to->shape().FlatTo3D(slice_dim);
  CHECK(from_shape[0] == to_shape[0] && from_shape[2] == to_shape[2])
      << "shape incompatible";
  CHECK(end - start == to_shape[1])
      << "shape incompatible";

  int a = from.ctx().dev_mask();
  int b = to->ctx().dev_mask();

  std::vector<Engine::VarHandle> const_vars{from.var()};
  NDArray ret = *to;

#define MXNET_COPYSLICETO_IMPL(xpu1, xpu2) \
    Engine::Get()->PushSync([from, ret, from_shape, start, end](RunContext ctx) { \
      ret.CheckAndAlloc(); \
      for (index_t i = 0; i < from_shape[0]; ++i) { \
        index_t src_idx = i * (from_shape[1] * from_shape[2]) + \
            start * from_shape[2]; \
        index_t length = from_shape[2] * (end - start); \
        index_t dst_idx = i * length; \
        \
        TBlob blob_from = from.raw_data(src_idx, length); \
        TBlob blob_to = ret.raw_data(dst_idx, length); \
        ndarray::Copy<xpu1, xpu2>(blob_from, &blob_to, \
                                  from.ctx(), ret.ctx(), ctx); \
      } \
    }, from.ctx(), const_vars, {ret.var()}, \
    FnProperty::kNormal, priority)

  if (a == cpu::kDevMask && b == cpu::kDevMask) {
    MXNET_COPYSLICETO_IMPL(cpu, cpu);
  } else {
#if MXNET_USE_CUDA
    if (a == cpu::kDevMask && b == gpu::kDevMask) {
      MXNET_COPYSLICETO_IMPL(cpu, gpu);
    } else if (a == gpu::kDevMask && b == cpu::kDevMask) {
      MXNET_COPYSLICETO_IMPL(gpu, cpu);
    } else if (a == gpu::kDevMask && b == gpu::kDevMask) {
      MXNET_COPYSLICETO_IMPL(gpu, gpu);
    } else {
      LOG(FATAL) << "unknown device mask";
    }
#else
    LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
  }
}

void CopyFromTo(const NDArray &from, NDArray *to, int priority) {
  if (from.var() == to->var()) {
    // skip to copy to itself
    return;
  }
  CHECK(from.shape() == to->shape())
      << "operands shape mismatch";
  CHECK(from.shape().ndim() != 0)
      << "source operands have zero dimension shape";
  // important: callback must always capture by value
  NDArray ret = *to;
  int a = from.ctx().dev_mask();
  int b = to->ctx().dev_mask();

  std::vector<Engine::VarHandle> const_vars;
  if (from.var() != ret.var()) const_vars.push_back(from.var());

  if (a == cpu::kDevMask && b == cpu::kDevMask) {
    Engine::Get()->PushSync([from, ret](RunContext ctx) {
        ret.CheckAndAlloc();
        TBlob tmp = ret.data();
        ndarray::Copy<cpu, cpu>(from.data(), &tmp,
                                from.ctx(), ret.ctx(), ctx);
      }, from.ctx(), const_vars, {ret.var()},
      FnProperty::kNormal, priority)