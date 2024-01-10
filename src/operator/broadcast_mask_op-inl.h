/*!
 * Copyright (c) 2015 by Contributors
 * \file broadcast_mask_op-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_BROADCAST_MASK_OP_INL_H_
#define MXNET_OPERATOR_BROADCAST_MASK_OP_INL_H_

#include <mxnet/operator_util.h>
#include "./operator_common.h"


#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {

inline TShape ElementwiseMaskShape_(const TShape& lhs,
                                    const TShape& rhs,
    