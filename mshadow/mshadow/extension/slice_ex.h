/*!
 *  Copyright (c) 2014 by Contributors
 * \file slice.h
 * \brief support for slice a certain dimension.
 */
#ifndef MSHADOW_EXTENSION_SLICE_EX_H_
#define MSHADOW_EXTENSION_SLICE_EX_H_

#include "../extension.h"

namespace mshadow {
namespace expr {
/*!
 * \brief slice expression, slice a tensor's channel
 * \tparam SrcExp left expression
 * \tparam DType the type of elements
 * \tparam srcdim dimension of src
 * \tparam dimsrc_m_cat dimsrc - dimcat
 */
template<typename SrcExp, typename Device,
         typename DType, int srcdim>
struct SliceExExp : public TRValue<SliceExExp<SrcExp,
                                              Device, DType,
                                              srcdim>,
                                   Device, srcdim, DType> {
  const SrcExp &src_;
  Shape<srcdim> src_shape_;
  Shape<srcdim> shape_;
  const Shape<srcdim> begin_;
  const Shape<srcdim> end_;
  SliceExExp(const SrcExp &src, Shape<srcdim> begin, Shape<srcdim> end)
      : src_(src), begin_(begin), end_(end) {
    src_shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    for (int i = 0; i < srcdim; ++i) {
      shape_[i] = end_[i] - begin_[i];
    }
  }
  template<typename E, int etype>
  inline void
  operator=(const expr::Exp<E, DType, etype>