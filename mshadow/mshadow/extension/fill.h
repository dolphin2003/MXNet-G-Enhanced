/*!
 *  Copyright (c) 2015 by Contributors
 * \file fill.h
 * \brief support for implicit array filling operation
 * \author Xingjian Shi
 */
#ifndef MSHADOW_EXTENSION_FILL_H_
#define MSHADOW_EXTENSION_FILL_H_

#include "../extension.h"


namespace mshadow {
namespace expr {
/*!
 * \brief Set value of a specific element in each line of the data matrix.
 * \tparam SrcExp type of src expression
 * \tparam ValExp type of val expression
 * \tparam IndexExp type of index expression
 * \tparam DType the type of ret expression
 */
template<typename SrcExp, typename ValExp, typename IndexExp, typename DType>
struct MatFillRowElementExp:
      public Exp<MatFillRowElementExp<SrcExp, ValExp, IndexExp, DType>,
                 DType, type::kChainer> {
  /*! \brief src operand */
  const SrcExp &src_;
  const ValExp &val_;
  /*! \brief index operand */
  const IndexExp &index_;
  /*! \brief constructor */
  MatFillRowElementExp(const SrcExp &src, const ValExp &val, const IndexExp &index)
      : src_(src), val_(val), index_(index) {}
};

template<typename SrcExp, typename ValExp, typename IndexExp,
        typename SDType, typename VDType, typename IDType, int e1, int e2, int e3>
inline MatFillRowElementExp<SrcExp, ValExp, IndexExp, SDType>
mat_fill_row_element(const Exp<SrcExp, SDType, e1> &src,
                     const Exp<ValExp, VDType, e2> &val,
                     const Exp<IndexExp, IDType, e3> &index) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim == 2 && ExpInfo<ValExp>::kDim == 1
                && ExpInfo<IndexExp>::kDim == 1>::Error_Expression_Does_Not_Meet_Dimension_Req();
  return MatFillRowElementExp<SrcExp, ValExp, IndexExp, SDType>(src.self(),
                                                                val.self(), index.self());
}

//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename ValExp, typename IndexExp, typename DType>
struct Plan<MatFillRowElementExp<SrcExp, ValExp, IndexExp, DType>, DType> {
 public:
  explicit Plan(const MatFillRowElementExp<SrcExp, ValExp, IndexExp, DType> &e)
      : src_(MakePlan(e.src_)),
        val_(MakePlan(e.val_)),
        index_(MakePlan(e.index_)) {
  }
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    index_t idx = static_cast<index_t>(index_.Eval(0, y));
    if (idx == x) {
      return static_cast<DType>(val_.Eval(0, y));
    } else {
      return static_cast<DType>(src_.Eval(y, x));
    }
  }

 private:
  expr::Plan<SrcExp, DType> src_;
  expr::Plan<ValExp, DType> val_;
  expr::Plan<IndexExp,