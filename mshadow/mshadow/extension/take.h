/*!
 * Copyright (c) 2015 by Contributors
 * \file take.h
 * \brief
 * \author Bing Xu
*/
#ifndef MSHADOW_EXTENSION_TAKE_H_
#define MSHADOW_EXTENSION_TAKE_H_

#include "../extension.h"

namespace mshadow {
namespace expr {

/*! \brief Take a column from a matrix
 *  \tparam IndexExp type of index expression
 *  \tparam SrcExp type of src expression
 *  \tparam DType data type
 */
template<typename IndexExp, typename SrcExp, typename DType>
struct TakeExp: public Exp<TakeExp<IndexExp, SrcExp, DType>,
                           DType, type::kChainer> {
  /*! \brief index oprand */
  const IndexExp &index_;
  /*! \brief 