/*!
 *  Copyright (c) 2016 by Contributors
 * \file complex.h
 * \brief support for complex operations
 * \author Xingjian Shi
 */
#ifndef MSHADOW_EXTENSION_COMPLEX_H_
#define MSHADOW_EXTENSION_COMPLEX_H_
#include <algorithm>
#include "../extension.h"

namespace mshadow {
namespace op {
namespace complex {
enum BinaryCalculationType { kBinaryCC, kBinaryCR, kBinaryRC};
enum UnitaryCalculationType { kUnitaryC2R, kUnitaryC2C };

struct mul {
  /*! \brief map a_real, a_imag, b_real, b_imag to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType RealMap(DType a_real, DType a_imag,
    DType b_real, DType b_imag) {
    return a_real * b_real - a_imag * b_imag;
  }
  template<typename DType>
  MSHADOW_XINLINE static DType ImagMap(DType a_real, DType a_imag,
    DType b_real, DType b_imag) {
    return a_real * b_imag + b_real * a_imag;
  }
};

struct div {
  /*! \brief map a_real, a_imag, b_real, b_imag to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType RealMap(DType a_real, DType a_imag,
    DType b_real, DType b_imag) {
    return (a_real * b_real + a_imag * b_imag) / (b_real * b_real + b_imag * b_imag);
  }
  template<typename DType>
  MSHADOW_XINLINE static DType ImagMap(DType a_real, DType a_imag,
    DType b_real, DType b_imag) {
    return (b_real * a_imag - a_real * b_imag) / (b_real * b_real + b_imag * b_imag);
  }
};

struct conjugate {
  template<typename TA, typename DType>
  MSHADOW_XINLINE static DType RealMap(const expr::Plan<TA, DType> &src_,
    index_t real_i, index_t real_j, index_t imag_i, index_t imag_j) {
    return src_.Eval(real_i, real_j);
  }
  template<typename TA, typename DType>
  MSHADOW_XINLINE static DType ImagMap(const expr::Plan<TA, DType> &src_,
    index_t real_i, index_t real_j, index_t imag_i, index_t imag_j) {
    return -src_.Eval(imag_i, imag_j);
  }
};

struct exchange {
  template<typename TA, typename DType>
  MSHADOW_XINLINE static DType RealMap(const expr::Plan<TA, DType> &src_,
    index_t real_i, index_t real_j, index_t imag_i, index_t imag_j) {
    return src_.Eval(imag_i, imag_j);
  }
  template<typename TA, typename DType>
  MSHADOW_XINLINE static DType ImagMap(const expr::Plan<TA, DType> &src_,
    index_t real_i, index_t real_j, index_t imag_i, index_t imag_j) {
    return src_.Eval(real_i, real_j);
  }
};

struct abs_square {
  template<typename TA, typename DType>
  MSHADOW_XINLINE static DType RealMap(const expr::Plan<TA, DType> &src_,
    index_t real_i, index_t real_j, index_t imag_i, index_t imag_j) {
    DType real_val = src_.Eval(real_i, real_j);
    DType image_val = src_.Eval(imag_i, imag_j);
    return real_val * real_val + image_val * image_val;
  }
};

struct sum_real_imag {
  template<typename TA, typename DType>
  MSHADOW_XINLINE static DType RealMap(const expr::Plan<TA, DType> &src_,
    index_t real_i, index_t real_j, index_t imag_i, index_t imag_j) {
    DType real_val = src_.Eval(real_i, real_j);
    DType image_val = src_.Eval(imag_i, imag_j);
    return real_val + image_val;
  }
};
}  // namespace complex
}  // namespace op

namespace expr {
//--------------------
// ComplexBinaryMapExp
//--------------------
  /*!
* \brief binary map expression lhs [op] rhs where lhs and rhs are complex tensors
* \tparam OP operator
* \tparam calctype type of the calculation
* \tparam TA type of lhs
* \tparam TB type of rhs
* \tparam etype expression type, sa namespace::type
*/
template<int calctype, typename OP, typename TA, typename TB, typename DType, int etype>
struct ComplexBinaryMapExp : public Exp<ComplexBinaryMapExp<calctype, OP, TA, TB, DType, etype>,
  DType, etype> {
  /*! \brief left operand */
  const TA &lhs_;
  /*! \brief right operand */
  const TB &rhs_;
  /*! \brief constructor */
  explicit ComplexBinaryMapExp(const TA &lhs, const TB &rhs)
    :lhs_(lhs), rhs_(rhs) {}
};

//-------------------
// ComplexConjExp
//-------------------
/*!
* \brief compute conj(src) where src is a complex tensor
* \tparam TA type of src
* \tparam etype expression type, sa namespace::type
*/
template<int calctype, typename OP, typename TA, typename DType, int etype>
struct ComplexUnitaryExp : public Exp<ComplexUnitaryExp<calctype, OP, TA, DType, etype>,
  DType, etype> {
  /*! \brief source expression */
  const TA &src_;
  /*! \brief constructor */
  explicit ComplexUnitaryExp(const TA &src) : src_(src) {}
};



template<int calctype, typename OP, typename TA, typename TB, typename DType, int ta, int tb>
inline ComplexBinaryMapExp<calctype, OP, TA, TB, DType, (ta | tb | type::kMapper)>
ComplexF(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, tb> &rhs) {
  return ComplexBinaryMapExp<calctype, OP, TA, TB, DType,
    (ta | tb | type::kMapper)>(lhs.self(), rhs.self());
}

/*!
* \brief conj Negation the imaginary part of A where A is a complex tensor
* \param src source tensor
* \tparam e1 type of source expression
*/
template<int calctype, typename OP, typename SrcExp, typename DType, int e1>
inline ComplexUnitaryExp<calctype, OP, SrcExp, DType, (e1 | type::kMapper)>
ComplexF(const Exp<SrcExp, DType, e1> &src) {
  return ComplexUnitaryExp<calctype, OP, SrcExp, DType, (e1 | type::kMapper)>(src.self());
}

/*!
* \brief complex_mul_cc Complex multipilication two complex tensors, A * B
*/
template<typename TA, typename TB, typename DType, int ta, int tb>
inline ComplexBinaryMapExp<op::complex::kBinaryCC, op::complex::mul,
  TA, TB, DType, (ta | tb | type::kMapper)>
complex_mul_cc(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, tb> &rhs) {
  return ComplexF<op::complex::kBinaryCC, op::complex::mul>(lhs, rhs);
}

/*!
* \brief complex_mul_cr Complex multipilication a complex tensor A and a real tensor B
*/
template<typename TA, typename TB, typename DType, int ta, int tb>
inline ComplexBinaryMapExp<op::complex::kBinaryCR, op::complex::mul,
  TA, TB, DType, (ta | tb | type::kMapper)>
complex_mul_cr(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, tb> &rhs) {
  return ComplexF<op::complex::kBinaryCR, op::complex::mul>(lhs, rhs);
}

/*!
* \brief complex_mul_rc Complex multipilication of a real tensor B and a complex tensor A
*/
template<typename TA, typename TB, typename DType, int ta, int tb>
inline ComplexBinaryMapExp<op::complex::kBinaryRC, op::complex::mul,
  TA, TB, DType, (ta | tb | type::kMapper)>
complex_mul_rc(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, t