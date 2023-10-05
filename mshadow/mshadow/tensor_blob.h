
/*!
 *  Copyright (c) 2014 by Contributors
 * \file tensor_blob.h
 * \brief TBlob class that holds common representation of
 *  arbirary dimension tensor, can be used to transformed
 *  to normal fixed dimenson tensor
 * \author Tianqi Chen
 */
#ifndef MSHADOW_TENSOR_BLOB_H_
#define MSHADOW_TENSOR_BLOB_H_
#include <vector>
#include <algorithm>
#include <iostream>
#include <cctype>
#include "./tensor.h"
#include "./logging.h"
namespace mshadow {
/*!
 * \brief dynamic shape class that can hold shape
 *   of arbirary dimension
 */
struct TShape {
 public:
  /*! \brief constructor */
  TShape()
      : ndim_(0),
        num_heap_allocated_(0),
        data_heap_(NULL) {}

  /*!
   * \brief construct an "all-one" TShape with given dimension
   * \param ndim the number of dimension of the shape
   */
  explicit TShape(index_t ndim)
      : ndim_(ndim) {
    if (ndim_ <= kStackCache) {
      data_heap_ = NULL;
      num_heap_allocated_ = 0;
      std::fill_n(data_stack_, ndim_, 1);
    } else {
      data_heap_ = new index_t[ndim_];
      num_heap_allocated_ = ndim_;
      std::fill_n(data_heap_, ndim_, 1);
    }
  }
  /*!
   * \brief constructor from TShape
   * \param s the source shape
   */
  TShape(const TShape &s)
      : ndim_(s.ndim_) {
    if (ndim_ <= kStackCache) {
      data_heap_ = NULL;
      num_heap_allocated_ = 0;
      std::copy(s.data_stack_, s.data_stack_ + ndim_, data_stack_);
    } else {
      data_heap_ = new index_t[ndim_];
      num_heap_allocated_ = ndim_;
      std::copy(s.data_heap_, s.data_heap_ + ndim_, data_heap_);
    }
  }
  /*!
   * \brief construct the TShape from content of iterator
   * \param begin the beginning of iterator
   * \param end end the end of the iterator
   * \tparam RandomAccessIterator iterator type
   */
  template<typename RandomAccessIterator>
  TShape(RandomAccessIterator begin,
         RandomAccessIterator end)
      : ndim_(0),
        num_heap_allocated_(0),
        data_heap_(NULL) {
    this->CopyFrom(begin, end);
  }
#if MSHADOW_IN_CXX11
  /*!
   * \brief move constructor from TShape
   * \param s the source shape
   */
  TShape(TShape &&s)
      : ndim_(s.ndim_),
        num_heap_allocated_(s.num_heap_allocated_),
        data_heap_(s.data_heap_) {
    if (ndim_ <= kStackCache) {
      std::copy(s.data_stack_, s.data_stack_ + ndim_, data_stack_);
    }
    // remove data heap space from s
    s.data_heap_ = NULL;
  }
  /*!
   * \brief move constructor from Shape
   * \param s the source shape
   */
  template<int dim>
  TShape(Shape<dim> &&s)  // NOLINT(*)
      : ndim_(0),
        num_heap_allocated_(0),
        data_heap_(NULL) {
    this->CopyFrom(s.shape_, s.shape_ + dim);
  }
#endif
  /*! \brief destructor */
  ~TShape() {
    // data_heap_ can be NULL
    delete [] data_heap_;
  }
  /*!
   * \brief copy shape from content betwen two iterators
   * \param begin the beginning of iterator
   * \param end the end of the iterator
   * \tparam RandomAccessIterator iterator type
   */
  template<typename RandomAccessIterator>
  inline void CopyFrom(RandomAccessIterator begin,
                       RandomAccessIterator end) {
    this->SetDim(end - begin);
    std::copy(begin, end, data());
  }
  /*!
   * \brief assignment from shape
   * \param shape source shape
   * \return reference of self
   */
  inline TShape &operator=(const TShape &shape) {
    this->SetDim(shape.ndim_);
    const index_t *src = shape.data();
    std::copy(src, src + ndim_, data());
    return *this;
  }
  /*!
   * \brief assignment from vector
   * \param shape source shape
   * \return reference of self
   */
  inline TShape &operator=(const std::vector<index_t> &shape) {
    this->CopyFrom(shape.begin(), shape.end());
    return *this;
  }
  /*!
   * \brief assignment from shape
   * \param shape source shape
   * \tparam dim shape dimension
   * \return reference of self
   */
  template<int dim>
  inline TShape &operator=(const Shape<dim> &shape) {
    this->SetDim(dim);
    index_t *d = dim <= kStackCache ? data_stack_ : data_heap_;
    for (int i = 0; i < dim; ++i) {
      d[i] = shape[i];
    }
    return *this;
  }
  /*! \return the data content of the shape */
  inline const index_t *data() const {
    return ndim_ <= kStackCache ? data_stack_ : data_heap_;
  }
  /*! \return the data content of the shape */
  inline index_t *data() {
    return ndim_ <= kStackCache ? data_stack_ : data_heap_;
  }
  /*! \brief return number of dimension of the tensor inside */
  inline index_t ndim(void) const {
    return ndim_;
  }
  /*!
   * \brief get corresponding index
   * \param i dimension index
   * \return the corresponding dimension size
   */
  inline index_t &operator[](index_t i) {
    return data()[i];
  }
  /*!
   * \brief get corresponding index
   * \param i dimension index
   * \return the corresponding dimension size
   */
  inline const index_t &operator[](index_t i) const {
    return data()[i];
  }
  /*! \brief total number of elements in the tensor */
  inline size_t Size(void) const {
    size_t size = 1;
    const index_t *d = this->data();
    for (index_t i = 0; i < ndim_; ++i) {
      size *= d[i];
    }
    return size;
  }
  /*!
   * flatten the higher dimension to second dimension, return a 2D shape
   * \return the flat 2d shape
   */
  inline Shape<2> FlatTo2D(void) const {
    Shape<2> s;
    if (ndim_ == 0) return Shape2(0, 0);
    const index_t *d = this->data();
    s.shape_[1] = d[ndim_ - 1];
    index_t ymax = 1;
    for (index_t i = 1; i < ndim_; ++i) {
      ymax *= d[i - 1];
    }
    s.shape_[0] = ymax;
    return s;
  }
  /*!
  * flatten the shape into three parts: [0, axis_begin), [axis_begin, axis_end], (axis_end, ndim)
  * \param axis_begin The beginning axis specified.
  * \param axis_end The ending axis specified.
  * \return the flat 3d shape
  */
  inline Shape<3> FlatTo3D(index_t axis_begin, index_t axis_end) const {
    CHECK(axis_end >= axis_begin);
    Shape<3> s;
    if (ndim_ == 0) return Shape3(0, 0, 0);
    const index_t *d = this->data();
    s.shape_[0] = 1;
    s.shape_[1] = 1;
    s.shape_[2] = 1;

    for (index_t i = 0; i < axis_begin; ++i) {
      s.shape_[0] *= d[i];
    }
    for (index_t i = axis_begin; i <= axis_end; ++i) {
      s.shape_[1] *= d[i];
    }
    for (index_t i = axis_end + 1; i < ndim_; ++i) {
      s.shape_[2] *= d[i];
    }
    return s;
  }
  /*!
   * flatten the axis before and after the specified axis, so it becomes 3D tensor
   * \param axis The axis specified.
   * \return the flat 3d shape
   */
  inline Shape<3> FlatTo3D(index_t axis) const {
    return FlatTo3D(axis, axis);
  }
  /*!
   * \return product shape in [dimstart,dimend)
   * \param dimstart start dimension
   * \param dimend end dimension
   */
  inline index_t ProdShape(int dimstart, int dimend) const {
    index_t num = 1;
    const index_t *d = this->data();
    for (int i = dimstart; i < dimend; ++i) {
      num *= d[i];
    }
    return num;
  }
  /*!
   * \brief get the shape of tensor specifying dim
   * \return the shape requested
   * \tparam dim dimension of the tensor
   */
  template<int dim>
  inline Shape<dim> get(void) const {
    CHECK_EQ(dim, ndim_) << "dimension do not match target dimension " << dim << " vs " << ndim_;
    const index_t *d = this->data();
    Shape<dim> s;
    for (int i = 0; i < dim; ++i) {
      s[i] = d[i];
    }
    return s;
  }
  /*!
   * \return whether two shape equals
   * \param s the shape to compare against
   */
  inline bool operator==(const TShape &s) const {
    if (ndim_ != s.ndim_) return false;
    if (ndim_ <= kStackCache) {
      for (index_t i = 0; i < ndim_; ++i) {
        if (data_stack_[i] != s.data_stack_[i]) return false;
      }
    } else {
      for (index_t i = 0; i < ndim_; ++i) {
        if (data_heap_[i] != s.data_heap_[i]) return false;
      }
    }
    return true;
  }
  /*!
   * \return whether two shape not equals
   * \param s the shape to compare against
   */
  inline bool operator!=(const TShape &s) const {
    return !(*this == s);
  }
  /*!
   * \return whether two shape equals
   * \param s the shape to compare against
   * \tparam dim dimension of the shape
   */
  template<int dim>
  inline bool operator==(const Shape<dim> &s) const {
    if (ndim_ != dim) return false;
    const index_t *d = dim <= kStackCache ? data_stack_ : data_heap_;
    for (index_t i = 0; i < dim; ++i) {
      if (d[i] != s.shape_[i]) return false;
    }
    return true;
  }
  /*!
   * \return whether two shape not equals
   * \param s the shape to compare against
   * \tparam dim dimension of the shape
   */
  template<int dim>
  inline bool operator!=(const Shape<dim> &s) const {
    return !(*this == s);
  }
  /*!
   * \brief save the content into binary stream
   * \param strm the output stream
   * \tparam TStream any stream type that have write
   */
  template<typename TStream>
  inline void Save(TStream *strm) const {
    strm->Write(&ndim_, sizeof(ndim_));
    strm->Write(data(), sizeof(index_t) * ndim_);
  }
  /*!
   * \brief load the content from binary stream
   * \param strm the output stream
   * \tparam TStream any stream type that have write
   * \return whether the load is successful
   */
  template<typename TStream>
  inline bool Load(TStream *strm) {
    if (strm->Read(&ndim_, sizeof(ndim_)) != sizeof(ndim_)) return false;
    this->SetDim(ndim_);
    size_t nread = sizeof(index_t) * ndim_;
    if (strm->Read(data(), nread) != nread) return false;
    return true;
  }

  friend std::ostream &operator<<(std::ostream &os, const TShape &shape);
  friend std::istream &operator>>(std::istream &is, TShape &shape);

 private:
  // the shape will be stored in data_stack_
  // when dimension is smaller than kStackCache
  // when it is bigger, it will be stored in data_heap_;
  /*! \brief size of in stack space */
  static const index_t kStackCache = 4;
  /*! \brief number of dimnsion of the shape */
  index_t ndim_;
  /*! \brief number of cells allocated in data_heap_ */
  index_t num_heap_allocated_;
  /*! \brief in stack space used to store shape when it is small */
  index_t data_stack_[kStackCache];
  /*! \brief space to store shape when dimension is big*/
  index_t *data_heap_;
  /*!
   * \brief internal function to set the dimension
   * \param dim the dimension of the shape
   */
  inline void SetDim(index_t dim) {
    if (dim > kStackCache &&
        dim > num_heap_allocated_) {
      // data_heap_ can be NULL
      delete [] data_heap_;
      data_heap_ = new index_t[dim];
      num_heap_allocated_ = dim;
    }
    ndim_ = dim;
  }
};

/*!
 * \brief allow string printing of the shape
 * \param os the output stream
 * \param shape the shape
 * \return the ostream