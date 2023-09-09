/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray.h
 * \brief NDArray interface that handles array arithematics.
 */
#ifndef MXNET_NDARRAY_H_
#define MXNET_NDARRAY_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>
#include <dmlc/registry.h>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "./base.h"
#include "./storage.h"
#include "./engine.h"

// check c++11
#if DMLC_USE_CXX11 == 0
#error "cxx11 was required for ndarray module"
#endif

namespace mxnet {
/*!
 * \brief ndarray interface
 */
class NDArray {
 public:
  /*! \brief default cosntructor */
  NDArray() {}
  /*!
   * \brief constructing a new dynamic NDArray
   * \param shape the shape of array
   * \param ctx context of NDArray
   * \param delay_alloc whether delay the allocation
   * \param dtype data type of this ndarray
   */
  NDArray(const TShape &shape, Context ctx,
          bool delay_alloc = false, int dtype = mshadow::default_type_flag)
      : ptr_(std::make_shared<Chunk>(shape.Size(), ctx, delay_alloc, dtype)),
        shape_(shape), offset_(0), dtype_(dtype) {
  }
  /*!
   * \brief constructing a static NDArray that shares data with TBlob
   *  Use with caution: allocate ONLY ONE NDArray for each TBlob,
   *  make sure the memory region is available through out the life of NDArray
   * \param data the memory content of static data
   * \param dev_id the device id this tensor sits at
   */
  NDArray(const TBlob &data, int dev_id)
      : ptr_(std::make_shared<Chunk>(data, dev_id)), shape_(data.shape_), offset_(0),
        dtype_(data.type_flag_) {
  }
  /*!
   * \return the shape of current NDArray
   */
  inline const TShape &shape() const {
    return shape_;
  }
  /*!
   * \return the data TBlob
   */
  inline TBlob data() const {
    TBlob res;
    MSHADOW_TYPE_SWITCH(dtype_, DType, {
      res = TBlob(static_cast<DType*>(ptr_->shandle.dptr)
        + offset_, shape_, ptr_->shandle.ctx.dev_mask());
    });
    return res;
  }
  /*!
   * \return a chunk of raw data in TBlob
   */
  inline TBlob raw_data(index_t offset, index_t length) const {
    TBlob res;
    TShape raw_shape(1);
    raw_shape[0] = length;
    MSHADOW_TYPE_SWITCH(dtype_, DType, {
      res = TBlob(static_cast<DType*>(ptr_->shandle.dptr)
        + offset_ + offset, raw_shape, ptr_->shandle.ctx.dev_mask());
    });
    return res;
  }
  /*!
   * \return the context of NDArray, this function is only valid when the NDArray is not empty
   */
  inline Context ctx() const {
    return ptr_->shandle.ctx;
  }
  /*!
   * \return the data type of NDArray, this function is only valid when the NDArray is not empty
   */
  inline int dtype() const {
    return dtype_;
  }
  /*! \return whether this ndarray is not initialized */
  inline bool is_none() const {
    return ptr_.get() == nullptr;
  }
  /*!
   * \brief Block until all the pending write operations with respect
   *    to current NDArray are finished, and read can be performed.
   */
  inline void WaitToRead() const {
    if (is_none()) return;
    Engine::Get()->WaitForVar(ptr_->var);
  }
  /*!
   * \brief Block until all the pending read/write operations with respect
   *    to current NDArray are finished, and write can be performed.
   */
  inline void WaitToWrite() const {
    if (is_none()) return;
    /*!
     * Push an empty mutable function to flush all preceding reads to the
     * variable.
     */
    Engine::Get()->PushSync([](RunContext) {}, Context{}, {}, {ptr_->var});
    Engine::Get()->WaitForVar(ptr_->var);
  }
  /*! \return the associated variable of the ndarray.*/
  inline Engine::VarHandle var() const {
    return ptr_->var;
  }
  /*!
   * \brief save the content into binary stream
   * \param strm the output stream
   */
  void Save(dmlc::Stream *strm) const;
  /*!
   * \brief load the content from binary stream
   * \param strm the output stream
   * \return whether the load is successful
   */
  bool Load(dmlc::Stream *strm);
  /*!
   * \brief set all the elements in ndarray to be scalar
   * \param scalar the scalar to set
   * \return reference of self
   */
  NDArray &operator=(real_t scalar);
  /*!
   * \brief elementwise add to current space
   *  this mutate the current NDArray
   * \param src the data to add
   * \return reference of self
   */
  NDArray &operator+=(const NDArray &src);
  /*!
   * \brief elementwise add to current space
   *  this mutate the current NDArray
   * \param src the data to add
   * \return reference of self
   */
  NDArray &operator+=(const real_t &src);
  /*!
   * \brief elementwise subtract from current ndarray
   * this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator-=(const NDArray &src);
  /*!
   * \brief elementwise subtract from current ndarray
   * this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator-=(const real_t &src);
  /*!
   * \brief elementwise multiplication to current ndarray
   *  this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator*=(const NDArray &src);
  /*!
   * \brief elementwise multiplication to current ndarray
   *  this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator*=(const real_t &src);
  /*!
   * \brief elementwise division from current ndarray
   *  this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator/=(const NDArray &src);
  /*!
   * \brief elementwise division from current ndarray
   *  this mutate the current NDArray
   * \param src the data to substract
   * \return reference of self
   */
  NDArray &operator/=(const real_t &src);
  /*!
   * \brief return transpose of current NDArray
   * \return a new transposed NDArray
   */
  NDArray T() const;
  /*!
   * \brief return a new copy this NDArray
   * \param ctx the new context of this NDArray
   * \return the new copy
   */
  NDArray Copy(Context ctx) const;
  /*!
   * \brief Do a synchronize copy from a continugous CPU memory region.
   *
   *  This function will call WaitToWrite before the copy is performed.
   *  This is useful to copy data from existing memory region that are
   *  not wrapped by NDArray(thus dependency not being tracked).
   *
   * \param data the data source to copy from.
   * \param size the size of the source array, in sizeof(DType) not raw btyes.
   */
  void SyncCopyFromCPU(const void *data, size_t size) const;
  /*!
   * \brief Do a synchronize copy to a continugous CPU memory region.
   *
   *  This function will call WaitToRead before the copy is performed.
   *  This is useful to copy data from existing memory region that are
   *  not wrapped by NDArray(thus dependency not being tracked).
   *
   * \param data the data source to copyinto.
   * \param size the 