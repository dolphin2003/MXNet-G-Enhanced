/*!
 *  Copyright (c) 2014 by Contributors
 * \file ps_dist-inl.h
 * \brief distributed version of PS
 *
 * \author Tianqi Chen, Mu Li
 */
#ifndef MSHADOW_PS_DIST_INL_H_ // NOLINT(*)
#define MSHADOW_PS_DIST_INL_H_ // NOLINT(*)

#include <vector>
#include "./mshadow_ps.h"
#include "./ps_local-inl.h"

#if MSHADOW_DIST_PS
#include "parameter/kv_layer.h"
namespace mshadow {
namespace ps {

/**
 * @brief bridge IModelUpdater to KVLayerUpdater
 */
template<typename DType>
class UpdaterWrapper {
 public:
  explicit UpdaterWrapper(IModelUpdater<DType> * updater)
      : updater_(updater) { }
  ~UpdaterWrapper() { delete updater_; }

  /// @brief initialize the data
  void Init(int id, size_t size, DType* data) {
    updater_->InitModel(id, data, size);
  }

  /// @brief update the model by using received data
  void Update(int id, size_t size, const DType* recv_data, DType* data) {
    updater_->Update(id, (DType*)recv_data, size);  // NOLINT(*)
  }
 private:
  IModelUpdater<DType> *updater_;
};


template<typename xpu, typename DType>
class DistModel : public LocalModel<xpu, DType> {
 public:
  // parent type
  typedef LocalModel<xpu, DType> Parent;

  // initialize the parameter server
  virtual void Init(const std::vector<int> &devices) {
    Parent::Init(devices);
    if (this->custom_server != NULL) {
      delete this->custom_server;
      this->custom_server = NULL;
    }
  }
  virtual ~DistModel(void) {
  }

 protected:
  // do nothing
  virtual void InitCustomerServer(void) {
  }
  virtual void ServerInitKey(Tensor<cpu, 2> weight, int key) {
    // th