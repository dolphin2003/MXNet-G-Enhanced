/**
 * Copyright (c) 2015 by Contributors
 * @file   kvstore_local.h
 * @brief  local implementation
 */
#ifndef MXNET_KVSTORE_KVSTORE_LOCAL_H_
#define MXNET_KVSTORE_KVSTORE_LOCAL_H_

#include <mxnet/kvstore.h>
#include <unordered_map>
#include <bitset>
#include <vector>
#include <utility>
#include <algorithm>
#include "./comm.h"

namespace mxnet {
namespace kvstore {
/**
 * \brief store data in local machine
 */
class KVStoreLocal : public KVStore {
 public:
  /*
   * \param use_device_comm
   */
  explicit KVStoreLocal(bool use_device_comm) : KVStore() {
    if (use_device_comm) {
      comm_ = new CommDevice();
    } else {
      comm_ = new CommCPU();
    }
    pinned_ctx_ = (MXNET_USE_CUDA != 0) ? Context::CPUPinned(0) : Co