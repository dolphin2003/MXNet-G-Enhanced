/*!
 * Copyright (c) 2015 by Contributors
 * \file kvstore_device.h
 * \brief Device implementation of KVStore that do reduction on GPU reduction.
 */
#ifndef MXNET_KVSTORE_KVSTORE_DEVICE_H_
#define MXNET_KVSTORE_KVSTORE_DEVICE_H_

#include <mxnet/kvstore.h>
#include <unordered_map>
#include <vector>
#include <utility>
#include <algorithm>
#include <limits>
#include "./kvstore_local.h"
#include "../common/utils.h"

namespace mxnet {
namespace kvstore {
/*!
 * \brief Device implementation of KVStore that do reduction on GPU reduction.
 */
class KVStoreDevice : public KVStoreLocal {
 protected:
  using KeyShape = std::pair<int, TShape>;
  void Init(const std::vector<int>& keys,
            const std::vector<NDArray>& values) override {
    KVStoreLocal::Init(keys, values);

    for (size_t i = 0; i < keys.size(); ++i) {
      sorted_key_shape_.push_back(std::make_pair(keys[i], values[i].shape()));
    }
  }

  void InitMergeBuffers(const std::vector<NDArray>& val) {
    std::sort(sorted_key_shape_.begin(), sorted_key_shape_.end(), [](
              const KeyShape& a, const KeyShape& b) {
      return a.second.Size() > b.second.Size();
    });

    CHECK(!val.empty());
    std::unordered_map<int, std::pair<Context, size_t>> ctx_info;
    for (size_t i = 0; i < val.size(); ++i) {
      int32_t dev_id = val[i].ctx().dev_id