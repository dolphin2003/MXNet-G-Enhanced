/*!
 *  Copyright (c) 2015 by Contributors
 * \file kvstore.h
 * \brief Rcpp Parameter Store interface of MXNet
 */
#ifndef MXNET_RCPP_KVSTORE_H_
#define MXNET_RCPP_KVSTORE_H_

#include <Rcpp.h>
#include <mxnet/c_api.h>
#include <string>
#include <vector>
#include <map>
#include "./base.h"

namespace mxnet {
namespace R {
/*!
 * \brief MXNet's Parameter store interface.
 */
class KVStore {
 public:
  /*!
   * \brief initialize all the weights
   * \param keys The keys of each weight.
   * \param weights the weights NDArray list.
   */
  void Init(const std::vector<int>& keys, const Rcpp::List& weights);
  /*!
   * \brief Push the weights to the KVStore.
   *
   *  This operation will do a aggregation first on weight_lists, the push things out.
   *
   *  sum_list[i] = sum(list[i] for list in weight_lists)
   *  Then push(keys[i], sum_list[i]) for each i.
   *
   * \param keys list of keys, corresponds to key of each location.
   * \param we