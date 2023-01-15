
/*!
 *  Copyright (c) 2015 by Contributors
 * \file executor.h
 * \brief Rcpp Symbolic Execution interface of MXNet
 */
#ifndef MXNET_RCPP_EXECUTOR_H_
#define MXNET_RCPP_EXECUTOR_H_

#include <Rcpp.h>
#include <mxnet/c_api.h>
#include <string>
#include "./base.h"
#include "./symbol.h"

namespace mxnet {
namespace R {
/*! \brief The Rcpp Symbol class of MXNet */
class Executor : public MXNetMovable<Executor> {
 public:
  /*! \return typename from R side. */
  inline static const char* TypeName() {
    return "MXExecutor";
  }
  /*!
   * \return Get reference of the arg arrays of executor.
   */
  const Rcpp::List& arg_arrays() const {
    return *arg_arrays_;
  }