/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_api_error.h
 * \brief Error handling for C API.
 */
#ifndef MXNET_C_API_C_API_ERROR_H_
#define MXNET_C_API_C_API_ERROR_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <mxnet/c_api.h>

/*! \brief  macro to guard beginning and end section of all functions */
#define API_BEGIN() try {
/*! \brief every function starts with API_BEGIN();
     and finishes with API_END() or API_END_HANDLE_ERROR */
#define API_END() } catch(dmlc::Error &_except_) { return MXAPIHandleException(_except_); } return 0;  // NOLINT(*)
/*!
 * \brief every function starts with API_BEGI