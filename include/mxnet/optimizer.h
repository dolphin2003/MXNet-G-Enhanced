/*!
 *  Copyright (c) 2015 by Contributors
 * \file optimizer.h
 * \brief Operator interface of mxnet.
 * \author Junyuan Xie
 */
#ifndef MXNET_OPTIMIZER_H_
#define MXNET_OPTIMIZER_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <mshadow/tensor.h>
#include <string>
#include <vector>
#include <utility>
#i