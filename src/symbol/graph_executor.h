/*!
 * Copyright (c) 2015 by Contributors
 * \file graph_executor.h
 * \brief Executor to execute the Forward and Backward on Composition Graph.
*/
#ifndef MXNET_SYMBOL_GRAPH_EXECUTOR_H_
#define MXNET_SYMBOL_GRAPH_EXECUTOR_H_

#include <mxnet/c_api.h>
#include <mxnet/symbolic.h>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include "./static_graph.h"
#i