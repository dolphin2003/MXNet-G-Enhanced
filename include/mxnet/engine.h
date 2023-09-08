/*!
 * Copyright (c) 2015 by Contributors
 * \file engine.h
 * \brief Engine that schedules all the operations according to dependency.
 */
#ifndef MXNET_ENGINE_H_
#define MXNET_ENGINE_H_

#include <dmlc/base.h>
#if DMLC_USE_CXX11
#include <memory>
#include <functional>
#endif
#include <vector>
#include "./base.h"

namespace mxnet {

// forward declare engine
class Engine;

/*! \brief namespace of engine internal types. */
namespace engine {
/*! \brief Internal representation of variable. */
struct Var;
/*! \brief Internal representation of operator.  */
struct Opr;
/*! \brief Variable pointer type, usually hold by user used to specify dependencies. */
typedef Var* VarHandle;
/*! \brief Operator pointer type, usually hold by user.*/
typedef Opr* OprHandle;
/*!
 * \brief OnComplete Callback to the engin