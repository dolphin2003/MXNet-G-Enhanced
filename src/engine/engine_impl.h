
/*!
 * Copyright (c) 2015 by Contributors
 * \file engine_impl.h
 * \brief Internal implementation header of engine components.
 */
#ifndef MXNET_ENGINE_ENGINE_IMPL_H_
#define MXNET_ENGINE_ENGINE_IMPL_H_

#include <mxnet/engine.h>

/*! \brief MACRO on whether or not enable debug option*/
#define ENGINE_DEBUG 0

namespace mxnet {
namespace engine {

/*! \brief base class of engine variables, used for type checking */
struct Var {
#if ENGINE_DEBUG
  virtual ~Var() = default;
#endif  // ENGINE_DEBUG
  /*!
   * \brief cast variable to derived type T
   * \tparam T the type we want to cast into.
   * \return A casted variable.
   */
  template <typename T>
  inline T* Cast();
};  // struct Var

/*! \brief base class of engine operators, used for type checking */
struct Opr {
#if ENGINE_DEBUG
  virtual ~Opr() = default;
#endif
  /*!
   * \brief cast variable to derived type T
   * \tparam T the type we want to cast into.
   * \return A casted variable.
   */
  template <typename T>
  inline T* Cast();
};  // struct Opr

// implementation of the inline functions
template <typename T>
inline T* Var::Cast() {
  static_assert(std::is_base_of<Var, T>::value,
                "must inherit `mxnet::engine::Var`");
#if ENGINE_DEBUG
  return dynamic_cast<T*>(this);
#else
  return static_cast<T*>(this);
#endif
}

template <typename T>
inline T* Opr::Cast() {
  static_assert(std::is_base_of<Opr, T>::value,
                "must inherit `mxnet::engine::Opr`");
#if ENGINE_DEBUG
  return dynamic_cast<T*>(this);
#else
  return static_cast<T*>(this);
#endif
}

/*! \brief Maximum number of GPUs */
static constexpr std::size_t kMaxNumGPUs = 16;

// predeclare factory function for each type of engine
/*! \return NaiveEngine instance */
Engine *CreateNaiveEngine();
#if MXNET_PREDICT_ONLY == 0
/*! \return ThreadedEnginePooled instance */
Engine *CreateThreadedEnginePooled();
/*! \return ThreadedEnginePerDevie instance */
Engine *CreateThreadedEnginePerDevice();
#endif
}  // namespace engine
}  // namespace mxnet
#endif  // MXNET_ENGINE_ENGINE_IMPL_H_