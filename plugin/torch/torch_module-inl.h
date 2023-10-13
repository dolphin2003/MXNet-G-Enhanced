/*!
 * Copyright (c) 2015 by Contributors
 * \file torch_module-inl.h
 * \brief torch module operator
 * \author Min Lin
*/
#ifndef PLUGIN_TORCH_TORCH_MODULE_INL_H_
#define PLUGIN_TORCH_TORCH_MODULE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <stdio.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../../src/operator/operator_common.h"
#include "./torch_base.h"

namespace mxnet {
namespace op {
struct TorchModuleParam : public dmlc::Parameter<TorchModuleParam> {
  std::string lua_string;
  uint32_t num_data;
  uint32_t num_params;
  uint32_t num_outputs;
  DMLC_DECLARE_PARAMETER(TorchModuleParam) {
    DMLC_DECLARE_FIELD(lua_string)
    .describe("lua string that is called to generate the torch module object");
    DMLC_DECLARE_FIELD(num_data)
    .describe("the number of input data");
    DMLC_DECLARE_FIELD(num_params)
    .describe("the number of parameters");
    DMLC_DECLARE_FIELD(num_outputs)
    .describe("the number of outputs");
  }
};

/**
 * \brief This is the implementation of activation operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu>
class TorchModuleOp : public Operator {
 private:
  TorchModuleParam param_;
  TorchState* torchState_;
  int lua_reference_;

 public:
  explicit TorchModuleOp(TorchModuleParam p, TorchState* torchState) : torchState_(torchState) {
    this->param_ = p;
    lua_State* L = torchState_->L;
    CHECK_EQ(lua_gettop(L), 0);
    std::string exec = std::string("return ") + p.lua_string
      + TorchTensor::ModuleType(xpu::kDevMask);
    CHECK_EQ(luaL_loadstring(L, exec.c_str()), 0);
    int err = lua_pcall(L, 0, 1, 0);
    CHECK_EQ(err, 0) << lua_tostring(L, -1);
    // Get number of parameters
    uint32_t param_num = 0;
    lua_getfield(L, -1, "parameters");
    lua_pushvalue(L, -2);
    CHECK_EQ(lua_pcall(L, 1, LUA_MULTRET, 0), 0);
    if (lua_gettop(L) == 1) {
      param_num = 0;
    } else {
      CHECK_EQ(lua_gettop(L), 3);
      param_num = lua_objlen(L, -2);
      lua_pop(L, 2);
    }
    CHECK_EQ(param_num, param_.num_params);
    // Free the parameters allocated by torch so it doesn't take up memory.
    if (param_.num_params != 0) {
      // get the parameters into the stack
      lua_getfield(L, -1, "parameters");
      lua_pushvalue(L, -2);
      int err = lua_pcall(L, 1, 1, 0);
      CHECK_EQ(err, 0);
      // iterate the parameters table to free tblobs inside
      lua_pushnil(L);
      while (lua_next(L, -2)) {
        CHECK(luaT_isudata(L, -1, TorchTensor::TensorType(xpu::kDevMask)));
        void* udata = luaT_toudata(L, -1, TorchTensor::TensorType(xpu::kDevMask));
        TorchTensor::FreeInternal(torchState_, static_cast<THGeneralTensor>(udata), xpu::kDevMask);
        lua_pop(L, 1);
      }
      lua_pop(L, 1);  // pop the parameter table
    }
    this->lua_reference_ = luaL_ref(L, LUA_REGISTRYINDEX);
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    lua_State* L = torchState_->L;

    CHECK_EQ(lua_gettop(L), 0);
    CHECK_EQ(in_data.size(), param_.num_params + param_.num_data);
    CHECK_EQ(out_data.size(), param_.num_outputs);
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    torchState_->SetStream(s);
    // Deserialize self table

    lua_rawgeti(L, LUA_REGISTRYINDEX, lua_reference_);

    std::vector<THGeneralTensor> th_output =
      TorchTensor::TBlobVectorAsTable(torchState_, out_data.begin(),
                                      out_data.begin() + param_.num_outputs);
    // set the output field
    lua_setfield(L, -2, "output");
    // set the parameters
    if (param_.num_params != 0) {
      // get the parameters into the stack
      lua_getfield(L, -1, "parameters");
      lua_pushvalue(L, -2);
      int err = lua_pcall(L, 1, 1, 0);
      CHECK_EQ(err, 0);
      // iterate the parameters table to put tblobs inside
      lua_pushnil(L);
      std::vector<TBlob>::const_iterator it = in_data.begin() + param_.num_data;
      while (lua_next(L, -2)) {
        CHECK(luaT_isudata(L, -1, TorchTensor::TensorType(*it)));
        void* udata = luaT_toudata(L, -1, TorchTensor::TensorType(*it));
        TorchTensor::SetInternal(torchState_, static_cast<THGeneralTensor>(udata), *(it));
        it++;
        lua_pop(L, 1);
      }
      lua_pop(L, 1);  // pop the parameter table
    }
    // call updateOutput
    // | self
    lua_getfield(L, -1, "updateOutput");
    // | self | updateOutput
    lua_pushvalue(L, -2);
    // | se