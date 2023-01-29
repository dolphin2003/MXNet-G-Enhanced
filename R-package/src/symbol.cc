/*!
 *  Copyright (c) 2015 by Contributors
 * \file symbol.cc
 * \brief Rcpp Symbol of MXNet.
 */
#include <Rcpp.h>
#include <string>
#include <algorithm>
#include "./base.h"
#include "./symbol.h"
#include "./name.h"
#include "./export.h"

namespace mxnet {
namespace R {

NameManager* NameManager::Get() {
  static NameManager inst;
  return &inst;
}

inline Symbol::RObjectType Symbol::Clone() const {
  SymbolHandle ohandle;
  MX_CALL(MXSymbolCopy(handle_, &ohandle));
  return Symbol::RObject(ohandle);
}

Symbol::RObjectType Symbol::Apply(const Rcpp::List& kwargs) const {
  RObjectType ret = this->Clone();
  if (kwargs.containsElementNamed("name")) {
    int index = kwargs.findName("name");
    std::string name = kwargs[index];
    Rcpp::List kw(kwargs);
    kw.erase(index);
    Symbol::XPtr(ret)->Compose(kw, name);
  } else {
    std::string name;
    Symbol::XPtr(ret)->Compose(kwargs, name);
  }
  return ret;
}

std::string Symbol::DebugStr() const {
  const char *str;
  MX_CALL(MXSymbolPrint(handle_, &str));
  return str;
}

void Symbol::Compose(const Rcpp::List& kwargs, const std::string &name) {
  std::string target_name;
  std::vector<std::string> keys = SafeGetListNames(kwargs);
  // get names
  bool positional = keys.size() == 0 || keys[0].length() == 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    RCHECK((keys[i].length() == 0) == positional)
        << "Input symbols need to be either positional or key=value style, not both\n";
  }
  if (positional) keys.resize(0);

  // string parameter keys
  std::vector<const char*> c_keys = CKeys(keys);
  // string parameter values
  std::vector<SymbolHandle> handles(kwargs.size());
  for (size_t i = 0; i < kwargs.size(); ++i) {
    handles[i] = Symbol::XPtr(kwargs[i])->handle_;
  }
  MX_CALL(MXSymbolCompose(
      handle_, name.c_str(),
      static_cast<mx_uint>(handles.size()),
      dmlc::BeginPtr(c_keys), dmlc::BeginPtr(handles)));
}

std::vector<std::string> Symbol::ListArguments() const {
  mx_uint size;
  const char **ret;
  MX_CALL(MXSymbolListArguments(handle_, &size, &ret));
  return std::vector<std::string>(ret, ret + size);
}

std::vector<std::string> Symbol::ListAuxiliaryStates() const {
  mx_uint size;
  const char **ret;
  MX_CALL(MXSymbolListAuxiliaryStates(handle_, &size, &ret));
  return std::vector<std::string>(ret, ret + size);
}

std::vector<std::string> Symbol::ListOuputs() const {
  mx_uint size;
  const char **ret;
  MX_CALL(MXSymbolListOutputs(handle_, &size, &ret));
  return std::vector<std::string>(ret, ret + size);
}

void Symbol::Save(const std::string& fname) const {
  MX_CALL(MXSymbolSaveToFile(handle_, fname.c_str()));
}

std::string Symbol::AsJSON() const {
  con