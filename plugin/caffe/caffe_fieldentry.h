/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_fieldentry.h
 * \brief Implement FieldEntry<caffe::LayerParameter>
 * \author Haoran Wang 
 */
#ifndef PLUGIN_CAFFE_CAFFE_FIELDENTRY_H_
#define PLUGIN_CAFFE_CAFFE_FIELDENTRY_H_

#include <caffe/proto/caffe.pb.h>
#include <dmlc/parameter.h>
#include <dmlc/base.h>
#include <dmlc/json.h>
#include <dmlc/logging.h>
#include <dmlc/type_traits.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <limits>
#include <map>
#include <set>
#include <typeinfo>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>

#include <caffe/util/io.hpp>
namespace dmlc {
namespace parameter {

// specialize define for Layer Parameter
template<>
class FieldEntry<caffe::LayerParameter>
    : public FieldEntryBase<FieldEntry<caffe::LayerParameter>, caffe::LayerParameter> {
 public:
  // parent class
  typedef FieldEntryBase<FieldEntry<caffe::LayerParameter>, caffe::LayerParameter> Parent;


  bool ReadProtoFromTextContent(const std::string& text,
                                ::google::protobuf::Message* proto) const {
    bool success = google::protobuf::TextFormat::ParseFromString(text, proto);
    return success;
  }

  /**
   * /brief Customize set method for LayerParameter
   * /tparam value string of caffe's layer configuration
   * */
  virtual void Set(void *head, co