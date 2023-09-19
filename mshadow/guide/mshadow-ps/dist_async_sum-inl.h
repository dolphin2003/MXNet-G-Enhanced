/**
 * @brief  Simple test of KVLayer
 */
#include "ps.h"
#include "parameter/kv_layer.h"
#include <cstdio>
#include <iostream>
#include <omp.h>
#include <map>
#include <mshadow/tensor.h>
#include <mshadow-ps/mshadow_ps.h>
#include "dbstr.h"
#include "glog/logging.h"

namespace mshadow {
namespace ps {


template<typename DType>
class Updater : public IModelUpdater<DType> {
 protected:
  void InitModel_(int key, Tensor<cpu, 1, DType> data) {
    data = 0;
    data_[key] = data;
  }

  void Update_(int key, Tensor<cpu, 1, DType> data) {
    data_[key] += data;
    // LOG(ERROR) << dbstr(data_[key]);
  }
  std::map<int, Tensor<cpu, 1, DType> > data_;
};

template<typename DType>
IModelUpdater<DType> *CreateModelUpdater(void) {
  return new Updater<DType>();
}

}  // namespace ps
}  // namespace mshadow

// this function is runed by specific thread
template<typename xpu>
inline void RunWorkerThread(int devid,
                            mshadow::ps::ISharedModel<xpu, float> *ps) {
  // initialize tensor engine
  mshadow::InitTensorEngine<xpu>(devid);
  mshadow::Stream<xpu> *stream  = mshadow::NewStream<xpu>();
  // allocate tensor on xpu
  mshadow::TensorContainer<xpu, 2> data(mshadow::Shape2(2, 3));
  // set the computation stream to the new allocated stream
  // this will make subsequent computation whose target is data
  // to use the stream, stream is needed for async execution in GPU
  data.set_stream(stream);
  // intiaialize the key, register the shape on parameter server
  ps