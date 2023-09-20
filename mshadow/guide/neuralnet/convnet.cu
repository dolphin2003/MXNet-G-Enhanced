// this implements a simple convolution neural net: conv-maxpool-fullc
#include <vector>
// header file to use mshadow
#include "mshadow/tensor.h"
// helper function to load mnist dataset
#include "util.h"
// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

// define operations
struct relu{
  MSHADOW_XINLINE static real_t Map(real_t a) {
    using namespace std;
    return max(a, 0.0f);
  }
};
struct relu_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a > 0.0f ? 1.0f : 0.0f;
  }
};

/*! \brief interface for nnet, interfacd allows use to use GPU/CPU implementation in a unified way */
class INNet{
 public:
  virtual void Forward(const Tensor<cpu, 4, real_t>& inbatch, Tensor<cpu, 2, real_t> &oubatch) = 0;
  virtual void Backprop(const Tensor<cpu, 2, real_t>& gradout) = 0;
  virtual void Update(void) = 0;
  virtual ~INNet() {}
};

/*!
 * \brief simple two layer conv-net conv-pool-flat-fullc
 *        this implementation is device invariant
 */
template<typename xpu>
class ConvNet : public INNet {
 public:
  // initialize the network
  ConvNet(int batch_size, int insize, int nchannel, int ksize, int kstride, int psize, int num_out)
      :rnd(0), ksize(ksize), kstride(kstride), psize(psize) {
    // setup stream
    Stream<xpu> *stream = NewStream<xpu>();
    ninput.set_stream(stream);
    nhidden.set_stream(stream);
    nhiddenbak.set_stream(stream);
    npool.set_stream(stream);
    npoolbak.set_stream(stream);
    nflat.set_stream(stream);
    nout.set_stream(stream);
    hbias.set_stream(stream); g_hbias.set_stream(stream);
    obias.set_stream(stream);  g_obias.set_stream(stream);
    Ki2h.set_stream(stream);  g_Ki2h.set_stream(stream);
    Wh2o.set_stream(stream);   g_Wh2o.set_stream(stream);
    tmp_col.set_stream(stream);
    tmp_dst.set_stream(stream);
    // setup nodes
    ninput.Resize(Shape4(batch_size, 1, insize, insize));
    nhidden.Resize(Shape4(batch_size, nchannel, (insize - ksize)/kstride+1, (insize -ksize)/kstride+1));
    nhiddenbak.Resize(nhidden.shape_);
    npool.Resize(Shape4(batch_size, nchannel, (nhidden.size(2)+1-psize)/psize, (nhidden.size(3)+1-psize)/psize));
    npoolbak.Resize(npool.shape_);
    nflat.Resize(Shape2(batch_size, npool.size(1)*npool.size(2)*npool.size(3)));
    nout.Resize(Shape2(batch_size, num_out));
    // setup bias
    hbias.Resize(Shape1(nchannel)); g_hbias.Resize(hbias.shape_);
    obias.Resize(Shape1(num_out));  g_obias.Resize(obias.shape_);
    hbias = 0.0f; obias = 0.0f;
    // setup weights
    Ki2h.Resize(Shape2(nchannel, ksize*ksize));  g_Ki2h.Resize(Ki2h.shape_);
    Wh2o.Resize(Shape2(nflat.size(1), num_out));   g_Wh2o.Resize(Wh2o.shape_);
    rnd.SampleGaussian(&Ki2h, 0, 0.01f);
    rnd.SampleGaussian(&Wh2o, 0, 0.01f);

    printf("conv=%d, pool=%d\n", nhidden.size(3), npool.size(3));
 