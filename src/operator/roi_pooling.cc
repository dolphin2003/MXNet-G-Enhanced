
/*!
 * Copyright (c) 2015 by Contributors
 * \file roi_pooling.cc
 * \brief roi pooling operator
 * \author Ross Girshick, Kye-Hyeon Kim, Jian Guo
*/
#include "./roi_pooling-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace mshadow {
template<typename Dtype>
inline void ROIPoolForward(const Tensor<cpu, 4, Dtype> &out,
                           const Tensor<cpu, 4, Dtype> &data,
                           const Tensor<cpu, 2, Dtype> &bbox,
                           const Tensor<cpu, 4, Dtype> &max_idx,
                           const float spatial_scale_) {
  const Dtype *bottom_data = data.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *top_data = out.dptr_;
  Dtype *argmax_data = max_idx.dptr_;
  const int channels_ = data.size(1);
  const int height_ = data.size(2);
  const int width_ = data.size(3);
  const int pooled_height_ = out.size(2);
  const int pooled_width_ = out.size(3);

  const int num_rois = bbox.size(0);
  const int batch_size = data.size(0);
  const int data_size = data.size(1) * data.size(2) * data.size(3);
  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale_);
    int roi_start_h = round(bottom_rois[2] * spatial_scale_);
    int roi_end_w = round(bottom_rois[3] * spatial_scale_);
    int roi_end_h = round(bottom_rois[4] * spatial_scale_);
    assert(roi_batch_ind >= 0);
    assert(roi_batch_ind < batch_size);

    // force malformed ROIs to be 1 * 1
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                             / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                             / static_cast<Dtype>(pooled_width_);

    const Dtype* batch_data = bottom_data + data_size * roi_batch_ind;

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                              * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                              * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                           * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                           * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), height_);
          hend = min(max(hend + roi_start_h, 0), height_);
          wstart = min(max(wstart + roi_start_w, 0), width_);
          wend = min(max(wend + roi_start_w, 0), width_);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw;
          if (is_empty) {
            top_data[pool_index] = 0;
            argmax_data[pool_index] = -1;
          }

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width_ + w;
              if (batch_data[index] > top_data[pool_index]) {
                top_data[pool_index] = batch_data[index];
                argmax_data[pool_index] = index;
              }
            }
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += data.size(2) * data.size(3);
      top_data += out.size(2) * out.size(3);
      argmax_data += max_idx.size(2) * max_idx.size(3);
    }
    // Increment ROI data pointer
    bottom_rois += bbox.size(1);
  }

  return;
}

template<typename Dtype>
inline void ROIPoolBackwardAcc(const Tensor<cpu, 4, Dtype> &in_grad,
                               const Tensor<cpu, 4, Dtype> &out_grad,
                               const Tensor<cpu, 2, Dtype> &bbox,
                               const Tensor<cpu, 4, Dtype> &max_idx,
                               const float spatial_scale_) {
  const Dtype *top_diff = out_grad.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *bottom_diff = in_grad.dptr_;
  Dtype *argmax_data = max_idx.dptr_;

  const int batch_size_ = in_grad.size(0);
  const int channels_ = in_grad.size(1);
  const int height_ = in_grad.size(2);
  const int width_ = in_grad.size(3);
  const int pooled_height_ = out_grad.size(2);
  const int pooled_width_ = out_grad.size(3);

  const int num_rois = bbox.size(0);

  for (int b = 0; b < batch_size_; ++b) {
    for (int c = 0; c < channels_; ++c) {
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          int offset_bottom_diff = (b * channels_ + c) * height_ * width_;
          offset_bottom_diff += h * width_ + w;

          Dtype gradient = 0;
          // Accumulate gradient over all ROIs that pooled this element
          for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
            const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
            int roi_batch_ind = offset_bottom_rois[0];
            assert(roi_batch_ind >= 0);
            assert(roi_batch_ind < batch_size_);
            if (b != roi_batch_ind) {
              continue;
            }

            int roi_start_w = round(offset_bottom_rois[1] * spatial_scale_);
            int roi_start_h = round(offset_bottom_rois[2] * spatial_scale_);
            int roi_end_w = round(offset_bottom_rois[3] * spatial_scale_);
            int roi_end_h = round(offset_bottom_rois[4] * spatial_scale_);

            bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
            if (!in_roi) {
              continue;
            }

            // force malformed ROIs to be 1 * 1
            int roi_height = max(roi_end_h - roi_start_h + 1, 1);
            int roi_width = max(roi_end_w - roi_start_w + 1, 1);
            const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                                     / static_cast<Dtype>(pooled_height_);
            const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                                     / static_cast<Dtype>(pooled_width_);

            // compute pooled regions correspond to original (h, w) point
            int phstart = static_cast<int>(floor(static_cast<Dtype>(h - roi_start_h)
                                                 / bin_size_h));
            int pwstart = static_cast<int>(floor(static_cast<Dtype>(w - roi_start_w)
                                                 / bin_size_w));
            int phend = static_cast<int>(ceil(static_cast<Dtype>(h - roi_start_h + 1)
                                              / bin_size_h));
            int pwend = static_cast<int>(ceil(static_cast<Dtype>(w - roi_start_w + 1)
                                              / bin_size_w));

            // clip to boundaries of pooled region
            phstart = min(max(phstart, 0), pooled_height_);
            phend = min(max(phend, 0), pooled_height_);
            pwstart = min(max(pwstart, 0), pooled_width_);
            pwend = min(max(pwend, 0), pooled_width_);

            // accumulate over gradients in pooled regions
            int offset = (roi_n * channels_ + c) * pooled_height_ * pooled_width_;
            const Dtype* offset_top_diff = top_diff + offset;
            const Dtype* offset_argmax_data = argmax_data + offset;
            for (int ph = phstart; ph < phend; ++ph) {
              for (int pw = pwstart; pw < pwend; ++pw) {
                const int pooled_index = ph * pooled_width_ + pw;
                if (static_cast<int>(offset_argmax_data[pooled_index]) == h * width_ + w) {
                  gradient += offset_top_diff[pooled_index];
                }
              }
            }
          }
          bottom_diff[offset_bottom_diff] += gradient;
        }
      }
    }
  }

  return;
}
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(ROIPoolingParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ROIPoolingOp<cpu, DType>(param);
  });
  return op;
}

Operator *ROIPoolingProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(ROIPoolingParam);

MXNET_REGISTER_OP_PROPERTY(ROIPooling, ROIPoolingProp)
.describe("Performs region-of-interest pooling on inputs. Resize bounding box coordinates by "
"spatial_scale and crop input feature maps accordingly. The cropped feature maps are pooled "
"by max pooling to a fixed size output indicated by pooled_size. batch_size will change to "
"the number of region bounding boxes after ROIPooling")
.add_argument("data", "Symbol", "Input data to the pooling operator, a 4D Feature maps")
.add_argument("rois", "Symbol", "Bounding box coordinates, a 2D array of "
"[[batch_index, x1, y1, x2, y2]]. (x1, y1) and (x2, y2) are top left and down right corners "
"of designated region of interest. batch_index indicates the index of corresponding image "
"in the input data")
.add_arguments(ROIPoolingParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet