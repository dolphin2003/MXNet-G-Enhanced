/*!
 *  Copyright (c) 2015 by Contributors
 * \file image_aug_default.cc
 * \brief Default augmenter.
 */
#include <mxnet/base.h>
#include <utility>
#include <string>
#include <algorithm>
#include <vector>
#include "./image_augmenter.h"
#include "../common/utils.h"

#if MXNET_USE_OPENCV
// Registers
namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::io::ImageAugmenterReg);
}  // namespace dmlc
#endif

namespace mxnet {
namespace io {

/*! \brief image augmentation parameters*/
struct DefaultImageAugmentParam : public dmlc::Parameter<DefaultImageAugmentParam> {
  /*! \brief whether we do random cropping */
  bool rand_crop;
  /*! \brief whether we do nonrandom croping */
  int crop_y_start;
  /*! \brief whether we do nonrandom croping */
  int crop_x_start;
  /*! \brief [-max_rotate_angle, max_rotate_angle] */
  int max_rotate_angle;
  /*! \brief max aspect ratio */
  float max_aspect_ratio;
  /*! \brief random shear the image [-max_shear_ratio, max_shear_ratio] */
  float max_shear_ratio;
  /*! \brief max crop size */
  int max_crop_size;
  /*! \brief min crop size */
  int min_crop_size;
  /*! \brief max scale ratio */
  float max_random_scale;
  /*! \brief min scale_ratio */
  float min_random_scale;
  /*! \brief min image size */
  float min_img_size;
  /*! \brief max image size */
  float max_img_size;
  /*! \brief max random in H channel */
  int random_h;
  /*! \brief max random in S channel */
  int random_s;
  /*! \brief max random in L channel */
  int random_l;
  /*! \brief rotate angle */
  int rotate;
  /*! \brief filled color while padding */
  int fill_value;
  /*! \brief interpolation method 0-NN 1-bilinear 2-cubic 3-area 4-lanczos4 9-auto 10-rand  */
  int inter_method;
  /*! \brief padding size */
  int pad;
  /*! \brief shape of the image data*/
  TShape data_shape;
  // declare parameters
  DMLC_DECLARE_PARAMETER(DefaultImageAugmentParam) {
    DMLC_DECLARE_FIELD(rand_crop).set_default(false)
        .describe("Augmentation Param: Whether to random crop on the image");
    DMLC_DECLARE_FIELD(crop_y_start).set_default(-1)
        .describe("Augmentation Param: Where to nonrandom crop on y.");
    DMLC_DECLARE_FIELD(crop_x_start).set_default(-1)
        .describe("Augmentation Param: Where to nonrandom crop on x.");
    DMLC_DECLARE_FIELD(max_rotate_angle).set_default(0.0f)
        .describe("Augmentation Param: rotated randomly in [-max_rotate_angle, max_rotate_angle].");
    DMLC_DECLARE_FIELD(max_aspect_ratio).set_default(0.0f)
        .describe("Augmentation Param: denotes the max ratio of random aspect ratio augmentation.");
    DMLC_DECLARE_FIELD(max_shear_ratio).set_default(0.0f)
        .describe("Augmentation Param: denotes the max random shearing ratio.");
    DMLC_DECLARE_FIELD(max_crop_size).set_default(-1)
        .describe("Augmentation Param: Maximum crop size.");
    DMLC_DECLARE_FIELD(min_crop_size).set_default(-1)
        .describe("Augmentation Param: Minimum crop size.");
    DMLC_DECLARE_FIELD(max_random_scale).set_default(1.0f)
        .describe("Augmentation Param: Maxmum scale ratio.");
    DMLC_DECLARE_FIELD(min_random_scale).set_default(1.0f)
        .describe("Augmentation Param: Minimum scale ratio.");
    DMLC_DECLARE_FIELD(max_img_size).set_default(1e10f)
        