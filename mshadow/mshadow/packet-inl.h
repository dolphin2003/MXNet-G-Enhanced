/*!
 *  Copyright (c) 2014 by Contributors
 * \file packet-inl.h
 * \brief Generic packet vectorization code
 */
#ifndef MSHADOW_PACKET_INL_H_
#define MSHADOW_PACKET_INL_H_

#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif
#include "./base.h"
#include "./tensor.h"
#include "./expression.h"


namespace mshadow {
/*! \brief namespace of packet math*/
namespace packet {

enum PacketArch {
  kPlain,
  kSSE2,
};

#if MSHADOW_USE_SSE
#define MSHADOW_DEFAULT_PACKET  ::mshadow::packet::kSSE2
#else
#define MSHADOW_DEFAULT_PACKET  ::mshadow::packet::kPlain
#endif

// whether packet operator is enabled.
/*!
 * \brief Generic packet type
 * \tparam DType The data type of the packet.
 * \tparam Arch the Arch of the packet.
 */
template<typename DType, PacketArch Arch = MSHADOW_DEFAULT_PACKET>
struct Packet;

template<PacketArch Arch>
struct AlignBytes {
  static const index_t value = 4;
};

}  // namespace packet
}  // namespace mshadow

namespace mshadow {
namespace packet {
/*!
 * \brief analog to cudaMallocPitch, allocate a aligned space with num_line * lspace cells
 * \param out_pitch output parameter, the actuall space allocated for each line
 * \param lspace number of cells required for each line
 * \param num_line number of lines to be allocated
 */
inline void* AlignedMallocPitch(size_t *out_pitch,
                                size_t lspace,
                                size_t num_line) {
  const index_t bits = AlignBytes<MSHADOW_DEFAULT_PACKET>::value;
  const index_t mask = (1 << bits) - 1;

  size_t pitch = ((lspace + mask) >> bits) << bits;
  *out_pitch = pitch;
#ifdef _MSC_VER
  void *res = _aligned_malloc(pitch * num_line, 1 << bits);
#else
  void *res;
  int ret = posix_memalign(&res, 1 << bits, pitch * num_line);
  CHECK_EQ(ret, 0) << "AlignedMallocPitch failed";
#endif
  if (res == NULL) {
    LOG(FATAL) << "AlignedMallocPitch failed";
  }
  return res;
}

/*!
 * \brief free aligned space
 * \param ptr pointer to space to be freed
 */
inline void AlignedFree(void *ptr) {
#ifdef _MSC_VER
  _aligned_free(ptr);
