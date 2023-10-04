/*!
 *  Copyright (c) 2014 by Contributors
 * \file sse-inl.h
 * \brief support of sse2 packet optimization of some operations
 * \author Tianqi Chen
 */
#ifndef MSHADOW_PACKET_SSE_INL_H_
#define MSHADOW_PACKET_SSE_INL_H_

#include <emmintrin.h>
#include "../base.h"
#include "../packet-inl.h"

namespace mshadow {
namespace packet {
template<>
struct Packet<f