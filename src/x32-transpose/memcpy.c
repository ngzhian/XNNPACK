// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <string.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/transpose.h>

void xnn_x32_transpose_ukernel__memcpy(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height)
{
  assert(input_stride >= block_width * sizeof(uint32_t));

  const size_t tile_bytes = block_width * sizeof(uint32_t);
  const uint32_t* i = input;
  uint32_t* o = output;

  for (size_t bh = 0; bh < block_height; ++bh){
    memcpy(o, i, tile_bytes);
    i = (const uint32_t*) ((uintptr_t) i + input_stride);
    o = (uint32_t*) ((uintptr_t) o + tile_bytes);
  }
}
