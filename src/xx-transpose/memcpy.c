// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <string.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/transpose.h>

void xnn_xx_transpose_ukernel__1xN_memcpy_scalar(
    const void* input,
    void* output,
    size_t input_stride,
    size_t element_size,
    size_t block_width,
    size_t block_height)
{
  assert(input_stride >= block_width * element_size);

  const size_t tile_bytes = block_width * element_size;
  const void* i = input;
  void* o = output;

  for (size_t bh = 0; bh < block_height; ++bh){
    memcpy(o, i, tile_bytes);
    i = (const void*) ((uintptr_t) i + input_stride);
    o = (void*) ((uintptr_t) o + tile_bytes);
  }
}
