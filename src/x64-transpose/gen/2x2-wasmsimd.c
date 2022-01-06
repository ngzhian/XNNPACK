// Auto-generated file. Do not edit!
//   Template: src/x32-transpose/wasm-simd.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <wasm_simd128.h>

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/transpose.h>

void xnn_x64_transpose_ukernel__2x2_wasmsimd(
    const uint64_t *input,
    uint64_t * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height)
{
  assert(output_stride >= block_height * sizeof(uint64_t));
  assert(input_stride >= block_width * sizeof(uint64_t));
  const size_t tile_height = 2;
  const size_t tile_width = 2;
  const size_t tile_hbytes = tile_height * sizeof(uint64_t);
  const size_t tile_wbytes = tile_width * sizeof(uint64_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint64_t);
  const size_t input_offset = tile_height * input_stride;
  const uint64_t* i0 = input;
  const uint64_t* i1 = (const uint64_t*) ((uintptr_t) i0 + input_stride);
  uint64_t* o0 = (uint64_t*) output;
  uint64_t* o1 = (uint64_t*) ((uintptr_t) o0 + output_stride);
  do {
    if XNN_UNPREDICTABLE(block_width < 2) {
      o1 = o0;
    }
    size_t bh = block_height;
    for (; bh >= 2; bh -= 2) {
      const v128_t v1_0 = wasm_v128_load(i0);
      i0 = (uint64_t*) ((uintptr_t) i0 + input_offset);
      const v128_t v1_1 = wasm_v128_load(i1);
      i1 = (uint64_t*) ((uintptr_t) i1 + input_offset);
      const v128_t v0_0 = wasm_v64x2_shuffle(v1_0, v1_1, 0, 2);
      const v128_t v0_1 = wasm_v64x2_shuffle(v1_0, v1_1, 1, 3);
      wasm_v128_store(o1, v0_1);
      o1 = (uint64_t*) ((uintptr_t) o1 + tile_hbytes);
      wasm_v128_store(o0, v0_0);
      o0 = (uint64_t*) ((uintptr_t) o0 + tile_hbytes);
    }
    if (bh != 0) {
      const v128_t v1_0 = wasm_v128_load(i0);
      v128_t v1_1 = wasm_v128_xor(v1_0, v1_0);
      v128_t v0_0 = wasm_v64x2_shuffle(v1_0, v1_1, 0, 2);
      v128_t v0_1 = wasm_v64x2_shuffle(v1_0, v1_1, 1, 3);
      if (bh & 1) {
        *((double*) o1) = wasm_f64x2_extract_lane(v0_1, 0);
        *((double*) o0) = wasm_f64x2_extract_lane(v0_0, 0);
      }
    }
    i0 = (const uint64_t*) ((uintptr_t) i0 + input_reset);
    i1 = (const uint64_t*) ((uintptr_t) i0 + input_stride);
    o0 = (uint64_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint64_t*) ((uintptr_t) o1 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}

