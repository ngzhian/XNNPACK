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

void xnn_x8_transpose_ukernel__16x16_wasmsimd(
    const uint8_t *input,
    uint8_t * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height)
{
  assert(output_stride >= block_height * sizeof(uint8_t));
  assert(input_stride >= block_width * sizeof(uint8_t));
  const size_t tile_height = 16;
  const size_t tile_width = 16;
  const size_t tile_hbytes = tile_height * sizeof(uint8_t);
  const size_t tile_wbytes = tile_width * sizeof(uint8_t);
  const size_t input_reset = tile_wbytes - (block_height - ((block_height % tile_height) != 0)) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(uint8_t);
  const uint8_t* i0 = input;
  uint8_t* o0 = (uint8_t*) output;
  do {
    size_t bh = block_height;
    for (; bh >= 16; bh -= 16) {
      const v128_t v4_0 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_1 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_2 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_3 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_4 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_5 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_6 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_7 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_8 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_9 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_10 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_11 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_12 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_13 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_14 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v4_15 = wasm_v128_load(i0);
      i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      const v128_t v3_0 = wasm_v8x16_shuffle(v4_0, v4_8, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_1 = wasm_v8x16_shuffle(v4_0, v4_8, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_2 = wasm_v8x16_shuffle(v4_1, v4_9, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_3 = wasm_v8x16_shuffle(v4_1, v4_9, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_4 = wasm_v8x16_shuffle(v4_2, v4_10, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_5 = wasm_v8x16_shuffle(v4_2, v4_10, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_6 = wasm_v8x16_shuffle(v4_3, v4_11, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_7 = wasm_v8x16_shuffle(v4_3, v4_11, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_8 = wasm_v8x16_shuffle(v4_4, v4_12, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_9 = wasm_v8x16_shuffle(v4_4, v4_12, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_10 = wasm_v8x16_shuffle(v4_5, v4_13, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_11 = wasm_v8x16_shuffle(v4_5, v4_13, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_12 = wasm_v8x16_shuffle(v4_6, v4_14, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_13 = wasm_v8x16_shuffle(v4_6, v4_14, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v3_14 = wasm_v8x16_shuffle(v4_7, v4_15, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v3_15 = wasm_v8x16_shuffle(v4_7, v4_15, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_0 = wasm_v8x16_shuffle(v3_0, v3_8, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_1 = wasm_v8x16_shuffle(v3_0, v3_8, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_2 = wasm_v8x16_shuffle(v3_1, v3_9, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_3 = wasm_v8x16_shuffle(v3_1, v3_9, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_4 = wasm_v8x16_shuffle(v3_2, v3_10, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_5 = wasm_v8x16_shuffle(v3_2, v3_10, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_6 = wasm_v8x16_shuffle(v3_3, v3_11, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_7 = wasm_v8x16_shuffle(v3_3, v3_11, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_8 = wasm_v8x16_shuffle(v3_4, v3_12, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_9 = wasm_v8x16_shuffle(v3_4, v3_12, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_10 = wasm_v8x16_shuffle(v3_5, v3_13, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_11 = wasm_v8x16_shuffle(v3_5, v3_13, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_12 = wasm_v8x16_shuffle(v3_6, v3_14, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_13 = wasm_v8x16_shuffle(v3_6, v3_14, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v2_14 = wasm_v8x16_shuffle(v3_7, v3_15, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v2_15 = wasm_v8x16_shuffle(v3_7, v3_15, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_0 = wasm_v8x16_shuffle(v2_0, v2_8, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_1 = wasm_v8x16_shuffle(v2_0, v2_8, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_2 = wasm_v8x16_shuffle(v2_1, v2_9, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_3 = wasm_v8x16_shuffle(v2_1, v2_9, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_4 = wasm_v8x16_shuffle(v2_2, v2_10, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_5 = wasm_v8x16_shuffle(v2_2, v2_10, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_6 = wasm_v8x16_shuffle(v2_3, v2_11, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_7 = wasm_v8x16_shuffle(v2_3, v2_11, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_8 = wasm_v8x16_shuffle(v2_4, v2_12, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_9 = wasm_v8x16_shuffle(v2_4, v2_12, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_10 = wasm_v8x16_shuffle(v2_5, v2_13, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_11 = wasm_v8x16_shuffle(v2_5, v2_13, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_12 = wasm_v8x16_shuffle(v2_6, v2_14, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_13 = wasm_v8x16_shuffle(v2_6, v2_14, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v1_14 = wasm_v8x16_shuffle(v2_7, v2_15, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v1_15 = wasm_v8x16_shuffle(v2_7, v2_15, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v0_0 = wasm_v8x16_shuffle(v1_0, v1_8, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v0_1 = wasm_v8x16_shuffle(v1_0, v1_8, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v0_2 = wasm_v8x16_shuffle(v1_1, v1_9, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v0_3 = wasm_v8x16_shuffle(v1_1, v1_9, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v0_4 = wasm_v8x16_shuffle(v1_2, v1_10, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v0_5 = wasm_v8x16_shuffle(v1_2, v1_10, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v0_6 = wasm_v8x16_shuffle(v1_3, v1_11, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v0_7 = wasm_v8x16_shuffle(v1_3, v1_11, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v0_8 = wasm_v8x16_shuffle(v1_4, v1_12, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v0_9 = wasm_v8x16_shuffle(v1_4, v1_12, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v0_10 = wasm_v8x16_shuffle(v1_5, v1_13, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v0_11 = wasm_v8x16_shuffle(v1_5, v1_13, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v0_12 = wasm_v8x16_shuffle(v1_6, v1_14, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v0_13 = wasm_v8x16_shuffle(v1_6, v1_14, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      const v128_t v0_14 = wasm_v8x16_shuffle(v1_7, v1_15, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      const v128_t v0_15 = wasm_v8x16_shuffle(v1_7, v1_15, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      size_t rem = min(block_width - 1, 15);
      uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + rem * output_stride);
      switch (rem) {
      case 15:
        wasm_v128_store(o1, v0_15);
        o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
      case 14:
        wasm_v128_store(o1, v0_14);
        o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
      case 13:
        wasm_v128_store(o1, v0_13);
        o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
      case 12:
        wasm_v128_store(o1, v0_12);
        o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
      case 11:
        wasm_v128_store(o1, v0_11);
        o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
      case 10:
        wasm_v128_store(o1, v0_10);
        o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
      case 9:
        wasm_v128_store(o1, v0_9);
        o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
      case 8:
        wasm_v128_store(o1, v0_8);
        o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
      case 7:
        wasm_v128_store(o1, v0_7);
        o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
      case 6:
        wasm_v128_store(o1, v0_6);
        o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
      case 5:
        wasm_v128_store(o1, v0_5);
        o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
      case 4:
        wasm_v128_store(o1, v0_4);
        o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
      case 3:
        wasm_v128_store(o1, v0_3);
        o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
      case 2:
        wasm_v128_store(o1, v0_2);
        o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
      case 1:
        wasm_v128_store(o1, v0_1);
      }
      wasm_v128_store(o0, v0_0);
      o0 = (uint8_t*) ((uintptr_t) o0 + tile_hbytes);
    }
    if (bh != 0) {
      const v128_t v4_0 = wasm_v128_load(i0);
      if XNN_UNPREDICTABLE(bh > 1) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const v128_t v4_1 = wasm_v128_load(i0);
      if XNN_UNPREDICTABLE(bh > 2) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const v128_t v4_2 = wasm_v128_load(i0);
      if XNN_UNPREDICTABLE(bh > 3) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const v128_t v4_3 = wasm_v128_load(i0);
      if XNN_UNPREDICTABLE(bh > 4) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const v128_t v4_4 = wasm_v128_load(i0);
      if XNN_UNPREDICTABLE(bh > 5) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const v128_t v4_5 = wasm_v128_load(i0);
      if XNN_UNPREDICTABLE(bh > 6) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const v128_t v4_6 = wasm_v128_load(i0);
      if XNN_UNPREDICTABLE(bh > 7) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const v128_t v4_7 = wasm_v128_load(i0);
      if XNN_UNPREDICTABLE(bh > 8) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const v128_t v4_8 = wasm_v128_load(i0);
      if XNN_UNPREDICTABLE(bh > 9) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const v128_t v4_9 = wasm_v128_load(i0);
      if XNN_UNPREDICTABLE(bh > 10) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const v128_t v4_10 = wasm_v128_load(i0);
      if XNN_UNPREDICTABLE(bh > 11) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const v128_t v4_11 = wasm_v128_load(i0);
      if XNN_UNPREDICTABLE(bh > 12) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const v128_t v4_12 = wasm_v128_load(i0);
      if XNN_UNPREDICTABLE(bh > 13) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const v128_t v4_13 = wasm_v128_load(i0);
      if XNN_UNPREDICTABLE(bh > 14) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      const v128_t v4_14 = wasm_v128_load(i0);
      if XNN_UNPREDICTABLE(bh > 15) {
        i0 = (uint8_t*) ((uintptr_t) i0 + input_stride);
      }
      v128_t v4_15 = wasm_v128_xor(v4_0, v4_0);
      v128_t v3_0 = wasm_v8x16_shuffle(v4_0, v4_8, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v3_1 = wasm_v8x16_shuffle(v4_0, v4_8, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v3_2 = wasm_v8x16_shuffle(v4_1, v4_9, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v3_3 = wasm_v8x16_shuffle(v4_1, v4_9, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v3_4 = wasm_v8x16_shuffle(v4_2, v4_10, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v3_5 = wasm_v8x16_shuffle(v4_2, v4_10, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v3_6 = wasm_v8x16_shuffle(v4_3, v4_11, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v3_7 = wasm_v8x16_shuffle(v4_3, v4_11, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v3_8 = wasm_v8x16_shuffle(v4_4, v4_12, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v3_9 = wasm_v8x16_shuffle(v4_4, v4_12, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v3_10 = wasm_v8x16_shuffle(v4_5, v4_13, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v3_11 = wasm_v8x16_shuffle(v4_5, v4_13, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v3_12 = wasm_v8x16_shuffle(v4_6, v4_14, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v3_13 = wasm_v8x16_shuffle(v4_6, v4_14, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v3_14 = wasm_v8x16_shuffle(v4_7, v4_15, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v3_15 = wasm_v8x16_shuffle(v4_7, v4_15, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v2_0 = wasm_v8x16_shuffle(v3_0, v3_8, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v2_1 = wasm_v8x16_shuffle(v3_0, v3_8, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v2_2 = wasm_v8x16_shuffle(v3_1, v3_9, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v2_3 = wasm_v8x16_shuffle(v3_1, v3_9, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v2_4 = wasm_v8x16_shuffle(v3_2, v3_10, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v2_5 = wasm_v8x16_shuffle(v3_2, v3_10, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v2_6 = wasm_v8x16_shuffle(v3_3, v3_11, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v2_7 = wasm_v8x16_shuffle(v3_3, v3_11, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v2_8 = wasm_v8x16_shuffle(v3_4, v3_12, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v2_9 = wasm_v8x16_shuffle(v3_4, v3_12, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v2_10 = wasm_v8x16_shuffle(v3_5, v3_13, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v2_11 = wasm_v8x16_shuffle(v3_5, v3_13, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v2_12 = wasm_v8x16_shuffle(v3_6, v3_14, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v2_13 = wasm_v8x16_shuffle(v3_6, v3_14, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v2_14 = wasm_v8x16_shuffle(v3_7, v3_15, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v2_15 = wasm_v8x16_shuffle(v3_7, v3_15, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v1_0 = wasm_v8x16_shuffle(v2_0, v2_8, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v1_1 = wasm_v8x16_shuffle(v2_0, v2_8, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v1_2 = wasm_v8x16_shuffle(v2_1, v2_9, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v1_3 = wasm_v8x16_shuffle(v2_1, v2_9, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v1_4 = wasm_v8x16_shuffle(v2_2, v2_10, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v1_5 = wasm_v8x16_shuffle(v2_2, v2_10, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v1_6 = wasm_v8x16_shuffle(v2_3, v2_11, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v1_7 = wasm_v8x16_shuffle(v2_3, v2_11, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v1_8 = wasm_v8x16_shuffle(v2_4, v2_12, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v1_9 = wasm_v8x16_shuffle(v2_4, v2_12, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v1_10 = wasm_v8x16_shuffle(v2_5, v2_13, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v1_11 = wasm_v8x16_shuffle(v2_5, v2_13, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v1_12 = wasm_v8x16_shuffle(v2_6, v2_14, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v1_13 = wasm_v8x16_shuffle(v2_6, v2_14, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v1_14 = wasm_v8x16_shuffle(v2_7, v2_15, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v1_15 = wasm_v8x16_shuffle(v2_7, v2_15, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v0_0 = wasm_v8x16_shuffle(v1_0, v1_8, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v0_1 = wasm_v8x16_shuffle(v1_0, v1_8, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v0_2 = wasm_v8x16_shuffle(v1_1, v1_9, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v0_3 = wasm_v8x16_shuffle(v1_1, v1_9, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v0_4 = wasm_v8x16_shuffle(v1_2, v1_10, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v0_5 = wasm_v8x16_shuffle(v1_2, v1_10, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v0_6 = wasm_v8x16_shuffle(v1_3, v1_11, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v0_7 = wasm_v8x16_shuffle(v1_3, v1_11, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v0_8 = wasm_v8x16_shuffle(v1_4, v1_12, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v0_9 = wasm_v8x16_shuffle(v1_4, v1_12, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v0_10 = wasm_v8x16_shuffle(v1_5, v1_13, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v0_11 = wasm_v8x16_shuffle(v1_5, v1_13, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v0_12 = wasm_v8x16_shuffle(v1_6, v1_14, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v0_13 = wasm_v8x16_shuffle(v1_6, v1_14, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      v128_t v0_14 = wasm_v8x16_shuffle(v1_7, v1_15, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
      v128_t v0_15 = wasm_v8x16_shuffle(v1_7, v1_15, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
      size_t rem = min(block_width - 1, 15);
      if (bh & 8) {
        uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + rem * output_stride);
        switch (rem) {
        case 15:
         *((double*) o1) = wasm_f64x2_extract_lane(v0_15, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 14:
         *((double*) o1) = wasm_f64x2_extract_lane(v0_14, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 13:
         *((double*) o1) = wasm_f64x2_extract_lane(v0_13, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 12:
         *((double*) o1) = wasm_f64x2_extract_lane(v0_12, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 11:
         *((double*) o1) = wasm_f64x2_extract_lane(v0_11, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 10:
         *((double*) o1) = wasm_f64x2_extract_lane(v0_10, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 9:
         *((double*) o1) = wasm_f64x2_extract_lane(v0_9, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 8:
         *((double*) o1) = wasm_f64x2_extract_lane(v0_8, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 7:
         *((double*) o1) = wasm_f64x2_extract_lane(v0_7, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 6:
         *((double*) o1) = wasm_f64x2_extract_lane(v0_6, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 5:
         *((double*) o1) = wasm_f64x2_extract_lane(v0_5, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 4:
         *((double*) o1) = wasm_f64x2_extract_lane(v0_4, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 3:
         *((double*) o1) = wasm_f64x2_extract_lane(v0_3, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 2:
         *((double*) o1) = wasm_f64x2_extract_lane(v0_2, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 1:
          *((double*) o1) = wasm_f64x2_extract_lane(v0_1, 0);
        }
        *((double*) o0) = wasm_f64x2_extract_lane(v0_0, 0);
        o0 += 8;
        v0_0 = wasm_v64x2_shuffle(v0_0, v0_0, 1, 1);
        v0_1 = wasm_v64x2_shuffle(v0_1, v0_1, 1, 1);
        v0_2 = wasm_v64x2_shuffle(v0_2, v0_2, 1, 1);
        v0_3 = wasm_v64x2_shuffle(v0_3, v0_3, 1, 1);
        v0_4 = wasm_v64x2_shuffle(v0_4, v0_4, 1, 1);
        v0_5 = wasm_v64x2_shuffle(v0_5, v0_5, 1, 1);
        v0_6 = wasm_v64x2_shuffle(v0_6, v0_6, 1, 1);
        v0_7 = wasm_v64x2_shuffle(v0_7, v0_7, 1, 1);
        v0_8 = wasm_v64x2_shuffle(v0_8, v0_8, 1, 1);
        v0_9 = wasm_v64x2_shuffle(v0_9, v0_9, 1, 1);
        v0_10 = wasm_v64x2_shuffle(v0_10, v0_10, 1, 1);
        v0_11 = wasm_v64x2_shuffle(v0_11, v0_11, 1, 1);
        v0_12 = wasm_v64x2_shuffle(v0_12, v0_12, 1, 1);
        v0_13 = wasm_v64x2_shuffle(v0_13, v0_13, 1, 1);
        v0_14 = wasm_v64x2_shuffle(v0_14, v0_14, 1, 1);
        v0_15 = wasm_v64x2_shuffle(v0_15, v0_15, 1, 1);
      }
      if (bh & 4) {
        uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + rem * output_stride);
        switch (rem) {
        case 15:
          *((float*) o1) = wasm_f32x4_extract_lane(v0_15, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 14:
          *((float*) o1) = wasm_f32x4_extract_lane(v0_14, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 13:
          *((float*) o1) = wasm_f32x4_extract_lane(v0_13, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 12:
          *((float*) o1) = wasm_f32x4_extract_lane(v0_12, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 11:
          *((float*) o1) = wasm_f32x4_extract_lane(v0_11, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 10:
          *((float*) o1) = wasm_f32x4_extract_lane(v0_10, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 9:
          *((float*) o1) = wasm_f32x4_extract_lane(v0_9, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 8:
          *((float*) o1) = wasm_f32x4_extract_lane(v0_8, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 7:
          *((float*) o1) = wasm_f32x4_extract_lane(v0_7, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 6:
          *((float*) o1) = wasm_f32x4_extract_lane(v0_6, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 5:
          *((float*) o1) = wasm_f32x4_extract_lane(v0_5, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 4:
          *((float*) o1) = wasm_f32x4_extract_lane(v0_4, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 3:
          *((float*) o1) = wasm_f32x4_extract_lane(v0_3, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 2:
          *((float*) o1) = wasm_f32x4_extract_lane(v0_2, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 1:
          *((float*) o1) = wasm_f32x4_extract_lane(v0_1, 0);
        }
        *((float*) o0) = wasm_f32x4_extract_lane(v0_0, 0);
        o0 += 4;
        v0_0 = wasm_u64x2_shr(v0_0, 32);
        v0_1 = wasm_u64x2_shr(v0_1, 32);
        v0_2 = wasm_u64x2_shr(v0_2, 32);
        v0_3 = wasm_u64x2_shr(v0_3, 32);
        v0_4 = wasm_u64x2_shr(v0_4, 32);
        v0_5 = wasm_u64x2_shr(v0_5, 32);
        v0_6 = wasm_u64x2_shr(v0_6, 32);
        v0_7 = wasm_u64x2_shr(v0_7, 32);
        v0_8 = wasm_u64x2_shr(v0_8, 32);
        v0_9 = wasm_u64x2_shr(v0_9, 32);
        v0_10 = wasm_u64x2_shr(v0_10, 32);
        v0_11 = wasm_u64x2_shr(v0_11, 32);
        v0_12 = wasm_u64x2_shr(v0_12, 32);
        v0_13 = wasm_u64x2_shr(v0_13, 32);
        v0_14 = wasm_u64x2_shr(v0_14, 32);
        v0_15 = wasm_u64x2_shr(v0_15, 32);
      }
      if (bh & 2) {
        uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + rem * output_stride);
        switch (rem) {
        case 15:
          *((uint16_t*) o1) = wasm_i16x8_extract_lane(v0_15, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 14:
          *((uint16_t*) o1) = wasm_i16x8_extract_lane(v0_14, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 13:
          *((uint16_t*) o1) = wasm_i16x8_extract_lane(v0_13, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 12:
          *((uint16_t*) o1) = wasm_i16x8_extract_lane(v0_12, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 11:
          *((uint16_t*) o1) = wasm_i16x8_extract_lane(v0_11, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 10:
          *((uint16_t*) o1) = wasm_i16x8_extract_lane(v0_10, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 9:
          *((uint16_t*) o1) = wasm_i16x8_extract_lane(v0_9, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 8:
          *((uint16_t*) o1) = wasm_i16x8_extract_lane(v0_8, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 7:
          *((uint16_t*) o1) = wasm_i16x8_extract_lane(v0_7, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 6:
          *((uint16_t*) o1) = wasm_i16x8_extract_lane(v0_6, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 5:
          *((uint16_t*) o1) = wasm_i16x8_extract_lane(v0_5, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 4:
          *((uint16_t*) o1) = wasm_i16x8_extract_lane(v0_4, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 3:
          *((uint16_t*) o1) = wasm_i16x8_extract_lane(v0_3, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 2:
          *((uint16_t*) o1) = wasm_i16x8_extract_lane(v0_2, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 1:
          *((uint16_t*) o1) = wasm_i16x8_extract_lane(v0_1, 0);
        }
        *((uint16_t*) o0) = wasm_i16x8_extract_lane(v0_0, 0);
        o0 += 2;
        v0_0 = wasm_u32x4_shr(v0_0, 16);
        v0_1 = wasm_u32x4_shr(v0_1, 16);
        v0_2 = wasm_u32x4_shr(v0_2, 16);
        v0_3 = wasm_u32x4_shr(v0_3, 16);
        v0_4 = wasm_u32x4_shr(v0_4, 16);
        v0_5 = wasm_u32x4_shr(v0_5, 16);
        v0_6 = wasm_u32x4_shr(v0_6, 16);
        v0_7 = wasm_u32x4_shr(v0_7, 16);
        v0_8 = wasm_u32x4_shr(v0_8, 16);
        v0_9 = wasm_u32x4_shr(v0_9, 16);
        v0_10 = wasm_u32x4_shr(v0_10, 16);
        v0_11 = wasm_u32x4_shr(v0_11, 16);
        v0_12 = wasm_u32x4_shr(v0_12, 16);
        v0_13 = wasm_u32x4_shr(v0_13, 16);
        v0_14 = wasm_u32x4_shr(v0_14, 16);
        v0_15 = wasm_u32x4_shr(v0_15, 16);
      }
      if (bh & 1) {
        uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + rem * output_stride);
        switch (rem) {
        case 15:
          *o1 = wasm_i8x16_extract_lane(v0_15, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 14:
          *o1 = wasm_i8x16_extract_lane(v0_14, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 13:
          *o1 = wasm_i8x16_extract_lane(v0_13, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 12:
          *o1 = wasm_i8x16_extract_lane(v0_12, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 11:
          *o1 = wasm_i8x16_extract_lane(v0_11, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 10:
          *o1 = wasm_i8x16_extract_lane(v0_10, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 9:
          *o1 = wasm_i8x16_extract_lane(v0_9, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 8:
          *o1 = wasm_i8x16_extract_lane(v0_8, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 7:
          *o1 = wasm_i8x16_extract_lane(v0_7, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 6:
          *o1 = wasm_i8x16_extract_lane(v0_6, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 5:
          *o1 = wasm_i8x16_extract_lane(v0_5, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 4:
          *o1 = wasm_i8x16_extract_lane(v0_4, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 3:
          *o1 = wasm_i8x16_extract_lane(v0_3, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 2:
          *o1 = wasm_i8x16_extract_lane(v0_2, 0);
          o1 = (uint8_t*) ((uintptr_t) o1 - output_stride);
        case 1:
          *o1 = wasm_i8x16_extract_lane(v0_1, 0);
        }
        *o0 = wasm_i8x16_extract_lane(v0_0, 0);
      }
    }
    i0 = (const uint8_t*) ((uintptr_t) i0 + input_reset);
    o0 = (uint8_t*) ((uintptr_t) o0 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}

