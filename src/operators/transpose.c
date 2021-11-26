// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/operator.h>
#include <xnnpack/pack.h>
#include <xnnpack/params.h>


void swap(size_t *a, size_t *b)
{
  size_t tmp = *a;
  *a = *b;
  *b = tmp;
}

void reverse(size_t *arr, size_t num_dims)
{
  size_t *start = &arr[0];
  size_t *end = &arr[num_dims-1];
  while (start < end) {
    swap(start, end);
    ++start;
    --end;
  }
}

static enum xnn_status create_transpose_nd(
    size_t num_dims,
    const size_t* shape,
    const size_t* perm,
    size_t element_size,
    uint32_t datatype_init_flags,
    enum xnn_operator_type operator_type,
    xnn_operator_t* transpose_op_out)
{
  xnn_operator_t transpose_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_unsupported_hardware;

  if ((xnn_params.init_flags & datatype_init_flags) != datatype_init_flags) {
    xnn_log_error(
      "failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (num_dims == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu num_dims: num_dims must be non-zero",
      xnn_operator_type_to_string(operator_type), num_dims);
    goto error;
  }

  for (int i = 0; i < num_dims; ++i) {
    if (perm[i] >= num_dims) {
      xnn_log_error(
          "failed to create %s operator with %zu perm and %zu num_dims: 0 <= perm < num_dims",
          xnn_operator_type_to_string(operator_type), perm[i], num_dims);
      goto error;
    }
  }

  for (int i = 0; i < num_dims - 1; ++i) {
    for (int j = i + 1; j < num_dims; ++j) {
      if (perm[i] == perm[j]) {
        xnn_log_error(
            "failed to create %s operator with duplicate entries in perm",
            xnn_operator_type_to_string(operator_type));
        goto error;
      }
    }
  }

  status = xnn_status_out_of_memory;

  transpose_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (transpose_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  transpose_op->flags = datatype_init_flags;

  size_t collasped_dims = num_dims;
  size_t full_copy = 1;
  size_t row_copy = 0;
  for (size_t i = 0; i < num_dims; ++i) {
    full_copy &= (perm[i] == i);
  }
  if (full_copy) {
    collasped_dims = 1;
  } else if (perm[0] == 0) {
    row_copy = 1;
  }
  struct transpose_context *context = &transpose_op->context.transpose;

  size_t shape_out[XNN_MAX_TENSOR_DIMS];
  size_t stride_in[XNN_MAX_TENSOR_DIMS];
  size_t n = shape[0];
  stride_in[0] = 1;
  context->stride[0] = 1;
  for (size_t i = 1; i < num_dims; ++i) {
    stride_in[i] = stride_in[i - 1] * shape[i - 1];
    n *= shape[i];
  }
  context->n = n;
  for (size_t i = 0; i < num_dims; ++i) {
    context->loop[i] = i;
    context->offset[i] = stride_in[perm[i]];
    shape_out[i] = shape[perm[i]];
  }
  for (size_t i = 1; i < num_dims; ++i) {
    context->stride[i] = context->stride[i - 1] * shape_out[i - 1];
  }

  reverse(context->offset, num_dims);
  size_t zero_index = 0;
  for (size_t i = 0; i < num_dims; ++i) {
    if (context->offset[i] == 1) {
      zero_index = i;
    }
  }
  swap(&context->loop[num_dims - 2], &context->loop[zero_index]);
  reverse(shape_out, num_dims);
  reverse(context->stride, num_dims);

  transpose_op->type = operator_type;
  context->memcpy_ukernel = xnn_params.x32.transpose.memcpy;
  if (row_copy) {
    context->ukernel = xnn_params.x32.transpose.transpose_memcpy;
  } else {
    context->ukernel = xnn_params.x32.transpose.transpose;
  }
  for (size_t i = 0; i < num_dims; ++i) {
    context->shape[i] = shape[i];
    context->perm[i] = perm[i];
    transpose_op->compute.range[i] = shape_out[context->loop[i]];
  }
  context->element_size = element_size;
  transpose_op->compute.tile[0] = xnn_params.x32.transpose.blocksize;
  transpose_op->compute.tile[1] = xnn_params.x32.transpose.blocksize;
  switch (collasped_dims) {
    case (1):
      transpose_op->compute.type = xnn_parallelization_type_1d_tile_1d;
      transpose_op->compute.task_1d = (pthreadpool_task_1d_t) xnn_compute_transpose_memcpy;
      transpose_op->compute.range[0] = n;
      transpose_op->compute.tile[0] = n;
      break;
    case (2):
      transpose_op->compute.type = xnn_parallelization_type_2d_tile_2d;
      transpose_op->compute.task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_transpose;
      break;
    case (3):
      transpose_op->compute.type = xnn_parallelization_type_3d_tile_2d;
      transpose_op->compute.task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_transpose_3d;
      break;
    case (4):
      transpose_op->compute.type = xnn_parallelization_type_4d_tile_2d;
      transpose_op->compute.task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_transpose_4d;
      break;
    case (5):
      transpose_op->compute.type = xnn_parallelization_type_5d_tile_2d;
      transpose_op->compute.task_5d_tile_2d = (pthreadpool_task_5d_tile_2d_t) xnn_compute_transpose_5d;
      break;
    case (6):
      transpose_op->compute.type = xnn_parallelization_type_6d_tile_2d;
      transpose_op->compute.task_6d_tile_2d = (pthreadpool_task_6d_tile_2d_t) xnn_compute_transpose_6d;
      break;
   default:
    xnn_log_error(
      "%zu dimensions not supported %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  *transpose_op_out = transpose_op;
  return xnn_status_success;

error:
  xnn_delete_operator(transpose_op);
  return status;
}

static enum xnn_status setup_transpose(
  xnn_operator_t transpose_op,
  const void* input,
  void* output)
{
  transpose_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(transpose_op->type));
    return xnn_status_uninitialized;
  }

  transpose_op->context.transpose.a = input;
  transpose_op->context.transpose.b = output;
  transpose_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_create_transpose_nd_x32(
  size_t num_dims,
  const size_t* shape,
  const size_t* perm,
  uint32_t flags,
  xnn_operator_t* transpose_op_out)
{
  return create_transpose_nd(
    num_dims, shape, perm,
    sizeof(uint32_t) /* sizeof(element) */,
    flags,
    xnn_operator_type_transpose_x32,
    transpose_op_out);
}

enum xnn_status xnn_setup_transpose_nd_x32(
    xnn_operator_t transpose_op,
    const uint32_t* input,
    uint32_t* output,
    pthreadpool_t threadpool)
{
  if (transpose_op->type != xnn_operator_type_transpose_x32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_transpose_x32),
      xnn_operator_type_to_string(transpose_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_transpose(
    transpose_op,
    input, output);
}
