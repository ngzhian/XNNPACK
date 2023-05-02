// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>
#include <xnnpack/subgraph-validation.h>


static enum xnn_status create_depth_to_space_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  struct xnn_weights_cache* weights_cache)
{
  assert(node->num_inputs == 1);
  const uint32_t input_id = node->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  assert(node->num_outputs == 1);
  const uint32_t output_id = node->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const size_t input_channel_dim = values[input_id].shape.dim[3];
  const size_t output_channel_dim = values[output_id].shape.dim[3];

  enum xnn_status status;
  if (values[input_id].layout == xnn_layout_type_nchw) {
    assert(values[output_id].layout == xnn_layout_type_nhwc);
    switch (node->compute_type) {
      case xnn_compute_type_fp16:
        status = xnn_create_depth_to_space_nchw2nhwc_x16(
            output_channel_dim /* output channels */,
            input_channel_dim /* input stride */,
            output_channel_dim /* output stride */,
            node->params.depth_to_space.block_size,
            node->flags,
            &opdata->operator_objects[0]);
        break;
      case xnn_compute_type_fp32:
        status = xnn_create_depth_to_space_nchw2nhwc_x32(
            output_channel_dim /* output channels */,
            input_channel_dim /* input stride */,
            output_channel_dim /* output stride */,
            node->params.depth_to_space.block_size,
            node->flags,
            &opdata->operator_objects[0]);
        break;
      default:
        XNN_UNREACHABLE;
    }
  } else {
    assert(values[input_id].layout == xnn_layout_type_nhwc);
    assert(values[output_id].layout == xnn_layout_type_nhwc);
    switch (node->compute_type) {
      case xnn_compute_type_fp16:
        status = xnn_create_depth_to_space_nhwc_x16(
            output_channel_dim /* output channels */,
            input_channel_dim /* input stride */,
            output_channel_dim /* output stride */,
            node->params.depth_to_space.block_size,
            node->flags,
            &opdata->operator_objects[0]);
        break;
      case xnn_compute_type_fp32:
        status = xnn_create_depth_to_space_nhwc_x32(
            output_channel_dim /* output channels */,
            input_channel_dim /* input stride */,
            output_channel_dim /* output stride */,
            node->params.depth_to_space.block_size,
            node->flags,
            &opdata->operator_objects[0]);
        break;
      case xnn_compute_type_qs8:
      case xnn_compute_type_qu8:
        status = xnn_create_depth_to_space_nhwc_x8(
            output_channel_dim /* output channels */,
            input_channel_dim /* input stride */,
            output_channel_dim /* output stride */,
            node->params.depth_to_space.block_size,
            node->flags,
            &opdata->operator_objects[0]);
        break;
      default:
        XNN_UNREACHABLE;
    }
  }
  if (status == xnn_status_success) {
    opdata->batch_size = values[input_id].shape.dim[0];
    opdata->input_height = values[input_id].shape.dim[1];
    opdata->input_width = values[input_id].shape.dim[2];
    opdata->output_height = values[output_id].shape.dim[1];
    opdata->output_width = values[output_id].shape.dim[2];
  }
  return status;
}

static enum xnn_status setup_depth_to_space_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const struct xnn_value* input_value = values + input_id;
  const void* input_data = input_value->data;
  assert(input_data != NULL);

  const struct xnn_value* output_value = values + output_id;
  void* output_data = output_value->data;
  assert(output_data != NULL);

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_depth_to_space_nchw2nhwc_x16:
      return xnn_setup_depth_to_space_nchw2nhwc_x16(
          opdata->operator_objects[0],
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          input_data,
          output_data,
          threadpool);
    case xnn_operator_type_depth_to_space_nchw2nhwc_x32:
      return xnn_setup_depth_to_space_nchw2nhwc_x32(
          opdata->operator_objects[0],
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          input_data,
          output_data,
          threadpool);
    case xnn_operator_type_depth_to_space_nhwc_x16:
      return xnn_setup_depth_to_space_nhwc_x16(
          opdata->operator_objects[0],
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          input_data,
          output_data,
          threadpool);
    case xnn_operator_type_depth_to_space_nhwc_x32:
      return xnn_setup_depth_to_space_nhwc_x32(
          opdata->operator_objects[0],
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          input_data,
          output_data,
          threadpool);
    case xnn_operator_type_depth_to_space_nhwc_x8:
      return xnn_setup_depth_to_space_nhwc_x8(
          opdata->operator_objects[0],
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          input_data,
          output_data,
          threadpool);
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_define_depth_to_space(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t block_size,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_depth_to_space)) != xnn_status_success) {
    return status;
  }

  if ((status = xnn_subgraph_check_input_node_id(xnn_node_type_depth_to_space, input_id, subgraph->num_values)) !=
      xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_depth_to_space, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp32:
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_depth_to_space), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_depth_to_space, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_depth_to_space, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  enum xnn_compute_type compute_type = xnn_compute_type_invalid;
  switch (output_value->datatype) {
    case xnn_datatype_fp32:
      compute_type = xnn_compute_type_fp32;
      break;
    case xnn_datatype_qint8:
      compute_type = xnn_compute_type_qs8;
      break;
    case xnn_datatype_quint8:
      compute_type = xnn_compute_type_qu8;
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_depth_to_space), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }
  assert(compute_type != xnn_compute_type_invalid);

  status = xnn_subgraph_check_datatype_matches(
    xnn_node_type_depth_to_space, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_quantization_parameter_matches(
      xnn_node_type_depth_to_space, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  if (block_size < 2) {
    xnn_log_error(
      "failed to define %s operator with block size #%" PRIu32 ": invalid block_size",
      xnn_node_type_to_string(xnn_node_type_depth_to_space), block_size);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_depth_to_space;
  node->compute_type = compute_type;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->params.depth_to_space.block_size = block_size;
  node->flags = flags;

  node->create = create_depth_to_space_operator;
  node->setup = setup_depth_to_space_operator;

  return xnn_status_success;
}
