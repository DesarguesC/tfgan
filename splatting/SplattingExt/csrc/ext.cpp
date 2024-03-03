/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include <torch/extension.h>
// clang-format on
#include "splatting/rendering_function.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rendering", &RenderingCUDA);
  m.def("rendering_backward", &RenderingBackwardCUDA);
}