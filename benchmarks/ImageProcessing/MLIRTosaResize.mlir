//===- MLIRConv2D.mlir ----------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file provides the MLIR Resize function.
//
//===----------------------------------------------------------------------===//

func @mlir_tosa_resize(%arg0: tensor<1x?x?x1xi32>, %arg1: tensor<1x?x?x1xf32>) {
  %0 = "tosa.resize"(%arg0) {mode = "NEAREST_NEIGHBOR", offset = [0, 0], offset_fp = [0.000000e+00 : f32, 0.000000e+00 : f32], output_size = [500, 500], shift = 8 : i32, stride = [1, 1], stride_fp = [0.000000e+00 : f32, 0.000000e+00 : f32]} : (tensor<1x?x?x1xi32>) -> tensor<1x?x?x1xi32>
  return
}
