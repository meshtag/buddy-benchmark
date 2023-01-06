//===- BuddyResize2D.mlir -------------------------------------------------===//
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
// This file provides the Buddy Resize2D function.
//
//===----------------------------------------------------------------------===//

func.func @resize_2d_nearest_neighbour_interpolation(%inputImage : memref<?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %outputImage : memref<?x?xf32>)
{
  dip.resize_2d NEAREST_NEIGHBOUR_INTERPOLATION %inputImage, %horizontal_scaling_factor, %vertical_scaling_factor, %outputImage : memref<?x?xf32>, f32, f32, memref<?x?xf32>
  return
}

func.func @resize_2d_bilinear_interpolation(%inputImage : memref<?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %outputImage : memref<?x?xf32>)
{
  dip.resize_2d BILINEAR_INTERPOLATION %inputImage, %horizontal_scaling_factor, %vertical_scaling_factor, %outputImage : memref<?x?xf32>, f32, f32, memref<?x?xf32>
  return
}
