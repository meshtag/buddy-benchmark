//===- MLIRConv2DBenchmark.cpp --------------------------------------------===//
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
// This file implements the benchmark for buddy-opt tool in buddy-mlir project.
//
//===----------------------------------------------------------------------===//

#include "ImageProcessing/Kernels.h"
#include "Utils/Container.h"
// #include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

using namespace std;

// Declare the conv2d C interface.
extern "C" {
// void _mlir_ciface_mlir_conv_2d(MemRef<float, 2> *inputConv2D,
//                                MemRef<float, 2> *kernelConv2D,
//                                MemRef<float, 2> *outputConv2D);

// void _mlir_ciface_mlir_tosa_resize(MemRef<float, 4> *output, MemRef<float, 4> *input);

// void _mlir_ciface_mlir_tosa_resize(MemRef<float, 4> *input);

// void _mlir_ciface_matmul_static_tensor();
}

int main()
{
  cv::Mat inputImage = cv::imread(
    "../../benchmarks/ImageProcessing/Images/YuTu.png", cv::IMREAD_GRAYSCALE);

  intptr_t sizesInput[4] = {1, inputImage.rows, inputImage.cols, 1};
  intptr_t sizesOutput[4] = {1, 500, 500, 1};

  MemRef<float, 4> input(inputImage, sizesInput);
  MemRef<float, 4> output(sizesOutput);

  // _mlir_ciface_mlir_tosa_resize(&input, &output);
  // _mlir_ciface_mlir_tosa_resize(&input);
  // _mlir_ciface_matmul_static_tensor();

  float *check = output.getData();

  float *checkInput = input.getData();

  std::cout << output.getSize() << "\n";
  std::cout << input.getSize() << "\n\n";

  std::cout << output.getRank() << "\n";
  std::cout << input.getRank() << "\n";

  // for (int i = 0; i < 32; ++i)
  //   std::cout << checkInput[i] << " ";
  // std::cout << "\n";

  // for (int i = 0; i < 32; ++i)
  //   std::cout << check[i] << " ";
  // std::cout << "\n";

  cv::Mat outputImage(500, 500, CV_8UC1);
  // cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);
  // cv::Mat outputImage(500, 500, CV_8UC1, output.getData());
  // cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1, input.getData());

  // std::cout << outputImage.rows << " " << outputImage.cols << "\n";

  // for (int i = 0; i < inputImage.rows; ++i)
  // {
  //   for (int j = 0; j < inputImage.cols; ++j)
  //   {
  //     outputImage.at<uchar>(i, j) = checkInput[i * inputImage.rows + j];
  //   }
  // }

  for (int i = 0; i < 500; ++i)
  {
    for (int j = 0; j < 500; ++j)
    {
      outputImage.at<uchar>(i, j) = check[i * 500 + j];
    }
  }

  // Choose a PNG compression level
  std::vector<int> compressionParams;
  compressionParams.push_back(cv::IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  cv::imwrite("ResultMLIRTosaResize.png", outputImage, compressionParams);
}

// %0 = "tosa.resize"(%arg0) {mode = "NEAREST_NEIGHBOR", offset = [0, 1], offset_fp = [0.000000e+00 : f32, 0.000000e+00 : f32], output_size = [4, 4], shift = 8 : i32, stride = [1, 1], stride_fp = [0.000000e+00 : f32, 0.000000e+00 : f32]} : (tensor<?x?xi32>) -> tensor<?x?xi32>
