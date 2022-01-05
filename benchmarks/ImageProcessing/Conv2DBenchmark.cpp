//===- Conv2DBenchmark.cpp ------------------------------------------------===//
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
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare the conv2d C interface.
extern "C" {
void _mlir_ciface_conv_2d(MemRef<float, 2> *inputConv2D,
                          MemRef<float, 2> *kernelConv2D,
                          MemRef<float, 2> *outputConv2D);
}

// Read input image.
Mat inputImageConv2D = imread(
    "../../benchmarks/ImageProcessing/Images/YuTu.png", IMREAD_GRAYSCALE);

// Define the kernel size.
int kernelRows3x3Conv2D = sobel3x3KernelRows;
int kernelCols3x3Conv2D = sobel3x3KernelCols;

int kernelRows5x5Conv2D = sobel5x5KernelRows;
int kernelCols5x5Conv2D = sobel5x5KernelCols;

int kernelRows7x7Conv2D = sobel7x7KernelRows;
int kernelCols7x7Conv2D = sobel7x7KernelCols;

int kernelRows9x9Conv2D = sobel9x9KernelRows;
int kernelCols9x9Conv2D = sobel9x9KernelCols;

// Define the output size.
int outputRows3x3Conv2D = inputImageConv2D.rows - kernelRows3x3Conv2D + 1;
int outputCols3x3Conv2D = inputImageConv2D.cols - kernelCols3x3Conv2D + 1;

int outputRows5x5Conv2D = inputImageConv2D.rows - kernelRows5x5Conv2D + 1;
int outputCols5x5Conv2D = inputImageConv2D.cols - kernelCols5x5Conv2D + 1;

int outputRows7x7Conv2D = inputImageConv2D.rows - kernelRows7x7Conv2D + 1;
int outputCols7x7Conv2D = inputImageConv2D.cols - kernelCols7x7Conv2D + 1;

int outputRows9x9Conv2D = inputImageConv2D.rows - kernelRows9x9Conv2D + 1;
int outputCols9x9Conv2D = inputImageConv2D.cols - kernelCols9x9Conv2D + 1;

// Define sizes of input, kernel, and output.
intptr_t sizesInputConv2D[2] = {inputImageConv2D.rows, inputImageConv2D.cols};
intptr_t sizesKernel3x3Conv2D[2] = {kernelRows3x3Conv2D, kernelCols3x3Conv2D};
intptr_t sizesKernel5x5Conv2D[2] = {kernelRows5x5Conv2D, kernelCols5x5Conv2D};
intptr_t sizesKernel7x7Conv2D[2] = {kernelRows7x7Conv2D, kernelCols7x7Conv2D};
intptr_t sizesKernel9x9Conv2D[2] = {kernelRows9x9Conv2D, kernelCols9x9Conv2D};
intptr_t sizesOutput3x3Conv2D[2] = {outputRows3x3Conv2D, outputCols3x3Conv2D};
intptr_t sizesOutput5x5Conv2D[2] = {outputRows5x5Conv2D, outputCols5x5Conv2D};
intptr_t sizesOutput7x7Conv2D[2] = {outputRows7x7Conv2D, outputCols7x7Conv2D};
intptr_t sizesOutput9x9Conv2D[2] = {outputRows9x9Conv2D, outputCols9x9Conv2D};

// Define the MemRef descriptor for input, kernel, and output.
MemRef<float, 2> inputConv2D(inputImageConv2D, sizesInputConv2D);
MemRef<float, 2> kernel3x3Conv2D(sobel3x3KernelAlign, sizesKernel3x3Conv2D);
MemRef<float, 2> kernel5x5Conv2D(sobel5x5KernelAlign, sizesKernel5x5Conv2D);
MemRef<float, 2> kernel7x7Conv2D(sobel7x7KernelAlign, sizesKernel7x7Conv2D);
MemRef<float, 2> kernel9x9Conv2D(sobel9x9KernelAlign, sizesKernel9x9Conv2D);
MemRef<float, 2> output3x3Conv2D(sizesOutput3x3Conv2D);
MemRef<float, 2> output5x5Conv2D(sizesOutput5x5Conv2D);
MemRef<float, 2> output7x7Conv2D(sizesOutput7x7Conv2D);
MemRef<float, 2> output9x9Conv2D(sizesOutput9x9Conv2D);

static void BM_3x3_Conv2D_Buddy(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_conv_2d(&inputConv2D, &kernel3x3Conv2D, &output3x3Conv2D);
    }
  }
}

// Register benchmarking function with different arguments.
BENCHMARK(BM_3x3_Conv2D_Buddy)->Arg(1);
BENCHMARK(BM_3x3_Conv2D_Buddy)->Arg(2);
BENCHMARK(BM_3x3_Conv2D_Buddy)->Arg(4);
BENCHMARK(BM_3x3_Conv2D_Buddy)->Arg(8);
BENCHMARK(BM_3x3_Conv2D_Buddy)->Arg(16);

static void BM_5x5_Conv2D_Buddy(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_conv_2d(&inputConv2D, &kernel5x5Conv2D, &output5x5Conv2D);
    }
  }
}

// Register benchmarking function with different arguments.
BENCHMARK(BM_5x5_Conv2D_Buddy)->Arg(1);
BENCHMARK(BM_5x5_Conv2D_Buddy)->Arg(2);
BENCHMARK(BM_5x5_Conv2D_Buddy)->Arg(4);
BENCHMARK(BM_5x5_Conv2D_Buddy)->Arg(8);
BENCHMARK(BM_5x5_Conv2D_Buddy)->Arg(16);

static void BM_7x7_Conv2D_Buddy(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_conv_2d(&inputConv2D, &kernel7x7Conv2D, &output7x7Conv2D);
    }
  }
}

// Register benchmarking function with different arguments.
BENCHMARK(BM_7x7_Conv2D_Buddy)->Arg(1);
BENCHMARK(BM_7x7_Conv2D_Buddy)->Arg(2);
BENCHMARK(BM_7x7_Conv2D_Buddy)->Arg(4);
BENCHMARK(BM_7x7_Conv2D_Buddy)->Arg(8);
BENCHMARK(BM_7x7_Conv2D_Buddy)->Arg(16);

static void BM_9x9_Conv2D_Buddy(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_conv_2d(&inputConv2D, &kernel9x9Conv2D, &output9x9Conv2D);
    }
  }
}

// Register benchmarking function with different arguments.
BENCHMARK(BM_9x9_Conv2D_Buddy)->Arg(1);
BENCHMARK(BM_9x9_Conv2D_Buddy)->Arg(2);
BENCHMARK(BM_9x9_Conv2D_Buddy)->Arg(4);
BENCHMARK(BM_9x9_Conv2D_Buddy)->Arg(8);
BENCHMARK(BM_9x9_Conv2D_Buddy)->Arg(16);

// Generate result image.
void generateResultConv2D() {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> input(inputImageConv2D, sizesInputConv2D);
  MemRef<float, 2> kernel(laplacianKernelAlign, sizesKernel3x3Conv2D);
  MemRef<float, 2> output(sizesOutput3x3Conv2D);
  // Run the 2D convolution.
  _mlir_ciface_conv_2d(&input, &kernel, &output);

  // Define a cv::Mat with the output of the convolution.
  Mat outputImage(outputRows3x3Conv2D, outputCols3x3Conv2D, CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultConv2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
