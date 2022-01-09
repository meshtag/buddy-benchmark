//===- Corr2DBenchmark.cpp ------------------------------------------------===//
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
// This file implements the benchmark for Corr2D operation.
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
void _mlir_ciface_corr_2d(MemRef<float, 2> *inputCorr2D,
                          MemRef<float, 2> *kernelCorr2D,
                          MemRef<float, 2> *outputCorr2D, unsigned int centerX,
                          unsigned int centerY, int boundaryOption);
}

// Read input image.
Mat inputImageCorr2D = imread(
    "../../benchmarks/ImageProcessing/Images/YuTu512.png", IMREAD_GRAYSCALE);

// Define the kernel size.
int kernelRows3x3Corr2D = sobel3x3KernelRows;
int kernelCols3x3Corr2D = sobel3x3KernelCols;

int kernelRows5x5Corr2D = sobel5x5KernelRows;
int kernelCols5x5Corr2D = sobel5x5KernelCols;

int kernelRows7x7Corr2D = sobel7x7KernelRows;
int kernelCols7x7Corr2D = sobel7x7KernelCols;

int kernelRows9x9Corr2D = sobel9x9KernelRows;
int kernelCols9x9Corr2D = sobel9x9KernelCols;

// Define the output size.
int outputRowsCorr2D = inputImageCorr2D.rows;
int outputColsCorr2D = inputImageCorr2D.cols;

// Define sizes of input, kernel, and output.
intptr_t sizesInputCorr2D[2] = {inputImageCorr2D.rows, inputImageCorr2D.cols};
intptr_t sizesKernel3x3Corr2D[2] = {kernelRows3x3Corr2D, kernelCols3x3Corr2D};
intptr_t sizesKernel5x5Corr2D[2] = {kernelRows5x5Corr2D, kernelCols5x5Corr2D};
intptr_t sizesKernel7x7Corr2D[2] = {kernelRows7x7Corr2D, kernelCols7x7Corr2D};
intptr_t sizesKernel9x9Corr2D[2] = {kernelRows9x9Corr2D, kernelCols9x9Corr2D};
intptr_t sizesOutputCorr2D[2] = {outputRowsCorr2D, outputColsCorr2D};

// Define the MemRef descriptor for input, kernel, and output.
MemRef<float, 2> inputCorr2D(inputImageCorr2D, sizesInputCorr2D);
MemRef<float, 2> kernel3x3Corr2D(sobel3x3KernelAlign, sizesKernel3x3Corr2D);
MemRef<float, 2> kernel5x5Corr2D(sobel5x5KernelAlign, sizesKernel5x5Corr2D);
MemRef<float, 2> kernel7x7Corr2D(sobel7x7KernelAlign, sizesKernel7x7Corr2D);
MemRef<float, 2> kernel9x9Corr2D(sobel9x9KernelAlign, sizesKernel9x9Corr2D);
MemRef<float, 2> outputCorr2D(sizesOutputCorr2D);

static void BM_3x3_DIP(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_corr_2d(&inputCorr2D, &kernel3x3Corr2D, &outputCorr2D,
                           1 /* Center X */, 1 /* Center Y */,
                           1 /* Boundary Option */);
    }
  }
}

// Register benchmarking function with different arguments.
BENCHMARK(BM_3x3_DIP)->Arg(1);
BENCHMARK(BM_3x3_DIP)->Arg(2);
BENCHMARK(BM_3x3_DIP)->Arg(4);
BENCHMARK(BM_3x3_DIP)->Arg(8);
BENCHMARK(BM_3x3_DIP)->Arg(16);

static void BM_5x5_Buddy(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_corr_2d(&inputCorr2D, &kernel5x5Corr2D, &outputCorr2D,
                           1 /* Center X */, 1 /* Center Y */,
                           1 /* Boundary Option */);
    }
  }
}

// // Register benchmarking function with different arguments.
// BENCHMARK(BM_5x5_Buddy)->Arg(1);
// BENCHMARK(BM_5x5_Buddy)->Arg(2);
// BENCHMARK(BM_5x5_Buddy)->Arg(4);
// BENCHMARK(BM_5x5_Buddy)->Arg(8);
// BENCHMARK(BM_5x5_Buddy)->Arg(16);

static void BM_7x7_Buddy(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_corr_2d(&inputCorr2D, &kernel7x7Corr2D, &outputCorr2D,
                           1 /* Center X */, 1 /* Center Y */,
                           1 /* Boundary Option */);
    }
  }
}

// // Register benchmarking function with different arguments.
// BENCHMARK(BM_7x7_Buddy)->Arg(1);
// BENCHMARK(BM_7x7_Buddy)->Arg(2);
// BENCHMARK(BM_7x7_Buddy)->Arg(4);
// BENCHMARK(BM_7x7_Buddy)->Arg(8);
// BENCHMARK(BM_7x7_Buddy)->Arg(16);

static void BM_9x9_Buddy(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_corr_2d(&inputCorr2D, &kernel9x9Corr2D, &outputCorr2D,
                           1 /* Center X */, 1 /* Center Y */,
                           1 /* Boundary Option */);
    }
  }
}

// // Register benchmarking function with different arguments.
// BENCHMARK(BM_9x9_Buddy)->Arg(1);
// BENCHMARK(BM_9x9_Buddy)->Arg(2);
// BENCHMARK(BM_9x9_Buddy)->Arg(4);
// BENCHMARK(BM_9x9_Buddy)->Arg(8);
// BENCHMARK(BM_9x9_Buddy)->Arg(16);

// Generate result image.
void generateResultCorr2D() {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> input(inputImageCorr2D, sizesInputCorr2D);
  MemRef<float, 2> kernel(laplacianKernelAlign, sizesKernel3x3Corr2D);
  MemRef<float, 2> output(sizesOutputCorr2D);
  // Run the 2D correlation.
  _mlir_ciface_corr_2d(&input, &kernel, &output, 1 /* Center X */,
                       1 /* Center Y */, 0 /* Boundary Option */);

  // Define a cv::Mat with the output of the correlation.
  Mat outputImage(outputRowsCorr2D, outputColsCorr2D, CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultCorr2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
