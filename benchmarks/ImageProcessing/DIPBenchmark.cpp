//===- DIPBenchmark.cpp -------------------------------------------------===//
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

// Declare the corr2d C interface.
extern "C" {
void _mlir_ciface_corr_2d(MemRef<float, 2> *input, MemRef<float, 2> *kernelDIP,
                          MemRef<float, 2> *outputDIP);
}

// Read input image
Mat inputImageDIP = imread("../../benchmarks/ImageProcessing/Images/YuTu.png",
                             IMREAD_GRAYSCALE);

// Define the kernelDIP.
int kernelDIPRows = laplacianKernelRows;
int kernelDIPCols = laplacianKernelCols;

// Define outputDIP for buddy mlir implementation.
int outputDIPRows = inputImageDIP.rows - kernelDIPRows + 1;
int outputDIPCols = inputImageDIP.cols - kernelDIPCols + 1;

// Define allocated, sizes, and strides.
intptr_t sizesInputDIP[2] = {inputImageDIP.rows, inputImageDIP.cols};
intptr_t sizeskernelDIP[2] = {kernelDIPRows, kernelDIPCols};
intptr_t sizesoutputDIP[2] = {outputDIPRows, outputDIPCols};

// Define input, kernelDIP, and outputDIP.
MemRef<float, 2> inputDIP(inputImageDIP, sizesInputDIP);
MemRef<float, 2> kernelDIP(laplacianKernelAlign, sizeskernelDIP);
MemRef<float, 2> outputDIP(sizesoutputDIP);

static void BM_DIP(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_corr_2d(&inputDIP, &kernelDIP, &outputDIP);
    }
  }
}

// Register benchmarking function with different arguments
BENCHMARK(BM_DIP)->Arg(1);
BENCHMARK(BM_DIP)->Arg(2);
BENCHMARK(BM_DIP)->Arg(4);
BENCHMARK(BM_DIP)->Arg(8);
BENCHMARK(BM_DIP)->Arg(16);
