//===- BuddyResize2DBenchmark.cpp -----------------------------------------===//
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
// This file implements the benchmark for Resize2D operation.
//
//===----------------------------------------------------------------------===//

#include "Kernels.h"
#include <benchmark/benchmark.h>
#include <buddy/core/Container.h>
#include <buddy/core/ImageContainer.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare the resize2d C interface.
extern "C" {
void _mlir_ciface_resize_2d_nearest_neighbour_interpolation(
    Img<float, 2> *inputBuddyResize2D, float horizontalScalingFactor,
    float verticalScalingFactor, MemRef<float, 2> *outputBuddyResize2D);

void _mlir_ciface_resize_2d_bilinear_interpolation(
    Img<float, 2> *inputBuddyResize2D, float horizontalScalingFactor,
    float verticalScalingFactor, MemRef<float, 2> *outputBuddyResize2D);
}

// Declare input image.
Mat inputImageBuddyResize2D;

// Define the output size.
int outputRowsBuddyResize2D, outputColsBuddyResize2D;

// Define sizes of input and output.
intptr_t sizesInputBuddyResize2D[2];
intptr_t sizesOutputBuddyResize2D[2];

// Declare approximation techniques supported.
enum Approximation { nearest_neighbour, bilinear };

// Define approximation selected.
Approximation ApproximationType;

void initializeBuddyResize2D(char **argv) {
  inputImageBuddyResize2D = imread(argv[1], IMREAD_GRAYSCALE);

  outputRowsBuddyResize2D = static_cast<size_t>(2.0 * inputImageBuddyResize2D.rows);
  outputColsBuddyResize2D = static_cast<size_t>(2.0 * inputImageBuddyResize2D.cols);

  sizesInputBuddyResize2D[0] = inputImageBuddyResize2D.rows;
  sizesInputBuddyResize2D[1] = inputImageBuddyResize2D.cols;

  sizesOutputBuddyResize2D[0] = outputRowsBuddyResize2D;
  sizesOutputBuddyResize2D[1] = outputColsBuddyResize2D;

  if (static_cast<string>(argv[4]) == "NEAREST_NEIGHBOUR") {
    ApproximationType = nearest_neighbour;
  } else {
    ApproximationType = bilinear;
  }
}

static void Buddy_Resize2D_Nearest_Neighbour(benchmark::State &state) {
  // Define the MemRef descriptor for input and output.
  Img<float, 2> inputBuddyResize2D(inputImageBuddyResize2D);
  MemRef<float, 2> outputBuddyResize2D(sizesOutputBuddyResize2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_resize_2d_nearest_neighbour_interpolation(
          &inputBuddyResize2D, 2.0, 2.0, &outputBuddyResize2D);
    }
  }
}

static void Buddy_Resize2D_Bilinear(benchmark::State &state) {
  // Define the MemRef descriptor for input and output.
  Img<float, 2> inputBuddyResize2D(inputImageBuddyResize2D);
  MemRef<float, 2> outputBuddyResize2D(sizesOutputBuddyResize2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_resize_2d_bilinear_interpolation(
          &inputBuddyResize2D, 2.0, 2.0, &outputBuddyResize2D);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyResize2D() {
  if (ApproximationType == nearest_neighbour) {
    BENCHMARK(Buddy_Resize2D_Nearest_Neighbour)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } else {
    BENCHMARK(Buddy_Resize2D_Bilinear)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  }
}

// Generate result image.
void generateResultBuddyResize2D(char **argv) {
  // Define the MemRef descriptor for input and output.
  Img<float, 2> input(inputImageBuddyResize2D);
  MemRef<float, 2> output(sizesOutputBuddyResize2D);
  // Run the 2D Resize operation.
  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    _mlir_ciface_resize_2d_nearest_neighbour_interpolation(&input, 2.0, 2.0, &output);
  } else {
    _mlir_ciface_resize_2d_bilinear_interpolation(&input, 2.0, 2.0, &output);
  }

  // Define a cv::Mat with the output of the Resize2D.
  Mat outputImage(outputRowsBuddyResize2D, outputColsBuddyResize2D, CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultBuddyResize2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
