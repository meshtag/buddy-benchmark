//===- BuddyRotate2DBenchmark.cpp -----------------------------------------===//
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
// This file implements the benchmark for Rotate2D operation.
//
//===----------------------------------------------------------------------===//

#include "Kernels.h"
#include <benchmark/benchmark.h>
#include <buddy/core/Container.h>
#include <buddy/core/ImageContainer.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare the conv2d C interface.
extern "C" {
// Declare the Rotate2D C interface.
void _mlir_ciface_rotate_2d(Img<float, 2> *input, float angleValue,
                            MemRef<float, 2> *output);
}

// Declare input image.
Mat inputImageBuddyRotate2D;

// Define the output size.
int outputRowsBuddyRotate2D, outputColsBuddyRotate2D;

// Define sizes of input and output.
intptr_t sizesInputBuddyRotate2D[2];
intptr_t sizesOutputBuddyRotate2D[2];

void initializeBuddyRotate2D(char **argv) {
  inputImageBuddyRotate2D = imread(argv[1], IMREAD_GRAYSCALE);

  outputRowsBuddyRotate2D = inputImageBuddyRotate2D.rows;
  outputColsBuddyRotate2D = inputImageBuddyRotate2D.cols;

  sizesInputBuddyRotate2D[0] = inputImageBuddyRotate2D.rows;
  sizesInputBuddyRotate2D[1] = inputImageBuddyRotate2D.cols;

  sizesOutputBuddyRotate2D[0] = outputRowsBuddyRotate2D;
  sizesOutputBuddyRotate2D[1] = outputColsBuddyRotate2D;
}

static void Buddy_Rotate2D(benchmark::State &state) {
  // Define the MemRef descriptor for input and output.
  Img<float, 2> inputBuddyRotate2D(inputImageBuddyRotate2D);
  MemRef<float, 2> outputBuddyRotate2D(sizesOutputBuddyRotate2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_rotate_2d(&inputBuddyRotate2D, 2.0944, &outputBuddyRotate2D);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyRotate2D() {
  BENCHMARK(Buddy_Rotate2D)
    ->Arg(1)
    ->Unit(benchmark::kMillisecond);
}

// Generate result image.
void generateResultBuddyRotate2D(char **argv) {
  // Define the MemRef descriptor for input and output.
  Img<float, 2> input(inputImageBuddyRotate2D);
  MemRef<float, 2> output(sizesOutputBuddyRotate2D);
  // Run the 2D rotation.
  _mlir_ciface_rotate_2d(&input, 2.0944, &output);

  // Define a cv::Mat with the output of the correlation.
  Mat outputImage(outputRowsBuddyRotate2D, outputColsBuddyRotate2D, CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultBuddyRotate2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
