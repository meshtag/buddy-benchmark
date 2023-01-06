//===- OpenCVResize2DBenchmark.cpp ----------------------------------------===//
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
// This file implements the benchmark for OpenCV Resize2D.
//
//===----------------------------------------------------------------------===//

#include "Kernels.h"
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare input image and output image.
Mat inputImageResize2D, outputResize2D;

// Declare OpenCVApproximation techniques supported.
enum OpenCVApproximation { nearest_neighbour, bilinear };

// Define OpenCVApproximation selected.
OpenCVApproximation OpenCVApproximationType;

void initializeOpenCVResize2D(char **argv) {
  inputImageResize2D = imread(argv[1], IMREAD_GRAYSCALE);

  if (static_cast<string>(argv[3]) == "NEAREST_NEIGHBOUR") {
    OpenCVApproximationType = nearest_neighbour;
  } else {
    OpenCVApproximationType = bilinear;
  }
}

// Benchmarking function.
static void OpenCV_Resize2D_Nearest_Neighbour(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
    resize(inputImageResize2D, outputResize2D, cv::Size(), 2.0, 2.0, INTER_NEAREST);
    }
  }
}

static void OpenCV_Resize2D_Bilinear(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
    resize(inputImageResize2D, outputResize2D, cv::Size(), 2.0, 2.0, INTER_LINEAR);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkOpenCVResize2D() {
  if (OpenCVApproximationType == nearest_neighbour) {
    BENCHMARK(OpenCV_Resize2D_Nearest_Neighbour)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } else {
    BENCHMARK(OpenCV_Resize2D_Bilinear)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  }
}

// Generate result image.
void generateResultOpenCVResize2D() {
  if (OpenCVApproximationType == nearest_neighbour) {
    resize(inputImageResize2D, outputResize2D, cv::Size(), 2.0, 2.0, INTER_NEAREST);
  } else {
    resize(inputImageResize2D, outputResize2D, cv::Size(), 2.0, 2.0, INTER_LINEAR);
  }

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result =
        imwrite("ResultOpenCVResize2D.png", outputResize2D, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
