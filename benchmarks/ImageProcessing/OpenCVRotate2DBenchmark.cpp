//===- OpenCVRotate2DBenchmark.cpp ---------------------------------------===//
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
// This file implements the benchmark for OpenCV Rotate2D.
//
//===----------------------------------------------------------------------===//

#include "Kernels.h"
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat rotate(Mat src, double angle)   //rotate function returning mat object with parametres imagefile and angle    
{
    Mat dst;      //Mat object for output image file
    Point2f pt(src.cols/2., src.rows/2.);          //point from where to rotate    
    Mat r = getRotationMatrix2D(pt, angle, 1.0);      //Mat object for storing after rotation
    
    std::cout << r << "\n\n";

    warpAffine(src, dst, r, Size(src.cols, src.rows));  ///applie an affine transforation to image.
    return dst;
}

// Declare input image and output image.
Mat inputImageRotate2D, outputRotate2D;

void initializeOpenCVRotate2D(char **argv) {
  inputImageRotate2D = imread(argv[1], IMREAD_GRAYSCALE);
}

// Benchmarking function.
static void OpenCV_Rotate2D(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // outputRotate2D = rotate(inputImageRotate2D, 120);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkOpenCVRotate2D() {
  BENCHMARK(OpenCV_Rotate2D)
    ->Arg(1)
    ->Unit(benchmark::kMillisecond);
}

// Generate result image.
void generateResultOpenCVRotate2D() {
  outputRotate2D = rotate(inputImageRotate2D, 120);

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  std::cout << "Here\n";

  // Write output to PNG.
  bool result = false;
  try {
    result =
        imwrite("ResultOpenCVRotate2D.png", outputRotate2D, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
