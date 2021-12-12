#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <time.h>

#include "Utils/Container.h"
#include "kernels.h"

using namespace cv;
using namespace std;

// Declare the Corr2D C interface.
extern "C" {
void _mlir_ciface_corr_2d(MemRef<float, 2> input, MemRef<float, 2> kernel,
                          MemRef<float, 2> output, unsigned int centerX,
                          unsigned int centerY, int boundaryOption);
}

bool equalImages(const Mat & a, const Mat & b)
{
    if ((a.rows != b.rows) || (a.cols != b.cols))
        return false;
    Scalar s = sum(a - b);
    return (s[0]==0) && (s[1]==0) && (s[2]==0);
}

// Read input image
Mat inputImage = imread("../../benchmarks/ImageProcessing/Images/YuTu.png",
                             IMREAD_GRAYSCALE);

void testKernel(unsigned int kernelRows, unsigned int kernelCols, float* kernelArray)
{
  // Define allocated, sizes, and strides.
  intptr_t sizesInput[2] = {inputImage.rows, inputImage.cols};
  intptr_t sizesKernel[2] = {kernelRows, kernelCols};
  intptr_t sizesOutput[2] = {inputImage.rows, inputImage.cols};

  // Define input, kernel, and output.
  MemRef<float, 2> input(inputImage, sizesInput);
  MemRef<float, 2> kernel(kernelArray, sizesKernel);
  MemRef<float, 2> output(sizesOutput);

  // Define a cv::Mat with the output of the corr_2d.
  Mat dipOutput(inputImage.rows, inputImage.cols, CV_32FC1, output.getData());
  Mat kernel1 = Mat(3, 3, CV_32FC1, kernelArray);
  Mat opencvOutput;

  for (unsigned int x = 0; x < kernelRows; ++x)
  {
    for (unsigned int y = 0; y < kernelCols; ++y)
    {
      _mlir_ciface_corr_2d(input, kernel, output, x, y, 1);
      filter2D(inputImage, opencvOutput, CV_8UC1, kernel1, cv::Point(x, y), 0.0,
           cv::BORDER_REPLICATE);

      if (!equalImages(dipOutput, opencvOutput))
      {
        std::cout << "Different images produced by OpenCV and DIP for kernel " << kernel1 << "\n";
        return;
      }
    }
  }
}

int main()
{

}
