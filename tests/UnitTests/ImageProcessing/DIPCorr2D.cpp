#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

#include "Utils/Container.h"
#include "kernels.h"

using namespace cv;
using namespace std;

// Declare the Corr2D C interface.
extern "C" {
void _mlir_ciface_corr_2d(MemRef<float, 2> *input, MemRef<float, 2> *kernel,
                          MemRef<float, 2> *output, unsigned int centerX,
                          unsigned int centerY, int boundaryOption);
}

// Read input image
Mat inputImage = imread("../../test_6x6.png",
                             IMREAD_GRAYSCALE);

int main()
{
  float* kernelArray = prewittKernelAlign;
  int kernelRows = prewittKernelRows, kernelCols = prewittKernelCols;

  // Define allocated, sizes, and strides.
  intptr_t sizesInput[2] = {inputImage.rows, inputImage.cols};
  intptr_t sizesKernel[2] = {kernelRows, kernelCols};
  intptr_t sizesOutput[2] = {inputImage.rows, inputImage.cols};

  // Define input, kernel, and output.
  MemRef<float, 2> input(inputImage, sizesInput);
  MemRef<float, 2> kernel(kernelArray, sizesKernel);
  MemRef<float, 2> output(sizesOutput);

  for (int i = 0; i < inputImage.rows; i++)
    for (int j = 0; j < inputImage.cols; j++)
      output[i * inputImage.rows + j] = 0;

  _mlir_ciface_corr_2d(&input, &kernel, &output, 1, 2, 0);
}
