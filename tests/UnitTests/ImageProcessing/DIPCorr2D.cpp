#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <time.h>

#include "kernels.h"

using namespace cv;
using namespace std;

// Declare the Corr2D C interface.
extern "C" {
void _mlir_ciface_corr_2d(MemRef_descriptor input, MemRef_descriptor kernel,
                          MemRef_descriptor output, unsigned int centerX,
                          unsigned int centerY, int boundaryOption);
}

// Read input image
Mat inputImageDIP = imread("../../benchmarks/ImageProcessing/Images/YuTu.png",
                             IMREAD_GRAYSCALE);

// Define the kernel.
int kernelRows = laplacianKernelRows;
int kernelCols = laplacianKernelCols;

// Define allocated, sizes, and strides.
intptr_t sizesInput[2] = {inputImageDIP.rows, inputImageDIP.cols};
intptr_t sizesKernel[2] = {kernelRows, kernelCols};
intptr_t sizesOutput[2] = {inputImageDIP.rows, inputImageDIP.cols};

// Define input, kernel, and output.
MemRef<float, 2> input(inputImageDIP, sizesInput);
MemRef<float, 2> kernel(laplacianKernelAlign, sizesKernel);
MemRef<float, 2> output(sizesOutput);
