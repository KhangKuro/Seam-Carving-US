#include <stdio.h>
#include <stdint.h>
#include "./src/library.h"
using namespace std;

// Global variables
int WIDTH;  // Width variable
__device__ int d_WIDTH;  // Device-side width variable

// Sobel filter kernels
int xSobel[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
int ySobel[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

// Constant memory for device
__constant__ int d_xSobel[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
__constant__ int d_ySobel[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

const int filterWidth = 3;  // Width of the filter

void checkInput(int argc, char **argv, int &width, int &height, uchar3 *&inPixels, int &targetedWidth, dim3 &blockSize) {
    // Check the number of arguments
    if (argc != 4 && argc != 6) {
        printf("The number of arguments is invalid\n");
        exit(EXIT_FAILURE);
    }

    // Read the file
    readPnm(argv[1], width, height, inPixels);
    printf("Image size (width x height): %i x %i\n\n", width, height);

    WIDTH = width; // Assign the width
    CHECK(cudaMemcpyToSymbol(d_WIDTH, &width, sizeof(int))); // Copy width to device constant

    // Check the width that the user wants to reach
    targetedWidth = atoi(argv[3]); // Convert user's input to integer

    // Validate user's chosen width
    if (targetedWidth <= 0 || targetedWidth >= width) {
        printf("Your chosen width must be between 0 and the current picture's width!\n");
        exit(EXIT_FAILURE);
    }

    // Handle block size
    if (argc == 6) {
        blockSize.x = atoi(argv[4]); // Set block x-size
        blockSize.y = atoi(argv[5]); // Set block y-size
    } 

    // Check if the GPU functions properly
    printDeviceInfo();
}

// HOST
int measurePixelEnergy(uint8_t *grayPixels, int row, int col, int width, int height) {
    int x_kernel = 0; // Initialize variable to store x-axis convolution result
    int y_kernel = 0; // Initialize variable to store y-axis convolution result

    for (int i = 0; i < 3; ++i) { // Loop through rows of the 3x3 filter matrix
        for (int j = 0; j < 3; ++j) { // Loop through columns of the 3x3 filter matrix

            // Ensure boundary conditions for the image
            // 0 <= row - 1 + i < height
            int r = min(max(0, row - 1 + i), height - 1); // Ensure row index stays within image boundaries
            // 0 <= col - 1 + j < width
            int c = min(max(0, col - 1 + j), width - 1); // Ensure column index stays within image boundaries

            uint8_t pixelVal = grayPixels[r * WIDTH + c]; // Access pixel value from the grayscale image

            // Apply Sobel filter convolution along x-axis and y-axis
            x_kernel += pixelVal * xSobel[i][j]; // Convolution with the x-axis Sobel filter
            y_kernel += pixelVal * ySobel[i][j]; // Convolution with the y-axis Sobel filter
        }
    }
    return abs(x_kernel) + abs(y_kernel); // Calculate energy by summing absolute values of the convolutions
}

void measureEnergyUps(int *energy, int *minEnergy, int width, int height) {
    // Copy the bottom row of energy to minEnergy
    int lastRowIdx = (height - 1) * width;
    for (int c = 0; c < width; ++c) {
        minEnergy[lastRowIdx + c] = energy[lastRowIdx + c];
    }

    // Start from the second last row and compute minEnergy upwards
    for (int r = height - 2; r >= 0; --r) {
        for (int c = 0; c < width; ++c) {
            int idx = r * WIDTH + c; // Current index in minEnergy
            int belowIdx = (r + 1) * WIDTH  + c; // Index of pixel directly below

            int min = minEnergy[belowIdx]; // Initialize minimum energy with the pixel below

            // Check energy values of neighboring pixels below and update minimum if necessary
            if (c > 0 && minEnergy[belowIdx - 1] < min) {
                min = minEnergy[belowIdx - 1];
            }
            if (c < width - 1 && minEnergy[belowIdx + 1] < min) {
                min = minEnergy[belowIdx + 1];
            }

            minEnergy[idx] = min + energy[idx]; // Update minEnergy for the current pixel
        }
    }
}

void colorizeEnergy(int *energy, uchar3 *colorPic, int width, int height) {
    int maxEnergy = 0; // Initialize maxEnergy

    // Find the maximum energy value
    for (int i = 0; i < width * height; ++i) {
        if (energy[i] > maxEnergy) {
            maxEnergy = energy[i];
        }
    }

    // Color the pixels based on normalized energy values
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x; // Calculate index for energy and color arrays

            // Normalize energy value to a range between 0 and 1
            float normalizedEnergy = (float)energy[idx] / maxEnergy;

            // Convert normalized energy to color values
            uchar3 color;
            color.x = (unsigned char)(125 * normalizedEnergy); // Red channel
            color.y = (unsigned char)(190 * normalizedEnergy); // Green channel
            color.z = (unsigned char)(190 * normalizedEnergy); // Blue channel

            // Assign the calculated color to the corresponding pixel in the output array
            colorPic[idx] = color;
        }
    }
}

void seamCarveHost(uchar3 *inPixels, int width, int height, int targetedWidth, uchar3 *outPixels, uchar3 *outPixelsColor) {
    GpuTimer timer;
    timer.Start();

    // Copy the input pixels to output pixels (initialization)
    memcpy(outPixels, inPixels, width * height * sizeof(uchar3));
    memcpy(outPixelsColor, inPixels, width * height * sizeof(uchar3));

    // Memory allocation for energy and minEnergy arrays
    int *energy = (int *)malloc(width * height * sizeof(int));
    int *minEnergy = (int *)malloc(width * height * sizeof(int));
    
    // Memory allocation and conversion of input RGB pixels to grayscale
    uint8_t *grayPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    convertRgb2Gray_host(inPixels, width, height, grayPixels);

    // Calculate energy for all pixels in the image
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            energy[r * WIDTH + c] = measurePixelEnergy(grayPixels, r, c, width, height);
        }
    }
    measureEnergyUps(energy, minEnergy, width, height);
    colorizeEnergy(minEnergy, outPixelsColor, width, height);

    while (width > targetedWidth) {
      // Calculate energy from the beginning. (go from top to bottom)
      measureEnergyUps(energy, minEnergy, width, height);

      // find min index of first row
      int minCol = 0, r = 0, prevMinCol;
      for (int c = 1; c < width; ++c) {
          if (minEnergy[r * WIDTH + c] < minEnergy[r * WIDTH + minCol])
              minCol = c;
      }

      // Find and remove seam from first to last row
      for (; r < height; ++r) {
          // remove seam pixel on row r
          for (int i = minCol; i < width - 1; ++i) {
              outPixels[r * WIDTH + i] = outPixels[r * WIDTH + i + 1];
              grayPixels[r * WIDTH + i] = grayPixels[r * WIDTH + i + 1];
              energy[r * WIDTH + i] = energy[r * WIDTH + i + 1];
          }
         
          // Update energy
          if (r > 0) {
              int affectedCol = max(0, prevMinCol - 2);

              while (affectedCol <= prevMinCol + 2 && affectedCol < width - 1) {
                  energy[(r - 1) * WIDTH + affectedCol] = measurePixelEnergy(grayPixels, r - 1, affectedCol, width - 1, height);
                  affectedCol += 1;
              }
          }

          // find to the bottom
          if (r < height - 1) {
              prevMinCol = minCol;

              int belowIdx = (r + 1) * WIDTH + minCol;
              int min = minEnergy[belowIdx], minColCpy = minCol;
              if (minColCpy > 0 && minEnergy[belowIdx - 1] < min) {
                  min = minEnergy[belowIdx - 1];
                  minCol = minColCpy - 1;
              }
              if (minColCpy < width - 1 && minEnergy[belowIdx + 1] < min) {
                  minCol = minColCpy + 1;
              }
          }
      }

      int affectedCol;
      for (affectedCol = max(0, minCol - 2); affectedCol <= minCol + 2 && affectedCol < width - 1; ++affectedCol) {
          energy[(height - 1) * WIDTH + affectedCol] = measurePixelEnergy(grayPixels, height - 1, affectedCol, width - 1, height);
      }

      --width;
    }


    // Free dynamically allocated memory
    free(grayPixels);
    free(minEnergy);
    free(energy);

    // Stop the timer and print the execution time for the host function
    timer.Stop();
    timer.printTime((char *)"host");
}


// Device
__global__ void kernelRgbToGray(uchar3 *inPixels, int width, int height, uint8_t *outPixels) {
    // Calculate the indices in the image for processing
    int r = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int c = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (r < height && c < width) { // Ensure threads are within the image bounds
        int i = r * width + c; // Calculate the linear index

        // Convert RGB pixel to grayscale using luminance formula
        outPixels[i] = 0.299f * inPixels[i].x + 0.587f * inPixels[i].y + 0.114f * inPixels[i].z;
    }
}

__global__ void calEnergy(uint8_t *inPixels, int width, int height, int *energy) {
    // Calculate the thread's row and column indices in the image
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate the energy for each thread
    if (col < width && row < height) {
        int x_kernel = 0, y_kernel = 0;

        for (int i = 0; i < filterWidth; ++i) {
            for (int j = 0; j < filterWidth; ++j) {
                int readRow = min(max(row - (filterWidth >> 1) + i, 0), height - 1);
                int readCol = min(max(col - (filterWidth >> 1) + j, 0), width - 1);
                
                uint8_t closest = inPixels[readRow * d_WIDTH + readCol];
                int filterIdx = i * filterWidth + j;
                x_kernel += closest * d_xSobel[filterIdx]; // Convolution with x-axis Sobel filter
                y_kernel += closest * d_ySobel[filterIdx]; // Convolution with y-axis Sobel filter
            }
        }

        energy[row * d_WIDTH + col] = abs(x_kernel) + abs(y_kernel);
    }
}


__global__ void calEnergyUpsKernel(int *energy, int *minEnergy, int width, int height, int fromRow) {
    size_t halfBlock = blockDim.x >> 1; // Half the block size

    int col = blockIdx.x * halfBlock - halfBlock + threadIdx.x; // Calculate column index

    if (fromRow == height - 1 && col < width) {
        minEnergy[fromRow * width + col] = energy[fromRow * width + col]; // Copy bottom row's energy to minEnergy
    }
    __syncthreads(); // Synchronize threads after copying bottom row

    // Interative computation of minimal energy upwards
    for (int stride = fromRow != height - 1 ? 0 : 1; stride < halfBlock && fromRow - stride >= 0; ++stride) {
        if (threadIdx.x < blockDim.x - (stride << 1)) {
            int curRow = fromRow - stride;
            int curCol = col + stride;

            // Ensure within bounds and process only valid columns
            if (curCol >= 0 && curCol < width) {
                int idx = curRow * d_WIDTH + curCol;
                int belowIdx = (curRow + 1) * d_WIDTH + curCol;

                int min = minEnergy[belowIdx]; // Initialize minimum energy with the pixel below

                // Update minimum energy by considering neighboring pixels below
                if (curCol > 0 && minEnergy[belowIdx - 1] < min)
                    min = minEnergy[belowIdx - 1];
                
                if (curCol < width - 1 && minEnergy[belowIdx + 1] < min)
                    min = minEnergy[belowIdx + 1];
                
                minEnergy[idx] = min + energy[idx]; // Update minEnergy for the current pixel
            }
        }
        __syncthreads(); // Synchronize threads after updating minEnergy
    }
}


__global__ void energyToColorKernel(int *energy, uchar3 *colorPic, int width, int height, int maxEnergy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the x index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Calculate the y index

    if (x < width && y < height) { // Ensure thread is within image bounds
        int idx = y * width + x; // Compute linear index

        // Normalize energy value to a range between 0 and 1
        float normalizedEnergy = (float)energy[idx] / maxEnergy;

        // Convert normalized energy to color values (RGB)
        uchar3 color;
        color.x = (unsigned char)(125 * (normalizedEnergy));  // Red channel
        color.y = (unsigned char)(190 * (normalizedEnergy));  // Green channel
        color.z = (unsigned char)(190 * (normalizedEnergy));  // Blue channel

        colorPic[idx] = color; // Assign calculated color to corresponding pixel in output array
    }
}

__global__ void carvingKernel(int * leastSignificantPixel, uchar3 * outPixels, uint8_t *grayPixels, int * energy, int width) {
    int row = blockIdx.x;
    int leastSignificant = leastSignificantPixel[row];

    // Update pixels only for threads whose index is within the range of leastSignificant to width - 1
    for (int i = leastSignificant + threadIdx.x; i < width - 1; i += blockDim.x) {
        int baseIdx = row * d_WIDTH + i;
        outPixels[baseIdx] = outPixels[baseIdx + 1];
        grayPixels[baseIdx] = grayPixels[baseIdx + 1];
        energy[baseIdx] = energy[baseIdx + 1];
    }
}

void findSeam(int * minEnergy, int *leastSignificantPixel, int width, int height) {
    int minCol = 0, r = 0; 

    for (int c = 1; c < width; ++c)
        if (minEnergy[r * WIDTH + c] < minEnergy[r * WIDTH + minCol])
            minCol = c;
    
    for (; r < height; ++r) { 
        leastSignificantPixel[r] = minCol;
        if (r < height - 1) { 
            int belowIdx = (r + 1) * WIDTH + minCol;
            int min = minEnergy[belowIdx], minColCpy = minCol;

            if (minColCpy > 0 && minEnergy[belowIdx - 1] < min) {
                min = minEnergy[belowIdx - 1];
                minCol = minColCpy - 1;
            }
            if (minColCpy < width - 1 && minEnergy[belowIdx + 1] < min) {
                minCol = minColCpy + 1;
            }
        }
    }
}

__global__ void findSeamKernel(int *minEnergy, int *leastSignificantPixel, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 

    if (col >= width) return; 

    int minCol = 0, r = 0;

    if (row == 0) { 
        for (int c = 1; c < width; ++c) {
            if (minEnergy[r * d_WIDTH + c] < minEnergy[r * d_WIDTH + minCol]) {
                minCol = c;
            }
        }
    }

    __syncthreads(); 

    for (; r < height; ++r) {
        leastSignificantPixel[r] = minCol;

        if (r < height - 1) {
            int belowIdx = (r + 1) * d_WIDTH + minCol;
            int min = minEnergy[belowIdx], minColCpy = minCol;

            if (minColCpy > 0 && minEnergy[belowIdx - 1] < min) {
                min = minEnergy[belowIdx - 1];
                minCol = minColCpy - 1;
            }

            if (minColCpy < width - 1 && minEnergy[belowIdx + 1] < min) {
                minCol = minColCpy + 1;
            }
        }

        __syncthreads(); 
    }
}


void seamCarveDevice(uchar3 *inPixels, int width, int height, int targetedWidth, uchar3 *outPixels, dim3 blockSize, uchar3 *outPixelsColor) {
    // GPU timer initialization
    GpuTimer timer;
    timer.Start();

    // Device memory allocation

    uchar3 *d_inPixels, *d_outPixels;
    uint8_t *d_grayPixels;
    int *d_energy, *d_minimalEnergy;
    CHECK(cudaMalloc(&d_inPixels, width * height * sizeof(uchar3)));
    CHECK(cudaMalloc(&d_outPixels, width * height * sizeof(uchar3)));
    CHECK(cudaMalloc(&d_grayPixels, width * height * sizeof(uint8_t)));
    CHECK(cudaMalloc(&d_energy, width * height * sizeof(int)));
    CHECK(cudaMalloc(&d_minimalEnergy, width * height * sizeof(int)));

    int * d_leastSignificantPixel;
    CHECK(cudaMalloc(&d_leastSignificantPixel, height * sizeof(int)));

    // Host memory allocation
    int *energy = (int *)malloc(width * height * sizeof(int));
    int * leastSignificantPixel = (int *)malloc(height * sizeof(int));
    int *minEnergy = (int *)malloc(width * height * sizeof(int));

	  int blockSizeDp = 256;
    int gridSizeDp = (((width - 1) / blockSizeDp + 1) << 1) + 1;
    int stripHeight = (blockSizeDp >> 1) + 1;

    // Copy input to device
    CHECK(cudaMemcpy(d_inPixels, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_outPixels, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));

    // Convert input image to grayscale on the device
    dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
    // convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_grayPixels);
    // cudaDeviceSynchronize();
    // CHECK(cudaGetLastError());
    uint8_t * grayPixels = (uint8_t *)malloc((width) * height * sizeof(uint8_t));
    uchar3 * tempPixels = (uchar3 *)malloc((width) * height * sizeof(uchar3));

    CHECK(cudaMemcpy(tempPixels, d_inPixels, (width) * height * sizeof(uchar3), cudaMemcpyDeviceToHost));
    
    convertRgb2Gray_host(tempPixels, width, height, grayPixels);

    CHECK(cudaMemcpy(d_grayPixels, grayPixels, (width) * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

    free(tempPixels);
    free(grayPixels);

    // update energy
    calEnergy<<<gridSize, blockSize>>>(d_grayPixels, width, height, d_energy);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Compute minimal seam table upwards in parallel
    for (int i = height - 1; i >= 0; i -= (stripHeight >> 1)) {
        calEnergyUpsKernel<<<gridSizeDp, blockSizeDp>>>(d_energy, d_minimalEnergy, width, height, i);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());
    }

    // Copy minimal energy table back to host memory to find max energy
    CHECK(cudaMemcpy(minEnergy, d_minimalEnergy, WIDTH * height * sizeof(int), cudaMemcpyDeviceToHost));
    int maxEnergy = 0;
    for (int i = 0; i < width * height; ++i) {
        if (minEnergy[i] > maxEnergy) {
            maxEnergy = minEnergy[i];
        }
    }
    // Convert energy values to color representation
    energyToColorKernel<<<gridSize, blockSize>>>(d_minimalEnergy, d_outPixels, width, height, maxEnergy);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());


    while (width > targetedWidth) {
        // update energy
        calEnergy<<<gridSize, blockSize>>>(d_grayPixels, width, height, d_energy);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        // Compute minimal seam table upwards in parallel
        for (int i = height - 1; i >= 0; i -= (stripHeight >> 1)) {
        calEnergyUpsKernel<<<gridSizeDp, blockSizeDp>>>(d_energy, d_minimalEnergy, width, height, i);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());
        }

        int numThreadsPerBlock = 256;
        int numBlocks = (width + numThreadsPerBlock - 1) / numThreadsPerBlock;
        findSeamKernel<<<numBlocks, numThreadsPerBlock>>>(d_minimalEnergy, d_leastSignificantPixel, width, height);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        carvingKernel<<<height, 1>>>(d_leastSignificantPixel, d_inPixels, d_grayPixels, d_energy, width);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());
        
        --width;
    }

    // Copy processed pixels back to host memory
    CHECK(cudaMemcpy(outPixels, d_inPixels, WIDTH * height * sizeof(uchar3), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(outPixelsColor, d_outPixels, WIDTH * height * sizeof(uchar3), cudaMemcpyDeviceToHost));
    

    // Free device memory
    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_outPixels));
    CHECK(cudaFree(d_grayPixels));
    CHECK(cudaFree(d_leastSignificantPixel));
    CHECK(cudaFree(d_energy));
    CHECK(cudaFree(d_minimalEnergy));

    // Free host memory
    free(minEnergy);
    free(leastSignificantPixel);
    free(energy);

    // Stop timer and print the execution time for the device function
    timer.Stop();
    timer.printTime((char *)"device");
}

// Main
int main(int argc, char **argv) {
    int width, height, targetedWidth;
    uchar3 *inPixels;
    dim3 blockSize(32, 32);

    // Check user's input
    checkInput(argc, argv, width, height, inPixels, targetedWidth, blockSize);

    // HOST: Perform energy calculation and color transformation on the CPU (host)
    uchar3 *out_host = (uchar3 *)malloc(width * height * sizeof(uchar3));
    uchar3 *out_host_color = (uchar3 *)malloc(width * height * sizeof(uchar3));
    seamCarveHost(inPixels, width, height, targetedWidth, out_host, out_host_color);

    // DEVICE: Perform energy calculation and color transformation on the GPU (device)
    uchar3 *out_device = (uchar3 *)malloc(width * height * sizeof(uchar3));
    uchar3 *out_device_color = (uchar3 *)malloc(width * height * sizeof(uchar3));
    seamCarveDevice(inPixels, width, height, targetedWidth, out_device, blockSize, out_device_color);

    // Compute error between device and host results
    printError((char *)"Error between device color result and host color result: ", out_host_color, out_device_color, targetedWidth, height);
    printError((char *)"Error between device result and host result: ", out_host, out_device, targetedWidth, height);

    // Write results to files
    printf("\nImage color energy output size (width x height) host: %i x %i\n", width, height);
    writePnm(out_host_color, width, height, width, concatStr(argv[2], "_energy_host.pnm"));

    printf("\nImage output size (width x height) host: %i x %i\n", targetedWidth, height);
    writePnm(out_host, targetedWidth, height, width, concatStr(argv[2], "_host.pnm"));

    printf("\nImage color energy output size (width x height) device: %i x %i\n", width, height);
    writePnm(out_device_color, width, height, width, concatStr(argv[2], "_energy_device.pnm"));

    printf("\nImage output size (width x height) device: %i x %i\n", targetedWidth, height);
    writePnm(out_device, targetedWidth, height, width, concatStr(argv[2], "_device.pnm"));

    // Free allocated memory
    free(inPixels);
    free(out_host);
    free(out_device);
}

