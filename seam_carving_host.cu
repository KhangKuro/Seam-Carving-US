#include <stdio.h>
#include <stdint.h>
#include "./src/library.h"
using namespace std;

// Global variables
int WIDTH;  // Width variable

// Sobel filter kernels
int xSobel[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
int ySobel[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};


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

void measureEnergyUps(int *energy, int *minimalEnergy, int width, int height) {
    // Copy the bottom row of energy to minimalEnergy
    int lastRowIdx = (height - 1) * width;
    for (int c = 0; c < width; ++c) {
        minimalEnergy[lastRowIdx + c] = energy[lastRowIdx + c];
    }

    // Start from the second last row and compute minimalEnergy upwards
    for (int r = height - 2; r >= 0; --r) {
        for (int c = 0; c < width; ++c) {
            int idx = r * WIDTH + c; // Current index in minimalEnergy
            int belowIdx = (r + 1) * WIDTH  + c; // Index of pixel directly below

            int min = minimalEnergy[belowIdx]; // Initialize minimum energy with the pixel below

            // Check energy values of neighboring pixels below and update minimum if necessary
            if (c > 0 && minimalEnergy[belowIdx - 1] < min) {
                min = minimalEnergy[belowIdx - 1];
            }
            if (c < width - 1 && minimalEnergy[belowIdx + 1] < min) {
                min = minimalEnergy[belowIdx + 1];
            }

            minimalEnergy[idx] = min + energy[idx]; // Update minimalEnergy for the current pixel
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

    // Memory allocation for energy and minimalEnergy arrays
    int *energy = (int *)malloc(width * height * sizeof(int));
    int *minimalEnergy = (int *)malloc(width * height * sizeof(int));
    
    // Memory allocation and conversion of input RGB pixels to grayscale
    uint8_t *grayPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    convertRgb2Gray_host(inPixels, width, height, grayPixels);

    // Calculate energy for all pixels in the image
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            energy[r * WIDTH + c] = measurePixelEnergy(grayPixels, r, c, width, height);
        }
    }
    measureEnergyUps(energy, minimalEnergy, width, height);
    colorizeEnergy(minimalEnergy, outPixelsColor, width, height);

    while (width > targetedWidth) {
      // Calculate energy from the beginning. (go from top to bottom)
      measureEnergyUps(energy, minimalEnergy, width, height);

      // find min index of first row
      int minCol = 0, r = 0, prevMinCol;
      for (int c = 1; c < width; ++c) {
          if (minimalEnergy[r * WIDTH + c] < minimalEnergy[r * WIDTH + minCol])
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
              int min = minimalEnergy[belowIdx], minColCpy = minCol;
              if (minColCpy > 0 && minimalEnergy[belowIdx - 1] < min) {
                  min = minimalEnergy[belowIdx - 1];
                  minCol = minColCpy - 1;
              }
              if (minColCpy < width - 1 && minimalEnergy[belowIdx + 1] < min) {
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
    free(minimalEnergy);
    free(energy);

    // Stop the timer and print the execution time for the host function
    timer.Stop();
    timer.printTime((char *)"host");
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

    // Write results to files
    printf("\nImage color energy output size (width x height): %i x %i\n", width, height);
    writePnm(out_host_color, width, height, width, concatStr(argv[2], "_energy_host.pnm"));

    printf("\nImage output size (width x height): %i x %i\n", targetedWidth, height);
    writePnm(out_host, targetedWidth, height, width, concatStr(argv[2], "_host.pnm"));

    // Free allocated memory
    free(inPixels);
    free(out_host);
}

