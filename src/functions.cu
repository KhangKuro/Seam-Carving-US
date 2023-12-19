#include <stdio.h>
#include <stdint.h>
#include "library.h" 
// Assuming this is a custom library header

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
    FILE * f = fopen(fileName, "r"); // Open the file for reading
    if (f == NULL)
    {
        printf("Cannot read %s\n", fileName); // If unable to open file, display an error message
        exit(EXIT_FAILURE); // Exit the program with failure status
    }

    char type[3];
    fscanf(f, "%s", type); // Read the file type (P3 or P2)

    if (strcmp(type, "P3") != 0) // Check if the file is of type P3
    {
        fclose(f); // Close the file
        printf("Cannot read %s\n", fileName); 
        exit(EXIT_FAILURE); // Exit if the file is not of type P3
    }

    fscanf(f, "%i", &width); // Read width of the image
    fscanf(f, "%i", &height); // Read height of the image

    int max_val;
    fscanf(f, "%i", &max_val); // Read maximum color value
    if (max_val > 255) // Ensure the maximum color value is within 1 byte range
    {
        fclose(f);
        printf("Cannot read %s\n", fileName); 
        exit(EXIT_FAILURE); 
    }

    pixels = (uchar3 *)malloc(width * height * sizeof(uchar3)); // Allocate memory for pixel data
    for (int i = 0; i < width * height; i++)
        fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z); // Read RGB values for each pixel

    fclose(f); // Close the file after reading all necessary data
}


void writePnm(uchar3 *pixels, int width, int height, int originalWidth, char *fileName)
{
    FILE * f = fopen(fileName, "w"); // Open the file for writing
    if (f == NULL)
    {
        printf("Cannot write %s\n", fileName); // Display an error message if unable to open the file
        exit(EXIT_FAILURE); // Exit the program with failure status
    }   

    fprintf(f, "P3\n%i\n%i\n255\n", width, height); // Write P3 format and image dimensions

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int i = r * originalWidth + c; // Calculate the index based on row and column
            fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z); // Write RGB values to the file
        }
    }
    
    fclose(f); // Close the file after writing the data
}


void convertRgb2Gray_host(uchar3 * rgbPic, int width, int height, uint8_t * grayPic) {
    // Loop through each row of the image
    for (int r = 0; r < height; ++r) {
        // Loop through each column of the image
        for (int c = 0; c < width; ++c) {
            // Calculate the index of the current pixel in the 1D array representation
            int i = r * width + c;
            
            // Convert the RGB pixel to grayscale using the formula:
            // Gray = 0.299 * Red + 0.587 * Green + 0.114 * Blue
            grayPic[i] = 0.299f * rgbPic[i].x + 0.587f * rgbPic[i].y + 0.114f * rgbPic[i].z;
        }
    }
}


void printDeviceInfo() {
    cudaDeviceProp devProv; // Declare a structure to hold device properties
    CHECK(cudaGetDeviceProperties(&devProv, 0)); // Get properties of the CUDA device (device 0)
    
    // Print GPU information
    printf("_____________GPU info_____________\n");
    printf("|Name:                   %s|\n", devProv.name); // GPU name
    printf("|Compute capability:          %d.%d|\n", devProv.major, devProv.minor); // Compute capability
    printf("|Num SMs:                      %d|\n", devProv.multiProcessorCount); // Number of Streaming Multiprocessors (SMs)
    printf("|Max num threads per SM:     %d|\n", devProv.maxThreadsPerMultiProcessor); // Max number of threads per SM
    printf("|Max num warps per SM:         %d|\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize); // Max number of warps per SM
    printf("|GMEM:           %zu byte|\n", devProv.totalGlobalMem); // Global memory size
    printf("|SMEM per SM:          %zu byte|\n", devProv.sharedMemPerMultiprocessor); // Shared memory per SM
    printf("|SMEM per block:       %zu byte|\n", devProv.sharedMemPerBlock); // Shared memory per block
    printf("|________________________________|\n");
}


float computeError(uchar3 * a1, uchar3 * a2, int n) {
    float err = 0; // Initialize the error variable to 0

    // Iterate through each element in the arrays
    for (int i = 0; i < n; i++) {
        // Calculate the absolute differences for each channel (R, G, B) and accumulate them
        err += abs((int)a1[i].x - (int)a2[i].x); // Absolute difference for the red channel
        err += abs((int)a1[i].y - (int)a2[i].y); // Absolute difference for the green channel
        err += abs((int)a1[i].z - (int)a2[i].z); // Absolute difference for the blue channel
    }

    // Normalize the error by dividing by the total number of elements times 3 (for R, G, B)
    err /= (n * 3);

    return err; // Return the computed error value
}


void printError(char * msg, uchar3 * in1, uchar3 * in2, int width, int height) {
    // Compute the error between the two arrays of uchar3 elements
    float err = computeError(in1, in2, width * height);
    
    // Print the message along with the computed error
    printf("%s: %f\n", msg, err);
}


char * concatStr(const char * s1, const char * s2) {
    // Allocate memory for the concatenated string
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    
    // Copy the content of the first string (s1) into the allocated memory
    strcpy(result, s1);
    
    // Append the content of the second string (s2) to the end of the result
    strcat(result, s2);
    
    // Return the concatenated string
    return result;
}
