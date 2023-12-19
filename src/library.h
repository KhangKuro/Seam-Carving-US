#ifndef HEADER_H
#define HEADER_H

#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

/**
 * Working with files
 * File types:
 * - .pnm file with "P2" or "P3" header
 * - filter (plain text)
 */

// Function to read a .pnm file and retrieve its properties and pixel data.
void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels);

// Function to write pixel data to a .pnm file.
void writePnm(uchar3 *pixels, int width, int height, int originalWidth, char *fileName);

/**
 * convert RGB to Grayscale
 */

// Function to convert an RGB image to grayscale using the CPU (host).
void convertRgb2Gray_host(uchar3 * rgbPic, int width, int height, uint8_t * grayPic);

/**
 * Another functions
 */

// Function to print information about the device (e.g., GPU) being used.
void printDeviceInfo();

// Function to print an error message along with additional information (two input images and dimensions).
void printError(char * msg, uchar3 * in1, uchar3 * in2, int width, int height);

// Function to concatenate two strings and return the result.
char * concatStr(const char * s1, const char * s2);

/**
 * Time counter
 */
struct GpuTimer {
    cudaEvent_t start;  // CUDA event to mark the start time
    cudaEvent_t stop;   // CUDA event to mark the stop time

    // Constructor: Creates CUDA events for start and stop
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    // Destructor: Destroys CUDA events
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Marks the start time using a CUDA event
    void Start() {
        cudaEventRecord(start, 0);  // Record the start event
        cudaEventSynchronize(start); // Ensure the event is recorded before continuing
    }

    // Marks the stop time using a CUDA event
    void Stop() {
        cudaEventRecord(stop, 0); // Record the stop event
    }

    // Computes and returns the elapsed time in milliseconds
    float Elapsed() {
        float elapsed;
        cudaEventSynchronize(stop); // Ensure the stop event is recorded
        cudaEventElapsedTime(&elapsed, start, stop); // Calculate the elapsed time
        return elapsed; // Return the elapsed time in milliseconds
    }

    // Prints the processing time with a provided message
    void printTime(char * s) {
        printf("Processing time of %s: %f ms\n\n", s, Elapsed());
    }
};

#endif