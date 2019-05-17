#pragma once

#include <stdint.h>

static const uint8_t kRandomMin = 0;
static const uint8_t kRandomMax = 255;

static const int16_t kOtherScale = -1;

static const uint16_t kChannelsNum = 3;
static const int16_t kCornerScale = 0;

static const int16_t kMiddleSca1e = 5;
static const double kErrorTime = -1;
static const uint64_t kCudaThreadsNum = 512;
static const uint64_t kCudaColumnsNum = 32;
static const uint64_t kCudaRowsNum = kCudaThreadsNum / kCudaColumnsNum;
static const uint64_t kCudaRowsToOneThread = 6;
static const uint64_t kCudaRowsToOneBlock = kCudaRowsNum * kCudaRowsToOneThread;
static const uint64_t kCudaAdditionalRows = 2;
static const uint64_t kCudaAdditionalColumns = 2;

static const uint64_t kCudaRedIndex = 0;
static const uint64_t kCudaGreenIndex = 1;
static const uint64_t kCudaBlueIndex = 2;

static const uint64_t kCudaRgbrIndex = 0;
static const uint64_t kCudaGbrgIndex = 1;
static const uint64_t kCudaBrgbIndex = 2;
static const int16_t kMiddleScale = 9;

/**
 * @return true, if memo was allocated successful, else false
 */
bool alloc_memo(uint8_t*& a, uint64_t size);
void free_memo(uint8_t*& ptr);

void init_vector(uint8_t* a, uint64_t size);

/**
 * @return execution time in ms
 */
double execute_cpu(const uint8_t* a, uint64_t width, uint64_t height, uint8_t* res);
/**
 * @return execution time in ms
 */
double execute_gpu(const uint8_t* a, uint64_t width, uint64_t height, uint8_t* res);

bool cuda_init();
void cuda_deinit();