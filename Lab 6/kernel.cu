#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <iostream>
#include <random>
#include <intrin.h>
#include <time.h>
#include "helper_cuda.h"
#include "helper_image.h"
#include "header.h"
#include "intrinsics_header.h"

#ifndef __CUDACC__

int __byte_perm(int i, int i1, int i2);
int __vadd2(int sum_column, const int buff_low);
int __vmaxs2(int sum_column, const int buff_low);
int __vmins2(int sum_column, const int buff_low);
void __syncthreads();

#endif

__global__ void calc_sector_cuda(int* input, int* res, uint64_t width, uint64_t height, uint64_t input_pitch, uint64_t res_pitch)
{
	const uint64_t buff_row_count = kCudaRowsToOneBlock + kCudaAdditionalRows;
	const uint64_t buff_column_count = kCudaColumnsNum * kChannelsNum + kCudaAdditionalColumns;

	__shared__ int input_buff[buff_row_count][buff_column_count];
	__shared__ int res_buff[kCudaRowsNum][kCudaColumnsNum * kChannelsNum];

	const uint64_t thread_x = threadIdx.x;
	const uint64_t thread_y = threadIdx.y;
	const uint64_t block_x = blockIdx.x;
	const uint64_t block_y = blockIdx.y;
	const uint64_t threads_in_block_x = blockDim.x;
	const uint64_t threads_in_block_y = blockDim.y;
	const uint64_t input_pitch_scaled = input_pitch / sizeof(int);
	const uint64_t res_pitch_scaled = res_pitch / sizeof(int);

	const uint64_t thread_basic_input_row = block_y * kCudaRowsToOneBlock + thread_y;
	const uint64_t thread_basic_input_column = block_x * threads_in_block_x * kChannelsNum + thread_x;

	//load main columns (with additional rows)
	for (uint8_t i = 0; i < kCudaRowsToOneThread || (thread_y < kCudaAdditionalRows && i < (kCudaRowsToOneThread + 1)); i++)
	{
		const uint64_t thread_load_y = thread_basic_input_row + i * threads_in_block_y;
		//load one sector (rgbr gbrg brgb)
		for (uint8_t j = 0; j < kChannelsNum; j++)
		{
			const uint64_t thread_load_x = thread_basic_input_column + j * threads_in_block_x;
			int input_num = 0;
			if (thread_load_y < (height + kCudaAdditionalRows) && (thread_load_x * sizeof(int)) < ((width + kCudaAdditionalColumns) * kChannelsNum))
				input_num = input[thread_load_y * input_pitch_scaled + thread_load_x];
			input_buff[thread_y + i * threads_in_block_y][thread_x + j * threads_in_block_x] = input_num;
		}
	}

	//load last additional column(s)
	for (uint8_t i = 0; i < kCudaAdditionalColumns; i++)
	{
		const uint64_t thread_save_y = thread_y * threads_in_block_x + thread_x;
		const uint64_t thread_save_x = kChannelsNum * threads_in_block_x + i;
		const uint64_t thread_load_y = block_y * kCudaRowsToOneBlock + thread_save_y;
		const uint64_t thread_load_x = kChannelsNum * block_x * threads_in_block_x + thread_save_x;
		if (thread_save_y < buff_row_count && thread_save_x < buff_column_count)
		{
			int input_num = 0;
			if (thread_load_y < (height + kCudaAdditionalRows) && (thread_load_x * sizeof(int)) < ((width + kCudaAdditionalColumns) * kChannelsNum))
				input_num = input[thread_load_y * input_pitch_scaled + thread_load_x];
			input_buff[thread_save_y][thread_save_x] = input_num;
		}
	}

	__syncthreads();

	//calc & save results
	for (uint8_t i = 0; i < kCudaRowsToOneThread; i++)
	{
		const uint64_t thread_save_y = thread_basic_input_row + i * threads_in_block_y;

		const uint64_t columns_to_calc = kChannelsNum + kCudaAdditionalColumns;
		const uint64_t rows_to_calc = 1 + kCudaAdditionalRows;

		int buff_single_color_before[kChannelsNum][rows_to_calc][2];
		int buff_multi_color_before[rows_to_calc][columns_to_calc];
		int buff_single_color_after[kChannelsNum];

		const uint64_t before_main_row_index = threads_in_block_y * i + thread_y;
		const uint64_t main_column_index = thread_x * kChannelsNum;

		//load multicolor from shared memo
		for (uint8_t k = 0; k < rows_to_calc; k++)
		{
			for (uint8_t j = 0; j < columns_to_calc; j++)
			{
				buff_multi_color_before[k][j] = input_buff[before_main_row_index + k][main_column_index + j];
			}
		}

		//sort colors (rgbr gbrg brgb) => (rrrr gggg bbbb)
		for (uint8_t k = 0; k < rows_to_calc; k++)
		{
			uint8_t block_index_in_single_color = 0;
			_CUSTOM_INTRINSIC_INT4_SORT_RGB_COLORS(
				buff_single_color_before[kCudaRedIndex][k][block_index_in_single_color],
				buff_single_color_before[kCudaGreenIndex][k][block_index_in_single_color],
				buff_single_color_before[kCudaBlueIndex][k][block_index_in_single_color],
				buff_multi_color_before[k][kCudaRgbrIndex + kChannelsNum * block_index_in_single_color],
				buff_multi_color_before[k][kCudaGbrgIndex + kChannelsNum * block_index_in_single_color],
				buff_multi_color_before[k][kCudaBrgbIndex + kChannelsNum * block_index_in_single_color]
			);

			block_index_in_single_color = 1;
			_CUSTOM_INTRINSIC_INT4_SORT_RGB_COLORS(
				buff_single_color_before[kCudaRedIndex][k][block_index_in_single_color],
				buff_single_color_before[kCudaGreenIndex][k][block_index_in_single_color],
				buff_single_color_before[kCudaBlueIndex][k][block_index_in_single_color],
				buff_multi_color_before[k][kCudaRgbrIndex + kChannelsNum * block_index_in_single_color],
				buff_multi_color_before[k][kCudaGbrgIndex + kChannelsNum * block_index_in_single_color],
				0
			);
		}

		//calc one color (channel)
		for (uint8_t k = 0; k < kChannelsNum; k++)
		{
			int sum_columns[3] = { 0 };
			int border_sum_for_mid[2] = { 0 };
			int mid[2] = { 0 };

			//sum values by columns & get middle values
			for (uint8_t j = 0; j < rows_to_calc; j++)
			{
				//add one row value
				const int buff = buff_single_color_before[k][j][0];
				const int buff_low = _CUSTOM_INTRINSIC_INT4_LOW_TO_INT2(buff);
				_CUSTOM_INTRINSIC_INT2_INCREMENT(sum_columns[0], buff_low);
				const int buff_high = _CUSTOM_INTRINSIC_INT4_HIGH_TO_INT2(buff);
				_CUSTOM_INTRINSIC_INT2_INCREMENT(sum_columns[1], buff_high);

				const int buff_next = buff_single_color_before[k][j][1];
				const int buff_next_low = _CUSTOM_INTRINSIC_INT4_LOW_TO_INT2(buff_next);
				_CUSTOM_INTRINSIC_INT2_INCREMENT(sum_columns[2], buff_next_low);

				if (j == 1)
				{
					//calc middle values
					mid[0] = _CUSTOM_INTRINSIC_INT2_MIDDLE(buff_low, buff_high);
					mid[1] = _CUSTOM_INTRINSIC_INT2_MIDDLE(buff_high, buff_next_low);
				}
			}

			for (uint8_t j = 0; j < 2; j++)
			{
				//sum values by row (what require)
				border_sum_for_mid[j] = _CUSTOM_INTRINSIC_INT2_MIDDLE(sum_columns[j], sum_columns[j + 1]);
				_CUSTOM_INTRINSIC_INT2_INCREMENT(border_sum_for_mid[j], sum_columns[j]);
				_CUSTOM_INTRINSIC_INT2_INCREMENT(border_sum_for_mid[j], sum_columns[j + 1]);
			
				//scale border values
				_CUSTOM_INTRINSIC_INT2_MUL_UPDATE(border_sum_for_mid[j], kOtherScale);

				//scale middle values
				_CUSTOM_INTRINSIC_INT2_MUL_UPDATE(mid[j], (kMiddleScale - kOtherScale));

				//calc sum of middle & border
				_CUSTOM_INTRINSIC_INT2_INCREMENT(mid[j], border_sum_for_mid[j]);

				//check with min value (must be 0 or higher, no lower)
				_CUSTOM_INTRINSIC_INT2_CHECK_0(mid[j]);

				//check with max value (must be 255 or lower, no higher)
				_CUSTOM_INTRINSIC_INT2_CHECK_255(mid[j]);
			}

			int answer = 0;

			//collect answer
			_CUSTOM_INTRINSIC_INT2_TO_INT4_LOW(answer, mid[0]);
			_CUSTOM_INTRINSIC_INT2_TO_INT4_HIGH(answer, mid[1]);

			//save results
			buff_single_color_after[k] = answer;
		}

		int buff_milti_color_after[kChannelsNum];

		//sort colors (rrrr gggg bbbb) => (rgbr gbrg brgb)
		_CUSTOM_INTRINSIC_INT4_SORT_RGB_COLORS_INVERT(
			buff_milti_color_after[kCudaRgbrIndex],
			buff_milti_color_after[kCudaGbrgIndex],
			buff_milti_color_after[kCudaBrgbIndex],
			buff_single_color_after[kCudaRedIndex],
			buff_single_color_after[kCudaGreenIndex],
			buff_single_color_after[kCudaBlueIndex]
		);

		//save results to shared memo (for warp)
		for (uint8_t k = 0; k < kChannelsNum; k++)
		{
			res_buff[thread_y][main_column_index + k] = buff_milti_color_after[k];
		}

		//save results to global memo (for warp)
		for (uint8_t k = 0; k < kChannelsNum; k++)
		{
			const uint64_t thread_save_x = thread_basic_input_column + k * threads_in_block_x;

			if (thread_save_y < (height) && (thread_save_x * sizeof(int)) < (width * kChannelsNum))
			{
				res[thread_save_y * res_pitch_scaled + thread_save_x] = res_buff[thread_y][k * threads_in_block_x + thread_x];
			}
		}
	}
}

double execute_gpu(const uint8_t* input, uint64_t width, uint64_t height, uint8_t* res)
{
	if (!cuda_init())
		return kErrorTime;

	double res_time = kErrorTime;
	cudaError_t err = cudaSuccess;
	uint8_t *input_cuda, *res_cuda;
	size_t input_pitch, res_pitch;
	err = cudaMallocPitch(&input_cuda, &input_pitch, width * kChannelsNum + kCudaAdditionalColumns * sizeof(int), height + kCudaAdditionalRows);
	checkCudaErrors(err);
	err = cudaMemset2DAsync(input_cuda, input_pitch, 0, width * kChannelsNum + kCudaAdditionalColumns * sizeof(int), height + kCudaAdditionalRows);
	checkCudaErrors(err);
	err = cudaMallocPitch(&res_cuda, &res_pitch, width * kChannelsNum, height);
	checkCudaErrors(err);

	cudaEvent_t begin, end;
	err = cudaEventCreate(&begin);
	checkCudaErrors(err);
	err = cudaEventCreate(&end);
	checkCudaErrors(err);

	err = cudaEventRecord(begin);
	checkCudaErrors(err);

	err = cudaMemcpy2DAsync(input_cuda + input_pitch + kChannelsNum, input_pitch, input, width * kChannelsNum, width * kChannelsNum * sizeof(uint8_t), height, cudaMemcpyHostToDevice);
	checkCudaErrors(err);

	dim3 thread_dim(kCudaColumnsNum, kCudaRowsNum);
	dim3 block_dim(static_cast<uint32_t>((width - 1) / (kCudaColumnsNum * sizeof(int)) + 1), static_cast<uint32_t>((height - 1) / (kCudaRowsToOneBlock)+1));
	calc_sector_cuda << <block_dim, thread_dim >> > (reinterpret_cast<int*>(input_cuda), reinterpret_cast<int*>(res_cuda), width, height, input_pitch, res_pitch);

	err = cudaMemcpy2DAsync(res, width * kChannelsNum, res_cuda, res_pitch, width * sizeof(uint8_t) * kChannelsNum, height, cudaMemcpyDeviceToHost);
	checkCudaErrors(err);

	err = cudaEventRecord(end);
	checkCudaErrors(err);

	err = cudaDeviceSynchronize();
	checkCudaErrors(err);

	float elapsed_time = .0;
	err = cudaEventElapsedTime(&elapsed_time, begin, end);
	checkCudaErrors(err);
	res_time = elapsed_time;

	cuda_deinit();
	return res_time;
}

int main()
{
	unsigned int width;
	unsigned int height;

	char filepath[] = "mountains_medium.ppm";
	char res_filepath[] = "res_GPU.ppm";
	char res_filepath_2[] = "res_CPU2.ppm";

	uint8_t* input_matrix = nullptr;
	unsigned channels = kChannelsNum;

	printf("Opening file \"%s\"\n", filepath);
	if (!__loadPPM(filepath, &input_matrix, &width, &height, &channels) || channels != kChannelsNum)
	{
		printf("Error: input file must be ppm RGB\n");
		if (input_matrix != NULL)
		{
			free(input_matrix);
		}
		system("PAUSE");
		return 1;
	}
	uint64_t full_size = width * height * channels;
	printf("Width = %d, height = %d\n", width, height);

	uint8_t* res_cpu = nullptr;
	if (!alloc_memo(res_cpu, full_size))
		system("PAUSE");
	uint8_t* res_gpu = nullptr;
	if (!alloc_memo(res_gpu, full_size))
		return 1;

	const double cpu_time = execute_cpu(input_matrix, width, height, res_cpu);
	printf("CPU execution time: %lf ms\n", cpu_time);
	const double gpu_time = execute_gpu(input_matrix, width, height, res_gpu);
	printf("GPU execution time: %lf ms\n", gpu_time);

	const bool are_answers_equal = (memcmp(res_cpu, res_gpu, full_size) == 0);
	std::cout << "Answers are " << (!are_answers_equal ? "not " : "") << "equal" << std::endl;

	printf("Writing result file \"%s\"\n", res_filepath);
	if (!__savePPM(res_filepath, res_gpu, width, height, channels))
		printf("Error: cannot save pgm RGB\n");

	printf("Writing result file \"%s\"\n", res_filepath_2);
	if (!__savePPM(res_filepath_2, res_cpu, width, height, channels))
		printf("Error: cannot save pgm RGB\n");

	free(input_matrix);
	free_memo(res_cpu);
	free_memo(res_gpu);

	system("PAUSE");
	return 0;
}

double execute_cpu(const uint8_t* a, uint64_t width, uint64_t height, uint8_t* res)
{
	int arr[8][2];
	clock_t begin = clock();
	width = width * kChannelsNum;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int check = a[width * y + x] * kMiddleScale;

			/*index00*/
			arr[0][1] = (y - 1);
			arr[0][0] = (x - kChannelsNum);
			/*index01*/
			arr[1][1] = (y - 1);
			arr[1][0] = x;
			/*index02*/
			arr[2][1] = (y - 1);
			arr[2][0] = (x + kChannelsNum);

			/*index10*/
			arr[3][1] = y;
			arr[3][0] = (x - kChannelsNum);
			/*index12*/
			arr[4][1] = y;
			arr[4][0] = (x + kChannelsNum);

			/*index20*/
			arr[5][1] = (y + 1);
			arr[5][0] = (x - kChannelsNum);
			/*index21*/
			arr[6][1] = (y + 1);
			arr[6][0] = x;
			/*index22*/
			arr[7][1] = (y + 1);
			arr[7][0] = (x + kChannelsNum);

			int sum = 0;

			for (int i = 0; i < 8; i++) {
				if (arr[i][0] < 0 || arr[i][0] >= width || arr[i][1] < 0 || arr[i][1] >= height) {
					continue;
				}
				//if (i % 2 != 0) continue;
				sum += a[width * arr[i][1] + arr[i][0]];
			}

			check += sum * kOtherScale;

			if (check < 0) {
				check = 0;
			}
			else if (check > 255) {
				check = 255;
			}
			res[width * y + x] = check;
		}
	}

	return (static_cast<double>(clock() - begin) / CLOCKS_PER_SEC * 1000.0);
}

bool alloc_memo(uint8_t*& a, uint64_t size)
{
	a = new uint8_t[size];
	if (a == nullptr)
	{
		std::cerr << "Cannot allocate memo" << std::endl;
		return false;
	}
	init_vector(a, size);
	return true;
}

void init_vector(uint8_t* a, uint64_t size)
{
	std::random_device rd;
	const double rand_scale = static_cast<double>(kRandomMax - kRandomMin) / (rd.max() - rd.min());
	for (uint64_t i = 0; i < size; ++i)
	{
#ifndef _DEBUG
		a[i] = static_cast<uint8_t>(rd() * rand_scale + kRandomMin);
#else
		a[i] = 1;
#endif
	}
}

void free_memo(uint8_t*& ptr)
{
	if (ptr != nullptr)
	{
		delete[] ptr;
		ptr = nullptr;
	}
}

bool cuda_init()
{
	int cuda_dev_counter = 0;
	cudaError_t err = cudaGetDeviceCount(&cuda_dev_counter);
	if (err != cudaSuccess || cuda_dev_counter <= 0)
	{
		std::cerr << "Cannot locate CUDA-supported devices. Error code = " << err << std::endl;
		return false;
	}
	err = cudaSetDevice(0);
	if (err != cudaSuccess)
	{
		std::cerr << "Cannot set CUDA-supported devices. Error code = " << err << std::endl;
		return false;
	}
	return true;
}

void cuda_deinit()
{
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "Cannot wait calculations finish. Error " << err << std::endl;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	err = cudaDeviceReset();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed! Error code = %d\n", err);
	}
}
