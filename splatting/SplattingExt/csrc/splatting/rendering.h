#ifndef CUDA_RENDERING_H_INCLUDED
#define CUDA_RENDERING_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRendering
{
	class Rendering
	{
	public:
		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int N,
			const int width, int height, float radius,
			const float* coordinates,
			const float* features,
            float* out_color
            );
		static void backward(
			//input
			const int N, int num_rendered,
			const int width, int height, float radius,
			const float* coordinates,
			const float* features,
			char* geom_buffer,
			char* binning_buffer,
			char* img_buffer,
			const float* dL_dpix,
			//output
			float* dL_dcoordinates,
			float* dL_dfeatures
            );
	};
};

#endif