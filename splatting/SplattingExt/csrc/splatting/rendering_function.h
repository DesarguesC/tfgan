#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RenderingCUDA(
	const torch::Tensor& coordinates,
    const torch::Tensor& features,
    const int image_width,
    const int image_height,
	const float radius);

std::tuple<torch::Tensor, torch::Tensor>
RenderingBackwardCUDA(
 	const torch::Tensor& coordinates,
	const torch::Tensor& features,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& geomBuffer,
	const int num_rendered, 
    const float radius,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer);
		
