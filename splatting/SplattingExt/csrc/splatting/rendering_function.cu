#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "config.h"
#include "rendering_function.h"
#include "rendering.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RenderingCUDA(
	const torch::Tensor& coordinates,
  const torch::Tensor& features,
  const int image_width,
  const int image_height,
	const float radius)
{  
  const int N = coordinates.size(0);
  const int H = image_height;
  const int W = image_width;
  const float R = radius;

  auto int_opts = coordinates.options().dtype(torch::kInt32);
  auto float_opts = coordinates.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({N}, 0, int_opts);
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(N != 0)
  {
    rendered = CudaRendering::Rendering::forward(
      geomFunc,
      binningFunc,
      imgFunc,
      N,
      W, H, R,
      coordinates.contiguous().data<float>(),
      features.contiguous().data<float>(), 
      out_color.contiguous().data<float>()
    );
  }
  return std::make_tuple(rendered, out_color, geomBuffer, binningBuffer, imgBuffer);
}


std::tuple<torch::Tensor, torch::Tensor>
RenderingBackwardCUDA(
 	const torch::Tensor& coordinates,
	const torch::Tensor& features,
  const torch::Tensor& dL_dout_color,
	const torch::Tensor& geomBuffer,
	const int num_rendered, 
  const float radius,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer)
{
  const int N = coordinates.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  const float R = radius;

  torch::Tensor dL_dcoordinates = torch::zeros({N, 3}, coordinates.options());
  torch::Tensor dL_dfeatures = torch::zeros({N, 4}, coordinates.options());
  
  if(N != 0)
  {  
	  CudaRendering::Rendering::backward(
    N, num_rendered,
	  W, H, R,
	  coordinates.contiguous().data<float>(),
	  features.contiguous().data<float>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dcoordinates.contiguous().data<float>(),
	  dL_dfeatures.contiguous().data<float>()
    );
  }

  return std::make_tuple(dL_dcoordinates, dL_dfeatures);
}