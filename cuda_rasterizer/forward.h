/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* means3D,
		const float* normal,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* cov3D_precomp_small,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* means2D,
		float* depths,
		float* cov3Ds,
		float* cov3D_smalls,
		float* rgb,
		float4* conic_opacity1,
		float4* conic_opacity2,
		uint4* conic_opacity3,
		float4* conic_opacity4,
		float3* conic_opacity5,
		uint4* conic_opacity6,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		float3* save_normal);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* means2D,
		const float* colors,
		const float4* conic_opacity1,
		const float4* conic_opacity2,
		const float4* conic_opacity4,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		const float3* normal); 
}


#endif