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

#include "forward.h"
#include "auxiliary.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
// __device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
// {
// 	// The following models the steps outlined by equations 29
// 	// and 31 in "EWA Splatting" (Zwicker et al., 2002).
// 	// Additionally considers aspect / scaling of viewport.
// 	// Transposes used to account for row-/column-major conventions.
// 	float3 t = transformPoint4x3(mean, viewmatrix);
//
// 	const float limx = 1.3f * tan_fovx;
// 	const float limy = 1.3f * tan_fovy;
// 	const float txtz = t.x / t.z;
// 	const float tytz = t.y / t.z;
// 	t.x = min(limx, max(-limx, txtz)) * t.z;
// 	t.y = min(limy, max(-limy, tytz)) * t.z;
//
// 	glm::mat3 J = glm::mat3(
// 		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
// 		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
// 		0, 0, 0);
//
// 	glm::mat3 W = glm::mat3(
// 		viewmatrix[0], viewmatrix[4], viewmatrix[8],
// 		viewmatrix[1], viewmatrix[5], viewmatrix[9],
// 		viewmatrix[2], viewmatrix[6], viewmatrix[10]);
//
// 	glm::mat3 T = W * J;
//
// 	glm::mat3 Vrk = glm::mat3(
// 		cov3D[0], cov3D[1], cov3D[2],
// 		cov3D[1], cov3D[3], cov3D[4],
// 		cov3D[2], cov3D[4], cov3D[5]);
//
// 	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
//
// 	// Apply low-pass filter: every Gaussian should be at least
// 	// one pixel wide/high. Discard 3rd row and column.
// 	cov[0][0] += 0.3f;
// 	cov[1][1] += 0.3f;
// 	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
// }
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002).
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

// 	glm::mat3 W = glm::mat3(
// 		viewmatrix[0], viewmatrix[4], viewmatrix[8],
// 		viewmatrix[1], viewmatrix[5], viewmatrix[9],
// 		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

//     glm::mat3 T = W*J;
	glm::mat3 T = J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}


__device__ float6 computeCov2D_halfgs(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, float6 view_cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002).
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	// Change J for half gaussian
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		(focal_x / t.z), 0.0, (-(focal_x * t.x) / (t.z * t.z)),
		0.0, (focal_y / t.z), (-(focal_y * t.y) / (t.z * t.z)),
		0, 0, (focal_y / t.z));

	glm::mat3 W = glm::mat3(
		(viewmatrix[0]), (viewmatrix[4]), (viewmatrix[8]),
		(viewmatrix[1]), (viewmatrix[5]), (viewmatrix[9]),
		(viewmatrix[2]), (viewmatrix[6]), (viewmatrix[10]));

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

    glm::mat3 rot_cov = glm::transpose(W) * glm::transpose(Vrk) * W;

    view_cov3D.x = rot_cov[0][0];
    view_cov3D.y = rot_cov[0][1];
    view_cov3D.z = rot_cov[1][1];
    view_cov3D.w = rot_cov[0][2];
    view_cov3D.u = rot_cov[1][2];
    view_cov3D.v = rot_cov[2][2];

//     glm::mat3 cov = glm::transpose(J) * glm::transpose(rot_cov) * J;


	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	cov[2][2] += 0.3f;
	return {cov[0][0], cov[0][1], cov[1][1], cov[0][2], cov[1][2], cov[2][2]};//{ float(cov[0][0]), float(cov[0][1]), float(cov[1][1]), float(cov[0][2]), float(cov[1][2]), float(cov[2][2])};
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	glm::mat3 S_inv = glm::mat3(1.0f);
	S_inv[0][0] = 1/S[0][0];
	S_inv[1][1] = 1/S[1][1];
	S_inv[2][2] = 1/S[2][2];

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	//glm::mat3 M_inv = S_inv * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;
	//glm::mat3 Sigma_inv = glm::transpose(M_inv) * M_inv;


	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];

    //float6 cov3D_inv;

    //glm::mat3 test = Sigma*Sigma_inv - glm::mat3(1.0f);
    //float sum_result =fabs(test[0][0])+fabs(test[0][1])+fabs(test[0][2]) + fabs(test[1][0])+fabs(test[1][1])+fabs(test[1][2])+fabs(test[2][0])+fabs(test[2][1])+fabs(test[2][2]);

	//cov3D_inv.x = sum_result >= 0.01f?0.f:Sigma_inv[0][0];
	//cov3D_inv.y = sum_result >= 0.01f?0.f:Sigma_inv[0][1];
	//cov3D_inv.z = sum_result >= 0.01f?0.f:Sigma_inv[0][2];
	//cov3D_inv.w = sum_result >= 0.01f?0.f:Sigma_inv[1][1];
	//cov3D_inv.u = sum_result >= 0.01f?0.f:Sigma_inv[1][2];
	//cov3D_inv.v = sum_result >= 0.01f?0.f:Sigma_inv[2][2];

//     glm::mat3 test = Sigma*Sigma_inv;
// //     if(test[0][0]+test[1][1]+test[2][2]>3.1 || test[0][0]+test[1][1]+test[2][2]<2.9)
// //     {return cov3D_inv;} //save all 0 if the gaussian is too small, then inv value will be too large
//
// 	cov3D_inv.x = (test[0][0]+test[1][1]+test[2][2]>3.01 || test[0][0]+test[1][1]+test[2][2]<2.99)?0.f:Sigma_inv[0][0];
// 	cov3D_inv.y = (test[0][0]+test[1][1]+test[2][2]>3.01 || test[0][0]+test[1][1]+test[2][2]<2.99)?0.f:Sigma_inv[0][1];
// 	cov3D_inv.z = (test[0][0]+test[1][1]+test[2][2]>3.01 || test[0][0]+test[1][1]+test[2][2]<2.99)?0.f:Sigma_inv[0][2];
// 	cov3D_inv.w = (test[0][0]+test[1][1]+test[2][2]>3.01 || test[0][0]+test[1][1]+test[2][2]<2.99)?0.f:Sigma_inv[1][1];
// 	cov3D_inv.u = (test[0][0]+test[1][1]+test[2][2]>3.01 || test[0][0]+test[1][1]+test[2][2]<2.99)?0.f:Sigma_inv[1][2];
// 	cov3D_inv.v = (test[0][0]+test[1][1]+test[2][2]>3.01 || test[0][0]+test[1][1]+test[2][2]<2.99)?0.f:Sigma_inv[2][2];

// 	cov3D_inv.x = Sigma_inv[0][0];
// 	cov3D_inv.y = Sigma_inv[0][1];
// 	cov3D_inv.z = Sigma_inv[0][2];
// 	cov3D_inv.w = Sigma_inv[1][1];
// 	cov3D_inv.u = Sigma_inv[1][2];
// 	cov3D_inv.v = Sigma_inv[2][2];

	 //////

//     {printf("wrong,");}
    //{printf("inverse is not very accurate, %f, result is:%f\n",cov3D_inv.x, test[0][0]+test[1][1]+test[2][2]);}
//     printf("test inv: %f,%f,%f;%f,%f,%f,%f\n",test[0][0],test[1][1],test[2][2],test[0][1],test[0][2],test[1][2],cov3D_inv.x);
	//////

	//return cov3D_inv;
}

// __device__ void generate_type(float3 cov, float op1, float op2, float& type, float3 normal, float x_term, float y_term)
// {
//     float diagional; //determine if the ellipsoid is on diagional direction or not, in diagional is 1, anti-dia is 0
//     float leftright; //determine if op1 is on left or right, left is 1, right is 0
//
//     float full = (fabsf(op1-op2)<0.0039)?0.f:1.f; //if type is 0, then we use the whole gaussian
//     float lambda = ((cov.x+cov.z) + sqrtf((cov.x+cov.z)*(cov.x+cov.z) - 4*(cov.x*cov.z-cov.y*cov.y)))/2;
//
//     float2 vec = {lambda - cov.z - cov.y, cov.x - lambda + cov.y}; //find the long axis direction
//     float op_small = (op1-op2)>0.f?0.0f:1.0f; // p_vec will direct to op1, if op1 is not small, then we use these part, else the other part
//     //find the angle between vec and [0,1], which will tell us diagional
//     diagional = vec.x*vec.y<0.0f ? 1.0f : 0.0f; //use mean as center, vector in diagional(1,4) or in anti-diagional(2,3)
//     float temp_left_right = ((normal.x*1.4142135f+x_term)*vec.x+(normal.y*1.4142135f+y_term)*vec.y) > 0.f? vec.x:-vec.x;
//     leftright = temp_left_right<0 ? op_small-0.f:1.f-op_small;
// //     leftright = p_vec<0.0f ? 1.0f-op_small : op_small-0.0f; // left part is 1, right is 0
//
//     type = full*(diagional*1.0f*leftright + (1.0f-diagional)*2.0f*(1.0f-leftright) + (1.0f-diagional)*3.0f*leftright + diagional*4.0f*(1.0f-leftright));
// }

// __device__ void generate_type(float3 cov, float3 cov_small, float op1, float op2, float& type, float3 normal, float x_term, float y_term, float3 conic)
// {
//     float diagional; //determine if the ellipsoid is on diagional direction or not, in diagional is 1, anti-dia is 0
//     float leftright; //determine if op1 is on left or right, left is 1, right is 0
//
//     float full = (fabsf(op1-op2)<0.0039)?0.f:1.f; //if type is 0, then we use the whole gaussian
//     float lambda = ((cov.x+cov.z) + sqrtf((cov.x+cov.z)*(cov.x+cov.z) - 4*(cov.x*cov.z-cov.y*cov.y)))/2;
//
//     float lambda_small = ((cov_small.x+cov_small.z) + sqrtf((cov_small.x+cov_small.z)*(cov_small.x+cov_small.z) - 4*(cov_small.x*cov_small.z-cov_small.y*cov_small.y)))/2;
//
//     float2 vec = {lambda - cov.z - cov.y, cov.x - lambda + cov.y}; //find the long axis direction
//
//     float2 vec_small = {lambda_small - cov_small.z - cov_small.y, cov_small.x - lambda_small + cov_small.y}; //find the long axis direction
//     float2 vec_small_short = {-vec_small.y, vec_small.x};
//
//     float op_small = (op1-op2)>0.f?0.0f:1.0f; // p_vec will direct to op1, if op1 is not small, then we use these part, else the other part
//     //find the angle between vec and [0,1], which will tell us diagional
//     //diagional = vec.x*vec.y<0.0f ? 1.0f : 0.0f; //use mean as center, vector in diagional(1,4) or in anti-diagional(2,3)
//
//     float change= ((normal.x*1.4142135f+x_term)*vec_small_short.x+(normal.y*1.4142135f+y_term)*vec_small_short.y) > 0.f?1.0f:0.f;
//     change = change<0 ? op_small-0.f:1.f-op_small;
// //     vec_small_short.x = change==1.f? vec_small_short.x:-vec_small_short.x;
// //     vec_small_short.y = change==1.f? vec_small_short.y:-vec_small_short.y;
//     float temp_left_right= ((normal.x*1.4142135f+x_term)*vec.x+(normal.y*1.4142135f+y_term)*vec.y) > 0.f?1.0f:0.f;
//     leftright = temp_left_right<0 ? op_small-0.f:1.f-op_small;
//
//
// //     leftright = p_vec<0.0f ? 1.0f-op_small : op_small-0.0f; // left part is 1, right is 0
//     //generate diagional
//     float cx = conic.x;
//     float cy = conic.y;
//     float cz = conic.z;
//     float det2 = 1/sqrtf(cx*cy*cy - 2*cx*cy*cy + cz*cx*cx);
//     float2 point3 = {3*cy*det2, -3*cx*det2};
//     float2 point1 = {-3*cy*det2, 3*cx*det2};
//     det2 = 1/sqrtf(cx*cz*cz - 2*cz*cy*cy + cz*cy*cy);
//     float2 point2 = {3*cz*det2, -3*cy*det2};
//     float2 point4 = {-3*cz*det2, 3*cy*det2};
//
//     float2 vertical_point4 = {point4.y, -point4.x};
//     float2 vertical_point2 = {point2.y, -point2.x};
//
// //     float temp_left_right = ;
//
//     diagional = ((point2.x*vec_small.y-point2.y*vec_small.x)>0 && (point1.x*vec_small.y-point1.y*vec_small.x)<0) || ((point4.x*vec_small.y-point4.y*vec_small.x)<0 && (point3.x*vec_small.y-point3.y*vec_small.x)>0);
//
// //     leftright = ((vertical_point2.x*vec_small_short.y-vertical_point2.y*vec_small_short.x)<0 && (vertical_point4.x*vec_small_short.y-vertical_point4.y*vec_small_short.x)>0) ? 1.f:0.f;;
//
//     type = full*(diagional*1.0f*leftright + (1.0f-diagional)*2.0f*(1.0f-leftright) + (1.0f-diagional)*3.0f*leftright + diagional*4.0f*(1.0f-leftright));
// }

__device__ void generate_type(float3 cov, float3 cov_small, float op1, float op2, float& type, float3 normal, float x_term, float y_term, float3 conic)
{
    //float diagional; //determine if the ellipsoid is on diagional direction or not, in diagional is 1, anti-dia is 0
    //float leftright; //determine if op1 is on left or right, left is 1, right is 0

    float full = (fabsf(op1-op2)<0.0039)?0.f:1.f; //if type is 0, then we use the whole gaussian
    //float lambda = ((cov.x+cov.z) + sqrtf((cov.x+cov.z)*(cov.x+cov.z) - 4*(cov.x*cov.z-cov.y*cov.y)))/2;

    //float2 vec = {lambda - cov.z - cov.y, cov.x - lambda + cov.y}; //find the long axis direction

//     float op_small = (op1-op2)>0.f?0.0f:1.0f; // p_vec will direct to op1, if op1 is not small, then we use these part, else the other part
    //find the angle between vec and [0,1], which will tell us diagional
    //diagional = vec.x*vec.y<0.0f ? 1.0f : 0.0f; //use mean as center, vector in diagional(1,4) or in anti-diagional(2,3)
    float op_inverse = op1 > op2 ? 1.0f: 0.0f;

    float cx = conic.x;
    float cy = conic.y;
    float cz = conic.z;
    float det2 = 1/sqrtf(cx*cy*cy - 2*cx*cy*cy + cz*cx*cx);
    float2 point3 = {cy*det2, -cx*det2};
    float2 point1 = {-cy*det2, cx*det2};
    det2 = 1/sqrtf(cx*cz*cz - 2*cz*cy*cy + cz*cy*cy);
    float2 point2 = {cz*det2, -cy*det2};
    float2 point4 = {-cz*det2, cy*det2};

    float part1 = (normal.x*1.4142135f+x_term);
    float part2 = (normal.y*1.4142135f+y_term);
    float point1_large= (part1*point1.x+part2*point1.y) > 0.f?1.0f-op_inverse:0.f+op_inverse;
    float point2_large= (part1*point2.x+part2*point2.y) > 0.f?1.0f-op_inverse:0.f+op_inverse;
    float point3_large= 1.f - point1_large;//(part1*point3.x+part2*point3.y) > 0.f?1.0f-op_inverse:0.f+op_inverse;
    float point4_large= 1.f - point2_large;//(part1*point4.x+part2*point4.y) > 0.f?1.0f-op_inverse:0.f+op_inverse;

    type = full*(point1_large*point4_large*(1-point2_large)*(1-point3_large)*1.f + point1_large*point2_large*(1-point4_large)*(1-point3_large)*2.f + point3_large*point4_large*(1-point2_large)*(1-point1_large)*3.f+point2_large*point3_large*(1-point1_large)*(1-point4_large)*4.f);
//     type = full*(diagional*1.0f*leftright + (1.0f-diagional)*2.0f*(1.0f-leftright) + (1.0f-diagional)*3.0f*leftright + diagional*4.0f*(1.0f-leftright));
}


//use large cov3D and normal to generate small cov
// __device__ void calculate_small(const float* cov_input, const float* viewmatrix, float3 n, float* cov_new_small) {
//     // 将输入的 float 类型直接用于 float 运算
//     glm::vec3 normal(n.x, n.y, n.z);
//     normal = glm::normalize(normal);
//
//     // 使用 float 类型的 3x3 矩阵
//     glm::mat3 W = glm::mat3(
//         viewmatrix[0], viewmatrix[4], viewmatrix[8],
//         viewmatrix[1], viewmatrix[5], viewmatrix[9],
//         viewmatrix[2], viewmatrix[6], viewmatrix[10]);
//
//     // 声明一个 3x3 的 float 类型的协方差矩阵
//     glm::mat3 cov;
//     cov[0][0] = cov_input[0]; // cov(0,0)
//     cov[0][1] = cov[1][0] = cov_input[1]; // cov(0,1) and cov(1,0)
//     cov[0][2] = cov[2][0] = cov_input[2]; // cov(0,2) and cov(2,0)
//     cov[1][1] = cov_input[3]; // cov(1,1)
//     cov[1][2] = cov[2][1] = cov_input[4]; // cov(1,2) and cov(2,1)
//     cov[2][2] = cov_input[5]; // cov(2,2)
//
//     // 执行矩阵变换
//     cov = glm::transpose(W) * cov * W;
//
//     // 使用 float 类型的旋转矩阵
//     glm::vec3 v1, v2;
//
//     if (normal.x == 0.0f && normal.y == 0.0f) {
//         // 如果 n.x 和 n.y 为 0，设置 v1 和 v2
//         v1 = glm::vec3(1.0f, 0.0f, 0.0f);
//         v2 = glm::vec3(0.0f, 1.0f, 0.0f);
//     } else {
//         // 正交化得到 v1 和 v2
//         v1 = glm::normalize(glm::vec3(normal.y, -normal.x, 0.0f));
//         v2 = glm::normalize(glm::cross(normal, v1));
//     }
//
//     glm::mat3 R_transform(v1, v2, normal);
//     glm::mat3 R_transform_T = glm::transpose(R_transform);
//
//     // 计算 cov_transformed 和 cov_inv2，使用 float 类型
//     glm::mat3 cov_transformed = R_transform_T * cov * R_transform;
//     float a = cov_transformed[0][0];  // cov_transformed(0,0)
//     float b = cov_transformed[0][1];  // cov_transformed(0,1) == cov_transformed(1,0)
//     float c = cov_transformed[1][1];  // cov_transformed(1,1)
//     float d = cov_transformed[0][2];  // cov_transformed(0,2) == cov_transformed(2,0)
//     float e = cov_transformed[1][2];  // cov_transformed(1,2) == cov_transformed(2,1)
//     float f = cov_transformed[2][2];  // cov_transformed(2,2)
//
//     // Schur补公式: 计算 A_{11} - B * C^{-1} * B^T
//     float inv_f = 1.0f / f;  // C的逆: 1/f
//     float B_C_inv_B_T_00 = (d * d) * inv_f;
//     float B_C_inv_B_T_01 = (d * e) * inv_f;
//     float B_C_inv_B_T_11 = (e * e) * inv_f;
//
//     // A_{11} - B * C^{-1} * B^T
//     float A11_00 = a - B_C_inv_B_T_00;
//     float A11_01 = b - B_C_inv_B_T_01;
//     float A11_11 = c - B_C_inv_B_T_11;
//
//     // 计算2x2矩阵的行列式
//     float det_A11 = A11_00 * A11_11 - A11_01 * A11_01;
//
//     // 计算逆矩阵的四个元素
//     float inv_A11_00 = A11_11 / det_A11;
//     float inv_A11_01 = -A11_01 / det_A11;
//     float inv_A11_11 = A11_00 / det_A11;
//
//     glm::mat2 twoD_mat(inv_A11_00, inv_A11_01, inv_A11_01, inv_A11_11);
//
//     glm::vec2 eig_val;
//     glm::mat2 eig_vec;
//
//     // 计算 2x2 矩阵的特征值
//     float trace = twoD_mat[0][0] + twoD_mat[1][1];
//     float det = twoD_mat[0][0] * twoD_mat[1][1] - twoD_mat[0][1] * twoD_mat[1][0];
//     float disc = sqrtf(trace * trace - 4.0f * det);
//
//     eig_val[0] = (trace + disc) / 2.0f;
//     eig_val[1] = (trace - disc) / 2.0f;
//
//     // 计算特征向量
//     eig_vec[0][0] = (twoD_mat[0][0] - eig_val[0]) != 0.0f ? -(twoD_mat[0][1] / (twoD_mat[0][0] - eig_val[0])) : 1.0f;
//     eig_vec[1][0] = (twoD_mat[0][0] - eig_val[0]) != 0.0f ? 1.0f : 0.0f;
//
//     eig_vec[0][1] = (twoD_mat[0][0] - eig_val[1]) != 0.0f ? -(twoD_mat[0][1] / (twoD_mat[0][0] - eig_val[1])) : 1.0f;
//     eig_vec[1][1] = (twoD_mat[0][0] - eig_val[1]) != 0.0f ? 1.0f : 0.0f;
//
//     // 归一化特征向量
//     eig_vec[1][0] = 1.0f / sqrtf(eig_vec[0][0] * eig_vec[0][0] + 1.0f);
//     eig_vec[0][0] *= eig_vec[1][0];
//     eig_vec[1][1] = 1.0f / sqrtf(eig_vec[0][1] * eig_vec[0][1] + 1.0f);
//     eig_vec[0][1] *= eig_vec[1][1];
//
//     // 将特征向量转换为 3D
//     glm::vec3 v1_2d(eig_vec[0][0], eig_vec[1][0], 0.0f);
//     glm::vec3 v2_2d(eig_vec[0][1], eig_vec[1][1], 0.0f);
//
//     // 生成 3D 特征向量矩阵
//     glm::mat3 v_3d(v1_2d, v2_2d, glm::vec3(0.0f, 0.0f, 1.0f));
//
//     // 构建特征值矩阵
//     glm::mat3 lam_mat(0.0f);
//     lam_mat[0][0] = 1.0f / eig_val[0];
//     lam_mat[1][1] = 1.0f / eig_val[1];
//     lam_mat[2][2] = 1.0f / (1000.0f * fmaxf(eig_val[0], eig_val[1]));
//
//     // 计算新的协方差矩阵 cov_new_3d
//     glm::mat3 cov_new_3d = v_3d * lam_mat * glm::transpose(v_3d);
//
//     // 将其转换回摄像机坐标系
//     cov_new_3d = R_transform * cov_new_3d * R_transform_T;
//
//     // 最终结果存储为 float32
//     cov_new_small[0] = cov_new_3d[0][0];
//     cov_new_small[1] = cov_new_3d[0][1];
//     cov_new_small[2] = cov_new_3d[0][2];
//     cov_new_small[3] = cov_new_3d[1][1];
//     cov_new_small[4] = cov_new_3d[1][2];
//     cov_new_small[5] = cov_new_3d[2][2];
// }
// __device__ void calculate_small(float6 cov_input, const float* viewmatrix, float3 normal, float* cov_new_small) {
//     // 如果法向量 n 的 x 和 y 分量都为 0，生成标准基向量
//     glm::vec3 v1, v2;
//
//     glm::vec3 n(normal.x, normal.y, normal.z);
//     n = glm::normalize(n);
//
//     glm::mat3 W = glm::mat3(
//         viewmatrix[0], viewmatrix[4], viewmatrix[8],
//         viewmatrix[1], viewmatrix[5], viewmatrix[9],
//         viewmatrix[2], viewmatrix[6], viewmatrix[10]);
//
//     // 声明一个 3x3 的 float 类型的协方差矩阵
//     glm::mat3 cov;
//     cov[0][0] = cov_input.x; // cov(0,0)
//     cov[0][1] = cov[1][0] = cov_input.y; // cov(0,1) and cov(1,0)
//     cov[0][2] = cov[2][0] = cov_input.z; // cov(0,2) and cov(2,0)
//     cov[1][1] = cov_input.w; // cov(1,1)
//     cov[1][2] = cov[2][1] = cov_input.u; // cov(1,2) and cov(2,1)
//     cov[2][2] = cov_input.v; // cov(2,2)
//
//     cov = glm::transpose(W) * cov * W;
//
//     // 如果 n 的 x 和 y 分量都为 0，则沿 z 轴对齐
//     if (n.x == 0.0f && n.y == 0.0f) {
//         v1 = glm::vec3(1.0f, 0.0f, 0.0f);
//         v2 = glm::vec3(0.0f, 1.0f, 0.0f);
//         //n = glm::normalize(glm::vec3(0.0f, 0.0f, n.z));  // 保证法向量沿 z 轴方向
//     } else {
//         // 归一化法向量 n
//         // 生成与 n 正交的 v1 和 v2 向量
//         v1 = glm::normalize(glm::vec3(n.y, -n.x, 0.0f));  // 创建垂直于 n 的 v1
//         v2 = glm::normalize(glm::cross(n, v1));            // 通过叉乘计算 v2
//     }
// //     if (fabsf(glm::dot(v1, v2))>0.00001f ||fabsf(glm::dot(n, v2))>0.00001f||fabsf(glm::dot(n, v1))>0.00001f) printf("n not correct is：[%f,%f,%f]\n",n.x, n.y, n.z);
//
//     // 构建基变换矩阵
//     glm::mat3 basis(v1, v2, n);
//     glm::mat3 R_transform = glm::transpose(basis);  // R_transform 是基变换矩阵的转置
//
//     // 将输入协方差矩阵从 float* 转换为 glm::mat3
// //     glm::mat3 cov_mat(
// //         cov[0], cov[1], cov[2],
// //         cov[3], cov[4], cov[5],
// //         cov[6], cov[7], cov[8]);
//
//     // 对协方差矩阵进行基变换
//     glm::mat3 cov_transformed = R_transform * cov * basis;
//
//     // 提取二维平面上的协方差矩阵的元素
//     float a = cov_transformed[0][0];
//     float b = cov_transformed[0][1];
//     float c = cov_transformed[1][1];
//
//     // 计算二维协方差矩阵的行列式
//     float det = a * c - b * b;
//     float inv_det = 1.0f / det;
//
//     // 计算特征值
//     float trace = a + c;
//     float lambda_use = (trace + sqrtf(trace * trace - 4.0f * det)) / 2.0f;
//
//     // 构建新的 3x3 协方差矩阵，使用 Schur 补
//     glm::mat3 cov_new_3d(
//         c * inv_det, -b * inv_det, 0.0f,
//         -b * inv_det, a * inv_det, 0.0f,
//         0.0f, 0.0f, 1.0f / (100.0f * lambda_use));
//
//     // 将结果转换回原始坐标系
//     cov_new_3d = basis * cov_new_3d * R_transform;
//
//     // 将新的协方差矩阵的结果存储在 cov_new_small 中
//     cov_new_small[0] = cov_new_3d[0][0];
//     cov_new_small[1] = cov_new_3d[0][1];
//     cov_new_small[2] = cov_new_3d[0][2];
//     cov_new_small[3] = cov_new_3d[1][1];
//     cov_new_small[4] = cov_new_3d[1][2];
//     cov_new_small[5] = cov_new_3d[2][2];
// }

// __device__ void calculate_small(float6 cov_input, const float* viewmatrix, float3 normal, float* cov_new_small, const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy) {
//     // 如果法向量 n 的 x 和 y 分量都为 0，生成标准基向量
//     glm::vec3 v1, v2;
//
//     glm::vec3 n(normal.x, normal.y, normal.z);
//     n = glm::normalize(n);
//
// //     glm::mat3 W = glm::mat3(
// //         viewmatrix[0], viewmatrix[4], viewmatrix[8],
// //         viewmatrix[1], viewmatrix[5], viewmatrix[9],
// //         viewmatrix[2], viewmatrix[6], viewmatrix[10]);
//
//     // 声明一个 3x3 的 float 类型的协方差矩阵
//     glm::mat3 cov;
//     cov[0][0] = cov_input.x; // cov(0,0)
//     cov[0][1] = cov[1][0] = cov_input.y; // cov(0,1) and cov(1,0)
//     cov[0][2] = cov[2][0] = cov_input.w; // cov(0,2) and cov(2,0)
//     cov[1][1] = cov_input.z; // cov(1,1)
//     cov[1][2] = cov[2][1] = cov_input.u; // cov(1,2) and cov(2,1)
//     cov[2][2] = cov_input.v; // cov(2,2)
//
// //     cov = glm::transpose(W) * cov * W;
//
//     // 如果 n 的 x 和 y 分量都为 0，则沿 z 轴对齐
//     if (n.x == 0.0f && n.y == 0.0f) {
//         v1 = glm::vec3(1.0f, 0.0f, 0.0f);
//         v2 = glm::vec3(0.0f, 1.0f, 0.0f);
//         //n = glm::normalize(glm::vec3(0.0f, 0.0f, n.z));  // 保证法向量沿 z 轴方向
//     } else {
//         // 归一化法向量 n
//         // 生成与 n 正交的 v1 和 v2 向量
//         v1 = glm::normalize(glm::vec3(n.y, -n.x, 0.0f));  // 创建垂直于 n 的 v1
//         v2 = glm::normalize(glm::cross(n, v1));            // 通过叉乘计算 v2
//     }
// //     if (fabsf(glm::dot(v1, v2))>0.00001f ||fabsf(glm::dot(n, v2))>0.00001f||fabsf(glm::dot(n, v1))>0.00001f) printf("n not correct is：[%f,%f,%f]\n",n.x, n.y, n.z);
//
//     // 构建基变换矩阵
//     glm::mat3 basis(v1, v2, n);
//     glm::mat3 R_transform = glm::transpose(basis);  // R_transform 是基变换矩阵的转置
//
//     // 将输入协方差矩阵从 float* 转换为 glm::mat3
// //     glm::mat3 cov_mat(
// //         cov[0], cov[1], cov[2],
// //         cov[3], cov[4], cov[5],
// //         cov[6], cov[7], cov[8]);
//
//     // 对协方差矩阵进行基变换
//     glm::mat3 cov_transformed = R_transform * cov * basis;
//
//     // 提取二维平面上的协方差矩阵的元素
//     float a = cov_transformed[0][0];
//     float b = cov_transformed[0][1];
//     float c = cov_transformed[1][1];
//     float d = cov_transformed[0][2];
//     float e = cov_transformed[1][2];
//     float f = cov_transformed[2][2];
//
//     float div = 1/f;
//     // 构建新的 3x3 协方差矩阵，使用 Schur 补
//     glm::mat3 cov_new_3d(
//         a-d*(d*div), b-d*(e*div), 0.0f,
//         b-d*(e*div), c-e*(e*div), 0.0f,
//         0.0f, 0.0f, min(a-d*(d*div), c-e*(e*div)) / (100.0f));
//
//     // 将结果转换回原始坐标系
//     cov_new_3d = basis * cov_new_3d * R_transform;
//
//
//     float3 t = transformPoint4x3(mean, viewmatrix);
//
// 	const float limx = 1.3f * tan_fovx;
// 	const float limy = 1.3f * tan_fovy;
// 	const float txtz = t.x / t.z;
// 	const float tytz = t.y / t.z;
// 	t.x = min(limx, max(-limx, txtz)) * t.z;
// 	t.y = min(limy, max(-limy, tytz)) * t.z;
//
// 	glm::mat3 J = glm::mat3(
// 		(focal_x / t.z), 0.0, (-(focal_x * t.x) / (t.z * t.z)),
// 		0.0, (focal_y / t.z), (-(focal_y * t.y) / (t.z * t.z)),
// 		0, 0, (focal_y / t.z));
//
// 	cov_new_3d = glm::transpose(J)*cov_new_3d*J;
//     // 将新的协方差矩阵的结果存储在 cov_new_small 中
//     cov_new_small[0] = cov_new_3d[0][0];
//     cov_new_small[1] = cov_new_3d[0][1];
//     cov_new_small[2] = cov_new_3d[0][2];
//     cov_new_small[3] = cov_new_3d[1][1];
//     cov_new_small[4] = cov_new_3d[1][2];
//     cov_new_small[5] = cov_new_3d[2][2];
// }

__device__ void calculate_small(float6 cov_input, const float* viewmatrix, float3 normal, float* cov_new_small) {
    // 如果法向量 n 的 x 和 y 分量都为 0，生成标准基向量
    glm::vec3 v1, v2;

    glm::vec3 n(normal.x, normal.y, normal.z);
    n = glm::normalize(n);

//     glm::mat3 W = glm::mat3(
//         viewmatrix[0], viewmatrix[4], viewmatrix[8],
//         viewmatrix[1], viewmatrix[5], viewmatrix[9],
//         viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    // 声明一个 3x3 的 float 类型的协方差矩阵
    glm::mat3 cov;
    cov[0][0] = cov_input.x; // cov(0,0)
    cov[0][1] = cov[1][0] = cov_input.y; // cov(0,1) and cov(1,0)
    cov[0][2] = cov[2][0] = cov_input.w; // cov(0,2) and cov(2,0)
    cov[1][1] = cov_input.z; // cov(1,1)
    cov[1][2] = cov[2][1] = cov_input.u; // cov(1,2) and cov(2,1)
    cov[2][2] = cov_input.v; // cov(2,2)

//     cov = glm::transpose(W) * cov * W;

    // 如果 n 的 x 和 y 分量都为 0，则沿 z 轴对齐
    if (n.x == 0.0f && n.y == 0.0f) {
        v1 = glm::vec3(1.0f, 0.0f, 0.0f);
        v2 = glm::vec3(0.0f, 1.0f, 0.0f);
        //n = glm::normalize(glm::vec3(0.0f, 0.0f, n.z));  // 保证法向量沿 z 轴方向
    } else {
        // 归一化法向量 n
        // 生成与 n 正交的 v1 和 v2 向量
        v1 = glm::normalize(glm::vec3(n.y, -n.x, 0.0f));  // 创建垂直于 n 的 v1
        v2 = glm::normalize(glm::cross(n, v1));            // 通过叉乘计算 v2
    }
//     if (fabsf(glm::dot(v1, v2))>0.00001f ||fabsf(glm::dot(n, v2))>0.00001f||fabsf(glm::dot(n, v1))>0.00001f) printf("n not correct is：[%f,%f,%f]\n",n.x, n.y, n.z);

    // 构建基变换矩阵
    glm::mat3 basis(v1, v2, n);
    glm::mat3 R_transform = glm::transpose(basis);  // R_transform 是基变换矩阵的转置

    // 将输入协方差矩阵从 float* 转换为 glm::mat3
//     glm::mat3 cov_mat(
//         cov[0], cov[1], cov[2],
//         cov[3], cov[4], cov[5],
//         cov[6], cov[7], cov[8]);

    // 对协方差矩阵进行基变换
    glm::mat3 cov_transformed = R_transform * cov * basis;

    // 提取二维平面上的协方差矩阵的元素
    float a = cov_transformed[0][0];
    float b = cov_transformed[0][1];
    float c = cov_transformed[1][1];
    float d = cov_transformed[0][2];
    float e = cov_transformed[1][2];
    float f = cov_transformed[2][2];

    float div = 1/f;

    //float lambda = 0.5f*((a-d*(d*div) + c-e*(e*div)) - sqrtf((a-d*(d*div) - c-e*(e*div))*(a-d*(d*div) + c-e*(e*div)) + 4.f*(b-d*(e*div))*(b-d*(e*div))));
    // 构建新的 3x3 协方差矩阵，使用 Schur 补
    glm::mat3 cov_new_3d(
        a-d*(d*div), b-d*(e*div), 0.0f,
        b-d*(e*div), c-e*(e*div), 0.0f,
        0.0f, 0.0f, max(a-d*(d*div), c-e*(e*div)) / (100.0f));
    //0.0f, 0.0f, lambda / (100.0f));
    // 将结果转换回原始坐标系
    cov_new_3d = basis * cov_new_3d * R_transform;

    // 将新的协方差矩阵的结果存储在 cov_new_small 中
    cov_new_small[0] = cov_new_3d[0][0];
    cov_new_small[1] = cov_new_3d[0][1];
    cov_new_small[2] = cov_new_3d[0][2];
    cov_new_small[3] = cov_new_3d[1][1];
    cov_new_small[4] = cov_new_3d[1][2];
    cov_new_small[5] = cov_new_3d[2][2];
}


// __device__ void calculate_small(float6 cov_input, const float* viewmatrix, float3 normal, float* cov_new_small) {
//     // 使用双精度类型进行中间计算
//     glm::dvec3 v1, v2;
//     glm::dvec3 n((double)normal.x, (double)normal.y, (double)normal.z);
//     n = glm::normalize(n);
//
//     glm::mat3x3 W = glm::mat3x3(
//         (double)viewmatrix[0], (double)viewmatrix[4], (double)viewmatrix[8],
//         (double)viewmatrix[1], (double)viewmatrix[5], (double)viewmatrix[9],
//         (double)viewmatrix[2], (double)viewmatrix[6], (double)viewmatrix[10]);
//
//     // 声明一个 3x3 的双精度协方差矩阵
//     glm::mat3x3 cov;
//     cov[0][0] = (double)cov_input.x; // cov(0,0)
//     cov[0][1] = cov[1][0] = (double)cov_input.y; // cov(0,1) and cov(1,0)
//     cov[0][2] = cov[2][0] = (double)cov_input.z; // cov(0,2) and cov(2,0)
//     cov[1][1] = (double)cov_input.w; // cov(1,1)
//     cov[1][2] = cov[2][1] = (double)cov_input.u; // cov(1,2) and cov(2,1)
//     cov[2][2] = (double)cov_input.v; // cov(2,2)
//
//     // 矩阵乘法，W 的转置乘 cov，再乘 W
//     cov = glm::transpose(W) * cov * W;
//
//     // 如果法向量的 x 和 y 分量为 0，设置标准基向量
//     if (n.x == 0.0 && n.y == 0.0) {
//         v1 = glm::dvec3(1.0, 0.0, 0.0);
//         v2 = glm::dvec3(0.0, 1.0, 0.0);
//     } else {
//         // 生成与 n 正交的向量 v1 和 v2
//         v1 = glm::normalize(glm::dvec3(n.y, -n.x, 0.0));
//         v2 = glm::normalize(glm::cross(n, v1));
//     }
//
//     // 基变换矩阵
//     glm::mat3x3 basis(v1, v2, n);
//     glm::mat3x3 R_transform = glm::transpose(basis);
//
//     // 变换协方差矩阵
//     glm::mat3x3 cov_transformed = R_transform * cov * basis;
//
//     // 提取二维平面上的协方差矩阵的元素
//     double a = cov_transformed[0][0];
//     double b = cov_transformed[0][1];
//     double c = cov_transformed[1][1];
//
//     // 计算二维协方差矩阵的行列式
//     double det = a * c - b * b;
//     double inv_det = 1.0 / det;
//
//     // 计算特征值
//     double trace = a + c;
//     double lambda_use = (trace + sqrt(trace * trace - 4.0 * det)) / 2.0;
//
//     // 构建新的 3x3 协方差矩阵，使用 Schur 补
//     glm::mat3x3 cov_new_3d(
//         c * inv_det, -b * inv_det, 0.0,
//         -b * inv_det, a * inv_det, 0.0,
//         0.0, 0.0, 1.0 / (1000.0 * lambda_use));
//
//     // 将结果转换回原始坐标系
//     cov_new_3d = basis * cov_new_3d * R_transform;
//
//     // 将新的协方差矩阵的结果存储在 cov_new_small 中
//     cov_new_small[0] = (float)cov_new_3d[0][0];
//     cov_new_small[1] = (float)cov_new_3d[0][1];
//     cov_new_small[2] = (float)cov_new_3d[0][2];
//     cov_new_small[3] = (float)cov_new_3d[1][1];
//     cov_new_small[4] = (float)cov_new_3d[1][2];
//     cov_new_small[5] = (float)cov_new_3d[2][2];
// }


// __device__ void calculate_small(float6 cov_input, const float* viewmatrix, float3 n, float* cov_new_small) {
//     // 使用 float 类型计算
//     glm::vec3 normal(n.x, n.y, n.z);
//     normal = glm::normalize(normal);
//
//     // 使用 float 类型的 view matrix
//     glm::mat3 W = glm::mat3(
//         viewmatrix[0], viewmatrix[4], viewmatrix[8],
//         viewmatrix[1], viewmatrix[5], viewmatrix[9],
//         viewmatrix[2], viewmatrix[6], viewmatrix[10]);
//
//     // 声明一个 3x3 的 float 类型的协方差矩阵
//     glm::mat3 cov;
//     cov[0][0] = cov_input.x; // cov(0,0)
//     cov[0][1] = cov[1][0] = cov_input.y; // cov(0,1) and cov(1,0)
//     cov[0][2] = cov[2][0] = cov_input.z; // cov(0,2) and cov(2,0)
//     cov[1][1] = cov_input.w; // cov(1,1)
//     cov[1][2] = cov[2][1] = cov_input.u; // cov(1,2) and cov(2,1)
//     cov[2][2] = cov_input.v; // cov(2,2)
//
//     cov = glm::transpose(W) * cov * W;
//
//     // 使用 float 类型的旋转矩阵
//     glm::vec3 v1, v2;
//
//     if (normal.x == 0.0f && normal.y == 0.0f) {
//         // 如果 n.x 和 n.y 为 0，设置 v1 和 v2
//         v1 = glm::vec3(1.0f, 0.0f, 0.0f);
//         v2 = glm::vec3(0.0f, 1.0f, 0.0f);
//     } else {
//         // 正交化得到 v1 和 v2
//         v1 = glm::normalize(glm::vec3(normal.y, -normal.x, 0.0f));
//         v2 = glm::normalize(glm::cross(normal, v1));
//     }
//
//     glm::mat3 R_transform(v1, v2, normal);
//     printf("R:%f,%f,%f, v10:%f, v20:%f,v30%f\n",R_transform[0][0],R_transform[0][1],R_transform[0][2],v1[0],v2[0],normal[0]);
//     glm::mat3 R_transform_T = glm::transpose(R_transform);
//
//     // 计算 cov_transformed 和 cov_inv2，使用 float 类型
//     glm::mat3 cov_transformed = R_transform_T * cov * R_transform;
//     glm::mat2 twoD_mat(cov_transformed[0][0], cov_transformed[0][1], cov_transformed[1][0], cov_transformed[1][1]);
//
//     glm::vec2 eig_val;
//     glm::mat2 eig_vec;
//
//     // 计算 2x2 矩阵的特征值
//     float trace = twoD_mat[0][0] + twoD_mat[1][1];
//     float det = twoD_mat[0][0] * twoD_mat[1][1] - twoD_mat[0][1] * twoD_mat[1][0];
//     float disc = sqrtf(trace * trace - 4.0f * det);
//
//     eig_val[0] = (trace + disc) / 2.0f;
//     eig_val[1] = (trace - disc) / 2.0f;
//
//     // 计算特征向量
//     eig_vec[0][0] = (twoD_mat[0][0] - eig_val[0]) != 0.0f ? -(twoD_mat[0][1] / (twoD_mat[0][0] - eig_val[0])) : 1.0f;
//     eig_vec[1][0] = (twoD_mat[0][0] - eig_val[0]) != 0.0f ? 1.0f : 0.0f;
//
//     eig_vec[0][1] = (twoD_mat[0][0] - eig_val[1]) != 0.0f ? -(twoD_mat[0][1] / (twoD_mat[0][0] - eig_val[1])) : 1.0f;
//     eig_vec[1][1] = (twoD_mat[0][0] - eig_val[1]) != 0.0f ? 1.0f : 0.0f;
//
//     // 归一化特征向量
//     eig_vec[1][0] = 1.0f / sqrtf(eig_vec[0][0] * eig_vec[0][0] + 1.0f);
//     eig_vec[0][0] *= eig_vec[1][0];
//     eig_vec[1][1] = 1.0f / sqrtf(eig_vec[0][1] * eig_vec[0][1] + 1.0f);
//     eig_vec[0][1] *= eig_vec[1][1];
//
//     // 将特征向量转换为 3D
//     glm::vec3 v1_2d(eig_vec[0][0], eig_vec[1][0], 0.0f);
//     glm::vec3 v2_2d(eig_vec[0][1], eig_vec[1][1], 0.0f);
//
//     // 生成 3D 特征向量矩阵
//     glm::mat3 v_3d(v1_2d, v2_2d, glm::vec3(0.0f, 0.0f, 1.0f));
//
//     // 构建特征值矩阵
//     glm::mat3 lam_mat(0.0f);
//     lam_mat[0][0] = 1.0f / eig_val[0];
//     lam_mat[1][1] = 1.0f / eig_val[1];
//     lam_mat[2][2] = 1.0f / (1000.0f * fmaxf(eig_val[0], eig_val[1]));
//
//     // 计算新的协方差矩阵 cov_new_3d
//     glm::mat3 cov_new_3d = v_3d * lam_mat * glm::transpose(v_3d);
//
//     // 将其转换回摄像机坐标系
//     cov_new_3d = R_transform * cov_new_3d * R_transform_T;
//
//     // 最终结果存储为 float32
//     cov_new_small[0] = cov_new_3d[0][0];
//     cov_new_small[1] = cov_new_3d[0][1];
//     cov_new_small[2] = cov_new_3d[0][2];
//     cov_new_small[3] = cov_new_3d[1][1];
//     cov_new_small[4] = cov_new_3d[1][2];
//     cov_new_small[5] = cov_new_3d[2][2];
// }


//high precision
// __device__ void calculate_small(float6 cov_input, const float* viewmatrix, float3 n, float* cov_new_small) {
//     // 将输入的 float 类型转换为 double 类型进行运算
//     glm::dvec3 normal((double)n.x, (double)n.y, (double)n.z);
//
//     normal = glm::normalize(normal);
//
//     glm::dmat3 W = glm::dmat3(
// 		(double)(viewmatrix[0]), (double)(viewmatrix[4]), (double)(viewmatrix[8]),
// 		(double)(viewmatrix[1]), (double)(viewmatrix[5]), (double)(viewmatrix[9]),
// 		(double)(viewmatrix[2]), (double)(viewmatrix[6]), (double)(viewmatrix[10]));
//
// //     normal[0] = normal[0]/(sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]));
// //     normal[1] = normal[1]/(sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]));
// //     normal[2] = normal[2]/(sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]));
//
//     // 声明一个 3x3 的 double 类型的协方差矩阵
//     glm::dmat3 cov;
//     cov[0][0] = (double)cov_input.x; // cov(0,0)
//     cov[0][1] = cov[1][0] = (double)cov_input.y; // cov(0,1) and cov(1,0)
//     cov[0][2] = cov[2][0] = (double)cov_input.z; // cov(0,2) and cov(2,0)
//     cov[1][1] = (double)cov_input.w; // cov(1,1)
//     cov[1][2] = cov[2][1] = (double)cov_input.u; // cov(1,2) and cov(2,1)
//     cov[2][2] = (double)cov_input.v; // cov(2,2)
//
// //     printf("%f,%f,%f,%f,%f,%f\n",cov_input.x,cov_input.y,cov_input.z,cov_input.w,cov_input.u,cov_input.v);
//
//     cov=glm::transpose(W)*cov*W;
// //     cov=W*cov*glm::transpose(W);
//
//     // 使用 double 类型的旋转矩阵
//     glm::dvec3 v1, v2;
//
//     if (normal.x == 0.0 && normal.y == 0.0) {
//         // if n.x and n.y are 0, set v1 and v2
//         v1 = glm::dvec3(1.0, 0.0, 0.0);
//         v2 = glm::dvec3(0.0, 1.0, 0.0);
//     } else {
//         // 正交化得到 v1 和 v2
// //         v1 = glm::dvec3(normal.y/(sqrtf(normal.y*normal.y+normal.x*normal.x)), -normal.x/(sqrtf(normal.y*normal.y+normal.x*normal.x)), 0.0);
//         v1 = glm::normalize(glm::dvec3(normal.y, -normal.x,0.0));
//         v2 = glm::normalize(glm::cross(normal, v1));
//     }
// //     printf("\n\n%f,%f,%f,%f,%f,%f\n\n", glm::dot(v1, v2), glm::dot(normal, v2), glm::dot(v1, normal), glm::dot(v1, v1), glm::dot(v2, v2), glm::dot(normal, normal));
// //     printf("\n\n%f\n\n", v1*v2);
//
//     glm::dmat3 R_transform(v1, v2, normal);
//     glm::dmat3 R_transform_T = glm::transpose(R_transform);
//
//     // 计算 cov_transformed 和 cov_inv2，使用双精度类型
//     glm::dmat3 cov_transformed = R_transform_T * cov * R_transform;
//     glm::dmat2 twoD_mat(cov_transformed[0][0], cov_transformed[0][1], cov_transformed[1][0], cov_transformed[1][1]);
// //     glm::dmat3 cov_inv2 = glm::inverse(cov_transformed);
// //     double a = cov_transformed[0][0];  // cov_transformed(0,0)
// //     double b = cov_transformed[0][1];  // cov_transformed(0,1) == cov_transformed(1,0)
// //     double c = cov_transformed[1][1];  // cov_transformed(1,1)
// //     double d = cov_transformed[0][2];  // cov_transformed(0,2) == cov_transformed(2,0)
// //     double e = cov_transformed[1][2];  // cov_transformed(1,2) == cov_transformed(2,1)
// //     double f = cov_transformed[2][2];  // cov_transformed(2,2)
// //
// //     // Schur补公式: 计算 A_{11} - B * C^{-1} * B^T
// //     double inv_f = 1.0 / f;  // C的逆: 1/f
// //     double B_C_inv_B_T_00 = (d * d) * inv_f;
// //     double B_C_inv_B_T_01 = (d * e) * inv_f;
// //     double B_C_inv_B_T_11 = (e * e) * inv_f;
// //
// //     // A_{11} - B * C^{-1} * B^T
// //     double A11_00 = a - B_C_inv_B_T_00;
// //     double A11_01 = b - B_C_inv_B_T_01;
// //     double A11_11 = c - B_C_inv_B_T_11;
// //
// //     // 计算2x2矩阵的行列式
// //     double det_A11 = A11_00 * A11_11 - A11_01 * A11_01;
// //
// //     // 计算逆矩阵的四个元素
// //     double inv_A11_00 = A11_11 / det_A11;
// //     double inv_A11_01 = -A11_01 / det_A11;
// //     double inv_A11_11 = A11_00 / det_A11;
// //
// //     glm::dmat2 twoD_mat(inv_A11_00, inv_A11_01, inv_A11_01, inv_A11_11);
//     // 将 cov_inv2 转换为 2x2 矩阵，用于计算特征值和特征向量
// //     glm::dmat2 twoD_mat(cov_inv2[0][0], cov_inv2[0][1], cov_inv2[1][0], cov_inv2[1][1]);
//
//     glm::dvec2 eig_val;
//     glm::dmat2 eig_vec;
//
//     // 计算 2x2 矩阵的特征值
//     double trace = twoD_mat[0][0] + twoD_mat[1][1];
//     double det = twoD_mat[0][0] * twoD_mat[1][1] - twoD_mat[0][1] * twoD_mat[1][0];
//     double disc = sqrt(trace * trace - 4 * det);
//
//     eig_val[0] = (trace + disc) / 2.0;
//     eig_val[1] = (trace - disc) / 2.0;
//
//     // 计算特征向量
//     eig_vec[0][0] = (twoD_mat[0][0] - eig_val[0]) != 0 ? -(twoD_mat[0][1] / (twoD_mat[0][0] - eig_val[0])) : 1.0;
//     eig_vec[1][0] = (twoD_mat[0][0] - eig_val[0]) != 0 ? 1.0 : 0.0;
//
//     eig_vec[0][1] = (twoD_mat[0][0] - eig_val[1]) != 0 ? -(twoD_mat[0][1] / (twoD_mat[0][0] - eig_val[1])) : 1.0;
//     eig_vec[1][1] = (twoD_mat[0][0] - eig_val[1]) != 0 ? 1.0 : 0.0;
//
//     // 归一化特征向量
//     eig_vec[1][0] = 1 / sqrt(eig_vec[0][0] * eig_vec[0][0] + 1.0);
//     eig_vec[0][0] *= eig_vec[1][0];
//     eig_vec[1][1] = 1 / sqrt(eig_vec[0][1] * eig_vec[0][1] + 1.0);
//     eig_vec[0][1] *= eig_vec[1][1];
//
//     // 将特征向量转换为 3D
//     glm::dvec3 v1_2d(eig_vec[0][0], eig_vec[1][0], 0.0);
//     glm::dvec3 v2_2d(eig_vec[0][1], eig_vec[1][1], 0.0);
//
//     // 生成 3D 特征向量矩阵
//     glm::dmat3 v_3d(v1_2d, v2_2d, glm::dvec3(0.0, 0.0, 1.0));
//
//     // 构建特征值矩阵
//     glm::dmat3 lam_mat(0.0);
//     lam_mat[0][0] = 1/eig_val[0];
//     lam_mat[1][1] = 1/eig_val[1];
//     lam_mat[2][2] = 1/(1000.0 * fmax(eig_val[0], eig_val[1]));
//
//     // 计算新的协方差矩阵 cov_new_3d
//     glm::dmat3 cov_new_3d = v_3d * lam_mat * glm::transpose(v_3d);//glm::inverse(glm::transpose(v_3d) * lam_mat * v_3d);
//
//     // 将其转换回摄像机坐标系
//     cov_new_3d = R_transform * cov_new_3d * R_transform_T;
//
//     // 最终结果存储为 float32（将 double 转换为 float）
//     cov_new_small[0] = (float)cov_new_3d[0][0];
//     cov_new_small[1] = (float)cov_new_3d[0][1];
//     cov_new_small[2] = (float)cov_new_3d[0][2];
//     cov_new_small[3] = (float)cov_new_3d[1][1];
//     cov_new_small[4] = (float)cov_new_3d[1][2];
//     cov_new_small[5] = (float)cov_new_3d[2][2];
// }

// __device__  float4 generate_J(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy,const float*  viewmatrix){
//     float3 t = transformPoint4x3(mean, viewmatrix);
//
// 	const float limx = 1.3f * tan_fovx;
// 	const float limy = 1.3f * tan_fovy;
// 	const float txtz = t.x / t.z;
// 	const float tytz = t.y / t.z;
// 	t.x = min(limx, max(-limx, txtz)) * t.z;
// 	t.y = min(limy, max(-limy, tytz)) * t.z;
//
// 	glm::mat3 J = glm::mat3(
// 		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
// 		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
// 		0, 0, 0);
// 	return {J[0][0],J[0][2],J[1][1],J[1][2]};
//  }
//
//
// // //////////////////////
// // 定义一个全局变量作为 mutex
// __device__ int mutex = 0;
//
// // 函数尝试获取 mutex
// __device__ void lock(int *mutex) {
//     while (atomicCAS(mutex, 0, 1) != 0);
// }
//
// // 函数释放 mutex
// __device__ void unlock(int *mutex) {
//     atomicExch(mutex, 0);
// }
// //////////////////////

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
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
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
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
	float3* save_normal)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float3 gs_normal = {normal[3*idx],normal[3*idx+1],normal[3*idx+2]};  // is this in world coordinate or camera coordiante

    // map the normal to ray space
    //printf("%f,%f,%f\n",viewmatrix[12],viewmatrix[13],viewmatrix[14]);
	gs_normal = transformPoint4x3(gs_normal, viewmatrix);   //transform normal to camera coordinate
// 	gs_normal = transformVec4x3(gs_normal, viewmatrix);

    float sign_term = (gs_normal.z+0.00000001f)/(fabsf(gs_normal.z)+0.00000001f);

//     gs_normal = transformVec4x3(gs_normal, viewmatrix);
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	//save_normal[idx] = gs_normal;

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
// 	const float* cov3D;
    const float* cov3D;
	//float6 cov3D_inv;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
// 		cov3D_inv = computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
        computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

    float3 cov;
    float6 view_cov3D;
	float6 cov_temp = computeCov2D_halfgs(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, view_cov3D, viewmatrix);
	cov = {cov_temp.x, cov_temp.y, cov_temp.z};

	float det = cov.x * cov.z - cov.y * cov.y; //det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	//float det_inv = 1.f / sqrt(det);
	float det_inv2 = 1.f / det;

	float3 conic = { cov.z * det_inv2, -cov.y * det_inv2, cov.x * det_inv2 };//conic is the inverse of the variance matrix for 2D
    //////calculate rectangle size for 2D ellipse:
    float power = logf(256.f * max(opacities[2 * idx], opacities[2 * idx+1]));//logf(2.f) * 8.0f + logf(2.f) * log2_opacity;
    int width = (int)(1.414214f * __fsqrt_rn(cov.x * power) + 1.0f);
    int height = (int)(1.414214f * __fsqrt_rn(cov.z * power) + 1.0f);

    float3 cov_small;
    const float* cov3D_small;
//     float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
//     if (cov3D_precomp_small != nullptr && cov3D_inv.x != 0&& cov3D_inv.y != 0&& cov3D_inv.z != 0 && 2*width > BLOCK_X && 2*height > BLOCK_Y)//
//     if (cov3D_precomp_small != nullptr && ((int)((point_image.x-width)/BLOCK_X)!=(int)((point_image.x)/BLOCK_X) || (int)((point_image.y-height)/BLOCK_X)!=(int)((point_image.y)/BLOCK_Y)))//2*width > BLOCK_X && 2*height > BLOCK_Y)//
	if (cov3D_precomp_small != nullptr && (2*width > BLOCK_X || 2*height > BLOCK_Y)) //cov3D_precomp_small != nullptr &&  && (fabsf(opacities[2 * idx]-opacities[2 * idx+1])>0.004f)
	{
//         calculate_small(cov3D_inv, viewmatrix, gs_normal, cov3D_smalls + idx * 6);
        calculate_small(cov_temp, viewmatrix, gs_normal, cov3D_smalls + idx * 6);
//         calculate_small(view_cov3D, viewmatrix, gs_normal, cov3D_smalls + idx * 6, p_orig, focal_x, focal_y, tan_fovx, tan_fovy);
        cov3D_small = cov3D_smalls + idx * 6;
//         cov3D_small = cov3D_precomp_small + idx * 6;
        cov_small = {cov3D_small[0],cov3D_small[1],cov3D_small[3]};//computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D_small, viewmatrix);
//         cov_small = cov;
    }
    else{
////         printf("too small!\n");
        cov_small = cov;
    }
	// Compute 2D screen-space covariance matrix
// 	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

//     float mid_small = 0.5f * (cov_small.x + cov_small.z);
//     float det_small = cov_small.x * cov_small.z - cov_small.y * cov_small.y;;
// 	float small_lam1 = mid_small + sqrt(max(0.1f, mid_small * mid_small - det_small));
// 	float small_lam2 = mid_small - sqrt(max(0.1f, mid_small * mid_small - det_small));
	float lam1 = cov_small.x;//cov_small.x >= cov_small.z?small_lam1:small_lam2;
	float lam2 = cov_small.z;//cov_small.x < cov_small.z?small_lam1:small_lam2;

	// Invert covariance (EWA algorithm)
//     int width_small = (int)(1.414214f * __fsqrt_rn(cov_small.x * power) + 1.0f);
//     int height_small = (int)(1.414214f * __fsqrt_rn(cov_small.z * power) + 1.0f);
    int width_small = (int)(1.414214f * __fsqrt_rn(lam1 * power) + 1.0f);
    int height_small = (int)(1.414214f * __fsqrt_rn(lam2 * power) + 1.f);

    width_small = min(width_small, width);
//     printf("%f\n",(float)height_small/(float)height);
    height_small = 2*height_small>height ? min(height_small, height):max(height_small, height);//((float)height_small/(float)height) > 0.35f ?min(height_small, height):max(height_small, height); //min(height_small, height);//
//     min(height_small, height);//
    //max(height_small, height);//
    if (width <= 0 || height <= 0){
        return;
    }

    float power2 = logf(256.f * min(opacities[2 * idx], opacities[2 * idx+1]));//logf(2.f) * 8.0f + logf(2.f) * log2_opacity;
    int width_another = (int)(1.414214f * __fsqrt_rn(cov.x * power2) + 1.0f);
    int height_another = (int)(1.414214f * __fsqrt_rn(cov.z * power2) + 1.0f);

//     int width_small_another = (int)(1.414214f * __fsqrt_rn(cov_small.x * power2) + 1.0f);
//     int height_small_another = (int)(1.414214f * __fsqrt_rn(cov_small.z * power2) + 1.0f);
    int width_small_another = (int)(1.414214f * __fsqrt_rn(lam1 * power2) + 1.0f);
    int height_small_another = (int)(1.414214f * __fsqrt_rn(lam2 * power2) + 1.f);

//     printf("%f,%f,%f,%f\n",cov.x,cov.z,lam1,lam2);

    width_small_another = min(width_small_another, width_another);
    height_small_another = 2*height_small_another>height_another?min(height_small_another, height_another):max(height_small_another, height_another);//((float)height_small_another/(float)height_another) > 0.35f ?min(height_small_another, height_another):max(height_small_another, height_another); //min(height_small_another, height_another);//
//     min(height_small_another, height_another);//
//     max(height_small_another, height_another);//
    //////
	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles.

	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	//float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(lambda1));//max(lambda1, lambda2)));
// 	float my_radius = sqrtf(max(width_small, width)*max(width_small, width)+max(height_small, height)*max(height_small, height));

	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	uint2 rect_min_another, rect_max_another;
	float type;

	float first_divide = 1/(1.4142135f*sqrtf(max(0.00000001f,cov_temp.v - (cov_temp.w*cov_temp.w* conic.x+2*cov_temp.w*cov_temp.u*conic.y+cov_temp.u*cov_temp.u* conic.z))));
	float x_term = cov_temp.w* conic.x + cov_temp.u* conic.y;
	float y_term = cov_temp.u* conic.z + cov_temp.w* conic.y;

	gs_normal.z = 1.f/(1.4142135f*(gs_normal.z+0.000001f)); //just in case z is 0
//     if (gs_normal.z == 0.f){gs_normal.z = 1.f/(1.4142135f*0.000001f);}
//     else{
//     gs_normal.z = 1.f/(1.4142135f*gs_normal.z); //1.4142135f*
//     }
    gs_normal.x = gs_normal.x*gs_normal.z;
    gs_normal.y = gs_normal.y*gs_normal.z;
//     generate_type(cov, opacities[2 * idx], opacities[2 * idx+1], type, gs_normal, x_term, y_term);
    generate_type(cov, cov_small, opacities[2 * idx], opacities[2 * idx+1], type, gs_normal, x_term, y_term, conic);
//     generate_type(cov, opacities[2 * idx], opacities[2 * idx+1], type, gs_normal);

// 	getRect(point_image, my_radius, my_radius, rect_min, rect_max, grid);
    //float amlp = 1.f;
    getRect_another(point_image, width, height, width_small, height_small, width_another, height_another, width_small_another, height_small_another, (int)type, rect_min, rect_max, rect_min_another, rect_max_another, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity1[idx] = { conic.x, conic.y, conic.z, opacities[2 * idx]};
// 	conic_opacity2[idx] = { conic.x, conic.y, conic.z, opacities[2 * idx + 1]};
// 	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	///for new half gaussian function


    gs_normal.x = gs_normal.x*1.4142135f*first_divide;
    gs_normal.y = gs_normal.y*1.4142135f*first_divide;
	gs_normal.z = gs_normal.z*1.4142135f*first_divide;
    save_normal[idx] = gs_normal;
    conic_opacity2[idx] = {x_term*first_divide, y_term*first_divide, first_divide, opacities[2 * idx + 1]};

    uint2 rect_min2, rect_max2;
	rect_min2.x = min(rect_min.x,rect_min_another.x);
	rect_min2.y = min(rect_min.y,rect_min_another.y);
	rect_max2.x = max(rect_max.x,rect_max_another.x);
	rect_max2.y = max(rect_max.y,rect_max_another.y);
    conic_opacity3[idx] = {rect_min2.x, rect_min2.y, rect_max2.x, rect_max2.y};
    conic_opacity4[idx] = {sign_term,0.f,0.f,0.f}; //{rect_min.x, rect_min.y, rect_max.x, rect_max.y};
    conic_opacity5[idx] = {cov_temp.w,cov_temp.u,cov_temp.v};
    //conic_opacity6[idx] = {rect_min_another.x, rect_min_another.y, rect_max_another.x, rect_max_another.y};
    tiles_touched[idx] = (rect_max2.x-rect_min2.x)*(rect_max2.y-rect_min2.y);
    //float overlap= max(0, min(rect_max.y, rect_max_another.y) - max(rect_min.y, rect_min_another.y)) * max(0, min(rect_max.x, rect_max_another.x) - max(rect_min.x, rect_min_another.x));
    //tiles_touched[idx] = (rect_max.x-rect_min.x)*(rect_max.y-rect_min.y) + (rect_max_another.x-rect_min_another.x)*(rect_max_another.y-rect_min_another.y) - overlap;//(rect_max2.x-rect_min2.x)*(rect_max2.y-rect_min2.y);

    ////////////////////////only for test
//     uint2 rect_min_test, rect_max_test;
//     getRect(point_image, width, height, rect_min_test, rect_max_test, grid);
//     float3 temp_normal = transformPoint4x3({normal[3*idx],normal[3*idx+1],normal[3*idx+2]}, viewmatrix);
//     //if(rect_min_test.x!=rect_min2.x||rect_min_test.y!=rect_min2.y||rect_max_test.x!=rect_max2.x||rect_max_test.y!=rect_max2.y)
//
// //     if((int)cov.x==22 && (int)cov.y==-31 && (int)cov.z==43){
// //     if(cov.x > 10000.f){
//     if ((rect_max_test.y-rect_min_test.y) > 2*(rect_max.y-rect_min.y) && rect_min.y != 0 && rect_max.y != 67){
// //     if (height_small/height < 0.98f){
// //     if(rect_min_test.x!=rect_min2.x||rect_min_test.y!=rect_min2.y||rect_max_test.x!=rect_max2.x||rect_max_test.y!=rect_max2.y){
// //     if(width_small == 0.f || height_small == 0.f || width_small_another == 0.f || height_small_another == 0.f){
//     float4 temp_use = generate_J(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, viewmatrix);
//
// //     const float* cov3D_small;
// 	const float* cov3D_small = cov3D_precomp_small + idx * 6;
//     glm::mat3 Vrk = glm::mat3(
// 		cov3D_small[0], cov3D_small[1], cov3D_small[2],
// 		cov3D_small[1], cov3D_small[3], cov3D_small[4],
// 		cov3D_small[2], cov3D_small[4], cov3D_small[5]);
// 	glm::vec3 normal_use(temp_normal.x, temp_normal.y, temp_normal.z);
// 	//glm::vec3 result = normal_use*Vrk;
// //     printf("cov_small_3D=np.array([[%f,%f,%f],[%f,%f,%f],[%f,%f,%f])\ncov_large_3D=np.array([[%f,%f,%f],[%f,%f,%f],[%f,%f,%f]])\nview_matrix=np.array([[%f,%f,%f],[%f,%f,%f],[%f,%f,%f])\nJ=np.array([[%f,0,%f],[0,%f,%f]])\ncov_small=np.array([[%f,%f],[%f,%f]])\ncov=np.array([[%f,%f],[%f,%f]])\nn1=%f\nn2=%f\nn3=%f\nop1=%f\nop2=%f\nrec_min=[%d, %d]\nrec_max=[%d,%d]\nrec_min_new=[%d, %d]\nrec_max_new=[%d,%d]\n#type=%f\n\n",cov3D_small[0], cov3D_small[1], cov3D_small[2],cov3D_small[1], cov3D_small[3], cov3D_small[4], cov3D_small[2], cov3D_small[4], cov3D_small[5],cov3D[0], cov3D[1], cov3D[2],cov3D[1], cov3D[3], cov3D[4], cov3D[2], cov3D[4], cov3D[5],viewmatrix[0], viewmatrix[4], viewmatrix[8],viewmatrix[1], viewmatrix[5], viewmatrix[9],viewmatrix[2], viewmatrix[6], viewmatrix[10], temp_use.x, temp_use.y, temp_use.z, temp_use.w,cov_small.x , cov_small.y, cov_small.y, cov_small.z,cov.x , cov.y, cov.y, cov.z,temp_normal.x, temp_normal.y, temp_normal.z, opacities[2 * idx], opacities[2 * idx+1],rect_min_test.x,rect_min_test.y,rect_max_test.x,rect_max_test.y,rect_min2.x,rect_min2.y,rect_max2.x,rect_max2.y,type);
//     lock(&mutex);
// //     float amp = 100.f;
//     printf("cov_small=np.array([[%.11f,%.11f],[%.11f,%.11f]])\ncov=np.array([[%.11f,%.11f],[%.11f,%.11f]])\nn1=%.11f\nn2=%.11f\nn3=%.11f\nop1=%f\nop2=%f\nrec_min=[%d, %d]\nrec_max=[%d,%d]\nrec_min_new=[%d, %d]\nrec_max_new=[%d,%d]\n#type=%f\n\n",cov_small.x , cov_small.y, cov_small.y, cov_small.z,cov.x , cov.y, cov.y, cov.z,temp_normal.x, temp_normal.y, temp_normal.z, opacities[2 * idx], opacities[2 * idx+1],rect_min_test.x,rect_min_test.y,rect_max_test.x,rect_max_test.y,rect_min2.x,rect_min2.y,rect_max2.x,rect_max2.y,type);
// //     printf("cov_small_3D=np.array([[%.11f,%.11f,%.11f],[%.11f,%.11f,%.11f],[%.11f,%.11f,%.11f]])\ncov_large_3D=np.array([[%.11f,%.11f,%.11f],[%.11f,%.11f,%.11f],[%.11f,%.11f,%.11f]])\nview_matrix=np.array([[%.12f,%.12f,%.12f],[%.12f,%.12f,%.12f],[%.12f,%.12f,%.12f]])\nJ=np.array([[%f,0,%f],[0,%f,%f],[0,0,0]])\n\n",cov3D_small[0], cov3D_small[1], cov3D_small[2],cov3D_small[1], cov3D_small[3], cov3D_small[4], cov3D_small[2], cov3D_small[4], cov3D_small[5],cov3D[0], cov3D[1], cov3D[2],cov3D[1], cov3D[3], cov3D[4], cov3D[2], cov3D[4], cov3D[5],viewmatrix[0], viewmatrix[4], viewmatrix[8],viewmatrix[1], viewmatrix[5], viewmatrix[9],viewmatrix[2], viewmatrix[6], viewmatrix[10],temp_use.x, temp_use.y, temp_use.z, temp_use.w);
//     printf("grid.x = %d\ngrid.y=%d\n", grid.x, grid.y);
//     printf("new_large=np.array([[%f,%f,%f],[%f,%f,%f],[%f,%f,%f]])\n#idx=%d\nn1_ori=%.12f\nn2_ori=%.12f\nn3_ori=%.12f\n", cov_temp.x, cov_temp.y, cov_temp.w,cov_temp.y, cov_temp.z, cov_temp.u,cov_temp.w,cov_temp.u,cov_temp.v,idx, normal[3*idx],normal[3*idx+1],normal[3*idx+2]);
//     printf("J=np.array([[%f,0,%f],[0,%f,%f]])\n",temp_use.x, temp_use.y, temp_use.z, temp_use.w);
//     //printf("cov_small_3D=np.array([[%f,%f,%f],[%f,%f,%f],[%f,%f,%f]])\ncov_large_3D=np.array([[%f,%f,%f],[%f,%f,%f],[%f,%f,%f]])\nview_matrix=np.array([[%f,%f,%f],[%f,%f,%f],[%f,%f,%f]])\nJ=np.array([[%f,0,%f],[0,%f,%f]])\n\n",cov3D_small[0], cov3D_small[1], cov3D_small[2],cov3D_small[1], cov3D_small[3], cov3D_small[4], cov3D_small[2], cov3D_small[4], cov3D_small[5],cov3D[0], cov3D[1], cov3D[2],cov3D[1], cov3D[3], cov3D[4], cov3D[2], cov3D[4], cov3D[5],viewmatrix[0], viewmatrix[4], viewmatrix[8],viewmatrix[1], viewmatrix[5], viewmatrix[9],viewmatrix[2], viewmatrix[6], viewmatrix[10], temp_use.x, temp_use.y, temp_use.z, temp_use.w);
//     printf("gs_normal=[%f,%f,%f]\ncorrect_gs_normal=[%f,%f,%f]\nconicx=%f\nconicy=%f\nconicz=%f\ncon_o2x=%f\ncon_o2y=%f\ncon_o2z=%f\n",gs_normal.x,gs_normal.y,gs_normal.z,temp_normal.x*first_divide/temp_normal.z, temp_normal.y*first_divide/temp_normal.z, first_divide/temp_normal.z,conic.x, conic.y, conic.z,x_term*first_divide, y_term*first_divide, first_divide);
//     printf("width=%d\nheight=%d\nwidth_small=%d\nheight_small=%d\nwidth_another=%d\nheight_another=%d\nwidth_small_another=%d\nheight_small_another=%d\npointx=%f\npointy=%f\n\n\n",width, height, width_small, height_small,width_another, height_another, width_small_another, height_small_another, point_image.x, point_image.y);
//     unlock(&mutex);
//     }
    ///////////////////////

}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity1,
	const float4* __restrict__ conic_opacity2,
	const float4* __restrict__ conic_opacity4,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float3* __restrict__ normal)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity1[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity2[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity4[BLOCK_SIZE];
	__shared__ float3 collected_normal[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity1[block.thread_rank()] = conic_opacity1[coll_id];
			collected_conic_opacity2[block.thread_rank()] = conic_opacity2[coll_id];
			collected_conic_opacity4[block.thread_rank()] = conic_opacity4[coll_id];
			collected_normal[block.thread_rank()] = normal[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];

			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o1 = collected_conic_opacity1[j];  //why con_o1 have three values???
			float4 con_o2 = collected_conic_opacity2[j];
			float4 con_o4 = collected_conic_opacity4[j];
			float3 norm_use = collected_normal[j];

            float power = -0.5f * (con_o1.x * d.x * d.x + con_o1.z * d.y * d.y) - con_o1.y * d.x * d.y; //
//             if (power > 0.0f || power < -4.8f)
// 				continue;
            float mask = (power <= 0.0f && power >= -4.8f) ? 1.0f : 0.0f;
            if (mask == 0.0f) continue;

			float exp_part = exp(power); //(con_o1.x * d.x * d.x + con_o1.z * d.y * d.y) - con_o1.y * d.x * d.y);
			float tanh_part1 = 1.0f+con_o4.x*erff(((norm_use.x+con_o2.x)*d.x + (norm_use.y+con_o2.y)*d.y));

//             tanh_part1 = 1.0f+erff(((norm_use.x+con_o2.x)*d.x + (norm_use.y+con_o2.y)*d.y));
//             tanh_part1 = 1.0f+erff(((norm_use.x+con_o2.x)*d.x + (norm_use.y+con_o2.y)*d.y));

			float tanh_part2 = 2.0f-tanh_part1;//(1.0f+tanh((norm_use.x*usex + norm_use.y*usey)/(-1.4142135f*norm_use.z)));

			float alpha1 = con_o1.w * exp_part * tanh_part1;
			// x,y need to be negative for another half
			float alpha2 = con_o2.w * exp_part * tanh_part2;
			//equation 9 in half gaussian paper
			float alpha = min(0.99f,0.5f*(alpha1 + alpha2)); //0.5 is for normalize each of the distribution

// 			if (alpha < 1.0f / 255.0f)
// 				continue;
            float alpha_mask = (alpha >= 1.0f / 255.0f) ? 1.0f : 0.0f;
            if (alpha_mask == 0.0f) continue;

			float test_T = T * (1 - alpha);
// 			if (test_T < 0.0001f)
// 			{
// 				done = true;
// 				continue;
// 			}
            done = (test_T < 0.0001f) ? true : done;
            if (done) continue;

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::render(
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
	const float3* normal)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity1,
		conic_opacity2,
		conic_opacity4,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		normal);
}

void FORWARD::preprocess(int P, int D, int M,
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
	float3* save_normal)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		normal,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		cov3D_precomp_small,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		cov3D_smalls,
		rgb,
		conic_opacity1,
		conic_opacity2,
		conic_opacity3,
		conic_opacity4,
		conic_opacity5,
		conic_opacity6,
		grid,
		tiles_touched,
		prefiltered,
		save_normal
		);
}

