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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float3* dL_normal,
	float* dL_dcov,
	float3* dL_conic_another)
{
	auto idx = cg::this_grid().thread_rank();
// 	printf("%d\n",radii[idx]);
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, h_y / t.z);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0, dL_dd = 0, dL_de = 0, dL_df = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);


	if (dL_da == 0)
	{

        dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
        dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
        dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

        dL_dd = 2*dL_conic_another[idx].x;
        dL_de = 2*dL_conic_another[idx].y;
        dL_df = dL_conic_another[idx].z;

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
// 		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
// 		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
// 		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);
        dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc + T[0][0] * T[2][0] * dL_dd + T[1][0] * T[2][0] * dL_de + T[2][0] * T[2][0] * dL_df);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc + T[0][1] * T[2][1] * dL_dd + T[1][1] * T[2][1] * dL_de + T[2][1] * T[2][1] * dL_df);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc + T[0][2] * T[2][2] * dL_dd + T[1][2] * T[2][2] * dL_de + T[2][2] * T[2][2] * dL_df);


		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
// 		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
// 		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
// 		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	    dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc + (T[0][0] * T[2][1] + T[0][1] * T[2][0]) * dL_dd + (T[1][0] * T[2][1] + T[1][1] * T[2][0]) * dL_de + 2 * T[2][0] * T[2][1] * dL_df;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc + (T[0][0] * T[2][2] + T[0][2] * T[2][0]) * dL_dd + (T[1][0] * T[2][2] + T[1][2] * T[2][0]) * dL_de + 2 * T[2][0] * T[2][2] * dL_df;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc + (T[0][1] * T[2][2] + T[0][2] * T[2][1]) * dL_dd + (T[1][1] * T[2][2] + T[1][2] * T[2][1]) * dL_de + 2 * T[2][1] * T[2][2] * dL_df;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
// 	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
// 		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
// 	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
// 		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
// 	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
// 		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
// 	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
// 		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
// 	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
// 		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
// 	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
// 		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;
    float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db +
		(T[2][0] * Vrk[0][0] + T[2][1] * Vrk[0][1] + T[2][2] * Vrk[0][2]) * dL_dd;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db +
		(T[2][0] * Vrk[1][0] + T[2][1] * Vrk[1][1] + T[2][2] * Vrk[1][2]) * dL_dd;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db +
		(T[2][0] * Vrk[2][0] + T[2][1] * Vrk[2][1] + T[2][2] * Vrk[2][2]) * dL_dd;

	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db +
		(T[2][0] * Vrk[0][0] + T[2][1] * Vrk[0][1] + T[2][2] * Vrk[0][2]) * dL_de;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db +
		(T[2][0] * Vrk[1][0] + T[2][1] * Vrk[1][1] + T[2][2] * Vrk[1][2]) * dL_de;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db +
		(T[2][0] * Vrk[2][0] + T[2][1] * Vrk[2][1] + T[2][2] * Vrk[2][2]) * dL_de;

	float dL_dT20 = 2 * (T[2][0] * Vrk[0][0] + T[2][1] * Vrk[0][1] + T[2][2] * Vrk[0][2]) * dL_df +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dd +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_de;
	float dL_dT21 = 2 * (T[2][0] * Vrk[1][0] + T[2][1] * Vrk[1][1] + T[2][2] * Vrk[1][2]) * dL_df +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dd +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_de;
	float dL_dT22 = 2 * (T[2][0] * Vrk[2][0] + T[2][1] * Vrk[2][1] + T[2][2] * Vrk[2][2]) * dL_df +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dd +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_de;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float dL_dJ22 = W[2][0] * dL_dT20 + W[2][1] * dL_dT21 + W[2][2] * dL_dT22;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
// 	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;
    float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12 - h_y * tz2 * dL_dJ22;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;


	const float temp_x = dL_normal[idx].x;
	const float temp_y = dL_normal[idx].y;
	const float temp_z = dL_normal[idx].z;

    dL_normal[idx].x = temp_x*view_matrix[0]+temp_y*view_matrix[1]+temp_z*view_matrix[2];
    dL_normal[idx].y = temp_x*view_matrix[4]+temp_y*view_matrix[5]+temp_z*view_matrix[6];
    dL_normal[idx].z = temp_x*view_matrix[8]+temp_y*view_matrix[9]+temp_z*view_matrix[10];
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity1,
	const float4* __restrict__ conic_opacity2,
	const float4* __restrict__ conic_opacity4,
	const float3* __restrict__ conic_opacity5,
	const float3* __restrict__ normal,
	const float* viewmatrix,
	const float* projmatrix,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity1,
	float* __restrict__ dL_dopacity2,
	float3* __restrict__ dL_dnormal,
	float* __restrict__ dL_dcolors,
	float3* __restrict__ dL_conic_another)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity1[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity2[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity4[BLOCK_SIZE];
	__shared__ float3 collected_conic_opacity5[BLOCK_SIZE];
	__shared__ float3 collected_normal[3 * BLOCK_SIZE]; // is 3 correct？？？？
	__shared__ float collected_colors[C * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity1[block.thread_rank()] = conic_opacity1[coll_id];
			collected_conic_opacity2[block.thread_rank()] = conic_opacity2[coll_id];
			collected_conic_opacity4[block.thread_rank()] = conic_opacity4[coll_id];
			collected_conic_opacity5[block.thread_rank()] = conic_opacity5[coll_id];
			collected_normal[block.thread_rank()] = normal[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = {xy.x - pixf.x, xy.y - pixf.y};

			const float4 con_o1 = collected_conic_opacity1[j];
			const float4 con_o2 = collected_conic_opacity2[j];
			const float4 con_o4 = collected_conic_opacity4[j];
			const float sigma6 = collected_conic_opacity5[j].x;
		    const float sigma7 = collected_conic_opacity5[j].y;
			const float3 norm_use = collected_normal[j];


			const float power = -0.5f * (con_o1.x * d.x * d.x + con_o1.z * d.y * d.y) - con_o1.y * d.x * d.y; //(usex*usex + usey*usey); //
			if (power > 0.0f)
				continue;


            float denomi_z = norm_use.z;


            const float temp = ((norm_use.x+con_o2.x)*d.x + (norm_use.y+con_o2.y)*d.y);//(norm_use.x*d.x + norm_use.y*d.y)*denomi_z;
			const float tanh_result1 = erff(temp); //tanh((norm_use.x*(con_o1.x * d.x + con_o1.y * d.y) + norm_use.y*(con_o1.z * d.y))/(1.4142135f*norm_use.z));//tanh((norm_use.x*d.x + norm_use.y*d.y)/(1.4142135f*norm_use.z));
            const float tanh_result2 = -tanh_result1;//tanh((norm_use.x*usex + norm_use.y*usey)/(-1.4142135f*norm_use.z)); //tanh(-(norm_use.x*(con_o1.x * d.x + con_o1.y * d.y) + norm_use.y*(con_o1.z * d.y))/(1.4142135f*norm_use.z));//tanh(-(norm_use.x*d.x + norm_use.y*d.y)/(1.4142135f*norm_use.z));
            const float dev_erf1 = 2/sqrt(3.141592654f) * exp(-temp*temp);
            const float dev_erf2 = dev_erf1;
            const float exp_part = con_o4.x*exp(power);

            const float G1 = exp_part*(1+tanh_result1);//exp(power) * (1.f + tanh((norm_use.x*usex + norm_use.y*usey)/(1.4142135f*norm_use.z))); //exp(-0.5f*((con_o1.x * d.x + con_o1.y * d.y)*(con_o1.x * d.x + con_o1.y * d.y) + (con_o1.z * d.y)*(con_o1.z * d.y)))*(1.0f+tanh((norm_use.x*(con_o1.x * d.x + con_o1.y * d.y) + norm_use.y*(con_o1.z * d.y))/(1.4142135f*norm_use.z)))/(2.0f*3.1415926f);
            const float G2 = exp_part*(1+tanh_result2);//exp(power) * (1.f + tanh((norm_use.x*usex + norm_use.y*usey)/(-1.4142135f*norm_use.z))); //exp(-0.5f*((con_o1.x * d.x + con_o1.y * d.y)*(con_o1.x * d.x + con_o1.y * d.y) + (con_o1.z * d.y)*(con_o1.z * d.y)))*(1.0f+tanh(-(norm_use.x*(con_o1.x * d.x + con_o1.y * d.y) + norm_use.y*(con_o1.z * d.y))/(1.4142135f*norm_use.z)))/(2.0f*3.1415926f);


			const float alpha1 = con_o1.w * G1;//exp(-0.5f*((con_o1.x * d.x + con_o1.y * d.y)*(con_o1.x * d.x + con_o1.y * d.y) + (con_o1.y * d.x + con_o1.z * d.y)*(con_o1.y * d.x + con_o1.z * d.y)))*(1.0f+tanh((norm_use.x*(con_o1.x * d.x + con_o1.y * d.y) + norm_use.y*(con_o1.y * d.x + con_o1.z * d.y))/(1.4142135f*norm_use.z)))/(2*3.1415926f); //min(0.99f, con_o.w * exp(power));
			// x,y need to be negative for another half
			const float alpha2 = con_o2.w * G2;//exp(-0.5f*((con_o1.x * d.x + con_o1.y * d.y)*(con_o1.x * d.x + con_o1.y * d.y) + (con_o1.y * d.x + con_o1.z * d.y)*(con_o1.y * d.x + con_o1.z * d.y)))*(1.0f+tanh(-(norm_use.x*(con_o1.x * d.x + con_o1.y * d.y) + norm_use.y*(con_o1.y * d.x + con_o1.z * d.y))/(1.4142135f*norm_use.z)))/(2*3.1415926f); //min(0.99f, con_o.w * exp(power));
			const float alpha = min(0.99f,0.5f*(alpha1 + alpha2));

            const float dG1_dx = (0.5f*W)*(-(con_o1.x*d.x + con_o1.y*d.y)*G1 + exp_part*dev_erf1*(norm_use.x+con_o2.x));//(0.5f*W)*(-(con_o1.x*d.x + con_o1.y*d.y)*G1 + exp_part*dev_erf1*(norm_use.x)*denomi_z); //(-(con_o1.x * d.x + con_o1.y * d.y)*G1 + exp_part*dev_erf1*(1.4142135f*norm_use.x/(W*norm_use.z))); //(2/W)*(-(con_o1.x*(con_o1.x * d.x + con_o1.y * d.y))*G1 *(2.0f*3.1415926f) + exp_part*((norm_use.x*con_o1.x)*(1.0f - tanh_result1*tanh_result1)/(1.4142135f*norm_use.z)))/(2.0f*3.1415926f);//exp_part*((2*d.x+2*d.x*tanh_result1+(1.0f-tanh_result1*tanh_result1)*norm_use.x/(1.4142135f*norm_use.z))/(2*3.1415926f));
            const float dG2_dx = (0.5f*W)*(-(con_o1.x*d.x + con_o1.y*d.y)*G2 - exp_part*dev_erf2*(norm_use.x+con_o2.x));//(0.5f*W)*(-(con_o1.x*d.x + con_o1.y*d.y)*G2 - exp_part*dev_erf2*(norm_use.x)*denomi_z); //(-(con_o1.x * d.x + con_o1.y * d.y)*G2 - exp_part*dev_erf2*(1.4142135f*norm_use.x/(W*norm_use.z))); //(2/W)*(-(con_o1.x*(con_o1.x * d.x + con_o1.y * d.y))*G2 *(2.0f*3.1415926f) + exp_part*((norm_use.x*con_o1.x)*(1.0f - tanh_result2*tanh_result2)/(-1.4142135f*norm_use.z)))/(2.0f*3.1415926f);//exp_part*((2*d.x+2*d.x*tanh_result2+(1.0f-tanh_result2*tanh_result2)*norm_use.x/(1.4142135f*norm_use.z))/(2*3.1415926f));
            const float dG1_dy = (0.5f*H)*(-(con_o1.z*d.y + con_o1.y*d.x)*G1 + exp_part*dev_erf1*(norm_use.y+con_o2.y));//(0.5f*H)*(-(con_o1.z*d.y + con_o1.y*d.x)*G1 + exp_part*dev_erf1*(norm_use.y)*denomi_z); //(-(con_o1.z * d.y + con_o1.y * d.x)*G1 + exp_part*dev_erf1*(1.4142135f*norm_use.y/(H*norm_use.z))); //(2/H)*(-(con_o1.y*(con_o1.x * d.x + con_o1.y * d.y) + con_o1.z*(con_o1.z * d.y))*G1 *(2.0f*3.1415926f) + exp_part*((norm_use.x*con_o1.y + norm_use.y*con_o1.z)*(1.0f - tanh_result1*tanh_result1)/(1.4142135f*norm_use.z)))/(2.0f*3.1415926f);//exp_part*((2*d.y+2*d.y*tanh_result1+(1.0f-tanh_result1*tanh_result1)*norm_use.y/(1.4142135f*norm_use.z))/(2*3.1415926f));
            const float dG2_dy = (0.5f*H)*(-(con_o1.z*d.y + con_o1.y*d.x)*G2 - exp_part*dev_erf1*(norm_use.y+con_o2.y));//(0.5f*H)*(-(con_o1.z*d.y + con_o1.y*d.x)*G2 - exp_part*dev_erf1*(norm_use.y)*denomi_z); //(-(con_o1.z * d.y + con_o1.y * d.x)*G2 - exp_part*dev_erf2*(1.4142135f*norm_use.y/(H*norm_use.z))); //(2/H)*(-(con_o1.y*(con_o1.x * d.x + con_o1.y * d.y) + con_o1.z*(con_o1.z * d.y))*G2 *(2.0f*3.1415926f) + exp_part*((norm_use.x*con_o1.y + norm_use.y*con_o1.z)*(1.0f - tanh_result2*tanh_result2)/(-1.4142135f*norm_use.z)))/(2.0f*3.1415926f);//exp_part*((2*d.y+2*d.y*tanh_result2+(1.0f-tanh_result2*tanh_result2)*norm_use.y/(1.4142135f*norm_use.z))/(2*3.1415926f));

            const float dG1_dcx = -(0.5f*d.x*d.x)*G1 + exp_part*dev_erf1*(0.5f*sigma6*sigma6*1.99999982358225f*con_o2.z*con_o2.z*temp + con_o2.z*sigma6*d.x);//-(0.5f*d.x*d.x)*G1; //+ exp_part*dev_erf1*(norm_use.x*d.x)*denomi_z; //-0.5f*d.x*d.x*G1; //(-d.x*(con_o1.x * d.x + con_o1.y * d.y)*G1 *(2.0f*3.1415926f) + exp_part*(((norm_use.x * d.x)/(1.4142135f*norm_use.z)*(1.0f - tanh_result1*tanh_result1))))/(2.0f*3.1415926f);
            const float dG2_dcx = -(0.5f*d.x*d.x)*G2 - exp_part*dev_erf2*(0.5f*sigma6*sigma6*1.99999982358225f*con_o2.z*con_o2.z*temp + con_o2.z*sigma6*d.x);//-(0.5f*d.x*d.x)*G2; //- exp_part*dev_erf1*(norm_use.x*d.x)*denomi_z; //-0.5f*d.x*d.x*G2; //(-d.x*(con_o1.x * d.x + con_o1.y * d.y)*G2 *(2.0f*3.1415926f) + exp_part*(((norm_use.x * d.x)/(-1.4142135f*norm_use.z)*(1.0f - tanh_result2*tanh_result2))))/(2.0f*3.1415926f);
            const float dG1_dcy = -(0.5f*d.x*d.y)*G1 + 0.5f*exp_part*dev_erf1*(sigma6*sigma7*1.99999982358225f*con_o2.z*con_o2.z*temp + con_o2.z*(sigma7*d.x+sigma6*d.y));//-(0.5f*d.x*d.y)*G1; //+ exp_part*dev_erf1*(norm_use.x*d.y)*denomi_z; //-d.x*d.y*G1; //(-(d.y*(con_o1.x * d.x + con_o1.y * d.y))*G1 *(2.0f*3.1415926f) + exp_part*(((norm_use.x * d.y)/(1.4142135f*norm_use.z)*(1.0f - tanh_result1*tanh_result1))))/(2.0f*3.1415926f);
            const float dG2_dcy = -(0.5f*d.x*d.y)*G2 - 0.5f*exp_part*dev_erf2*(sigma6*sigma7*1.99999982358225f*con_o2.z*con_o2.z*temp + con_o2.z*(sigma7*d.x+sigma6*d.y));//-(0.5f*d.x*d.y)*G2; //- exp_part*dev_erf1*(norm_use.x*d.y)*denomi_z; //-d.x*d.y*G2; //(-(d.y*(con_o1.x * d.x + con_o1.y * d.y))*G2 *(2.0f*3.1415926f) + exp_part*(((norm_use.x * d.y)/(-1.4142135f*norm_use.z)*(1.0f - tanh_result2*tanh_result2))))/(2.0f*3.1415926f);
            const float dG1_dcz = -(0.5f*d.y*d.y)*G1 + exp_part*dev_erf1*(0.5f*sigma7*sigma7*1.99999982358225f*con_o2.z*con_o2.z*temp + con_o2.z*sigma7*d.y);//-(0.5f*d.y*d.y)*G1; //+ exp_part*dev_erf1*(norm_use.y*d.y)*denomi_z; //-0.5f*d.y*d.y*G1; //(-d.y*(con_o1.z * d.y)*G1 *(2.0f*3.1415926f) + exp_part*(((norm_use.y * d.y)/(1.4142135f*norm_use.z)*(1.0f - tanh_result1*tanh_result1))))/(2.0f*3.1415926f);
            const float dG2_dcz = -(0.5f*d.y*d.y)*G2 - exp_part*dev_erf2*(0.5f*sigma7*sigma7*1.99999982358225f*con_o2.z*con_o2.z*temp + con_o2.z*sigma7*d.y);//-(0.5f*d.y*d.y)*G2; //- exp_part*dev_erf1*(norm_use.y*d.y)*denomi_z; //-0.5f*d.y*d.y*G2; //(-d.y*(con_o1.z * d.y)*G2 *(2.0f*3.1415926f) + exp_part*(((norm_use.y * d.y)/(-1.4142135f*norm_use.z)*(1.0f - tanh_result2*tanh_result2))))/(2.0f*3.1415926f);

            const float dG1_dn1 =  exp_part*dev_erf1*d.x*denomi_z;//exp_part*dev_erf1*d.x*denomi_z; //(con_o1.x*d.x + con_o1.y*d.y)*denomi_z;//(1.4142135f*d.x/(W*norm_use.z)); //(exp_part)*(1.0f - tanh_result1*tanh_result1)*((con_o1.x * d.x + con_o1.y * d.y)/(1.4142135f*norm_use.z))/(2.0f*3.1415926f);//exp_part*(1.0f-tanh_result1*tanh_result1)*d.x/(1.4142135f*norm_use.z) / (2*3.1415926f);
            const float dG1_dn2 =  exp_part*dev_erf1*d.y*denomi_z;//exp_part*dev_erf1*d.y*denomi_z; //(con_o1.z*d.y)*denomi_z;//(1.4142135f*d.y/(H*norm_use.z)); //(exp_part)*(1.0f - tanh_result1*tanh_result1)*((con_o1.z * d.y)/(1.4142135f*norm_use.z))/(2.0f*3.1415926f);//exp_part*(1.0f-tanh_result1*tanh_result1)*d.y/(1.4142135f*norm_use.z) / (2*3.1415926f);
            const float dG1_dn3 =  -exp_part*dev_erf1*((norm_use.x*d.x + norm_use.y*d.y)*norm_use.z/(con_o2.z));//-exp_part*dev_erf1*((norm_use.x*d.x + norm_use.y*d.y)/(sqrt(2.f)*norm_use.z*norm_use.z)); //((norm_use.x*usex + norm_use.y*usey)/(1.4142135f*norm_use.z*norm_use.z));//(-1.4142135f*(norm_use.x*d.x/W + norm_use.y*d.y/H)/(norm_use.z*norm_use.z)); //(exp_part)*(1.0f - tanh_result1*tanh_result1)*((norm_use.x*(con_o1.x * d.x + con_o1.y * d.y) + norm_use.y*(con_o1.z * d.y))/(-1.4142135f*norm_use.z*norm_use.z))/(2.0f*3.1415926f);//exp_part*(1.0f-tanh_result1*tanh_result1)*(norm_use.x*d.x + norm_use.y*d.y)/(1.4142135f*norm_use.z) / (2*3.1415926f);

            const float dG2_dn1 =  -dG1_dn1;//exp_part*dev_erf2*(-1.4142135f*d.x/(W*norm_use.z));//(exp_part)*(1.0f - tanh_result2*tanh_result2)*((con_o1.x * d.x + con_o1.y * d.y)/(-1.4142135f*norm_use.z))/(2.0f*3.1415926f);//-exp_part*(1.0f-tanh_result2*tanh_result2)*d.x/(1.4142135f*norm_use.z) / (2*3.1415926f);
            const float dG2_dn2 =  -dG1_dn2;//exp_part*dev_erf2*(-1.4142135f*d.y/(H*norm_use.z)); //(exp_part)*(1.0f - tanh_result2*tanh_result2)*((con_o1.z * d.y)/(-1.4142135f*norm_use.z))/(2.0f*3.1415926f);//-exp_part*(1.0f-tanh_result2*tanh_result2)*d.y/(1.4142135f*norm_use.z) / (2*3.1415926f);
            const float dG2_dn3 =  -dG1_dn3;//exp_part*dev_erf2*(1.4142135f*(norm_use.x*d.x/W + norm_use.y*d.y/H)/(norm_use.z*norm_use.z));//(exp_part)*(1.0f - tanh_result2*tanh_result2)*((norm_use.x*(con_o1.x * d.x + con_o1.y * d.y) + norm_use.y*(con_o1.z * d.y))/(1.4142135f*norm_use.z*norm_use.z))/(2.0f*3.1415926f);//-exp_part*(1.0f-tanh_result2*tanh_result2)*(norm_use.x*d.x + norm_use.y*d.y)/(1.4142135f*norm_use.z) / (2*3.1415926f);

            const float dG1_ds6 = 0.5f*exp_part*dev_erf1*((sigma6*con_o1.x+sigma7*con_o1.y)*1.99999982358225f*con_o2.z*con_o2.z*temp + con_o2.z*(con_o1.x*d.x + con_o1.y*d.y));
            const float dG1_ds7 = 0.5f*exp_part*dev_erf1*((sigma6*con_o1.y+sigma7*con_o1.z)*1.99999982358225f*con_o2.z*con_o2.z*temp + con_o2.z*(con_o1.y*d.x + con_o1.z*d.y));
            const float dG1_ds8 = exp_part*dev_erf1*(-0.5f*1.99999982358225f*con_o2.z*con_o2.z*temp);

            const float dG2_ds6 = -dG1_ds6;
            const float dG2_ds7 = -dG1_ds7;
            const float dG2_ds8 = -dG1_ds8;

			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dalpha*(0.5f*con_o1.w*dG1_dx + 0.5f*con_o2.w*dG2_dx));//dL_dalpha* (0.5f*con_o1.w*dG1_dx + 0.5f*con_o2.w*dG2_dx));//dL_dalpha* 0.5*(alpha1*(-d.x*dev_xy_front1 + dev_xy_back_x1) + alpha2*(-d.x*dev_xy_front2 + dev_xy_back_x2)));//dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dalpha*(0.5f*con_o1.w*dG1_dy + 0.5f*con_o2.w*dG2_dy));//dL_dalpha* (0.5f*con_o1.w*dG1_dy + 0.5f*con_o2.w*dG2_dy));//dL_dalpha* 0.5*(alpha1*(-d.y*dev_xy_front1 + dev_xy_back_y1) + alpha2*(-d.x*dev_xy_front2 + dev_xy_back_y2)));//dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)

			atomicAdd(&dL_dconic2D[global_id].x, dL_dalpha*(0.5f*con_o1.w*dG1_dcx + 0.5f*con_o2.w*dG2_dcx));//-0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, dL_dalpha*(0.5f*con_o1.w*dG1_dcy + 0.5f*con_o2.w*dG2_dcy));//-0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, dL_dalpha*(0.5f*con_o1.w*dG1_dcz + 0.5f*con_o2.w*dG2_dcz));//-0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity1[global_id]), 0.5f * G1 * dL_dalpha);
			atomicAdd(&(dL_dopacity2[global_id]), 0.5f * G2 * dL_dalpha);

			//also need to calculate grad for normal

            atomicAdd(&dL_dnormal[global_id].x, 0.5f* dL_dalpha*(con_o1.w*dG1_dn1 + con_o2.w*dG2_dn1));//(projmatrix[0]*viewmatrix[0]+projmatrix[4]*viewmatrix[1]+projmatrix[8]*viewmatrix[2])*dL_dn1p + (projmatrix[0]*viewmatrix[4]+projmatrix[4]*viewmatrix[5]+projmatrix[8]*viewmatrix[6])*dL_dn2p + (projmatrix[0]*viewmatrix[8]+projmatrix[4]*viewmatrix[9]+projmatrix[8]*viewmatrix[10])*dL_dn3p);//0.5f*(alpha1*dev_part_nx1 + alpha2*dev_part_nx2)* dL_dalpha); //&dL_normal[global_id].x,
            atomicAdd(&dL_dnormal[global_id].y, 0.5f* dL_dalpha*(con_o1.w*dG1_dn2 + con_o2.w*dG2_dn2));//(projmatrix[1]*viewmatrix[0]+projmatrix[5]*viewmatrix[1]+projmatrix[9]*viewmatrix[2])*dL_dn1p + (projmatrix[1]*viewmatrix[4]+projmatrix[5]*viewmatrix[5]+projmatrix[9]*viewmatrix[6])*dL_dn2p + (projmatrix[1]*viewmatrix[8]+projmatrix[5]*viewmatrix[9]+projmatrix[9]*viewmatrix[10])*dL_dn3p);//0.5f*(alpha1*dev_part_ny1 + alpha2*dev_part_ny2)* dL_dalpha);
            atomicAdd(&dL_dnormal[global_id].z, 0.5f* dL_dalpha*(con_o1.w*dG1_dn3 + con_o2.w*dG2_dn3));//(projmatrix[2]*viewmatrix[0]+projmatrix[6]*viewmatrix[1]+projmatrix[10]*viewmatrix[2])*dL_dn1p + (projmatrix[2]*viewmatrix[4]+projmatrix[6]*viewmatrix[5]+projmatrix[10]*viewmatrix[6])*dL_dn2p + (projmatrix[2]*viewmatrix[8]+projmatrix[6]*viewmatrix[9]+projmatrix[10]*viewmatrix[10])*dL_dn3p);

            atomicAdd(&dL_conic_another[global_id].x, 0.5f* dL_dalpha*(con_o1.w*dG1_ds6 + con_o2.w*dG2_ds6)); //sigma6
            atomicAdd(&dL_conic_another[global_id].y, 0.5f* dL_dalpha*(con_o1.w*dG1_ds7 + con_o2.w*dG2_ds7)); //sigma7
            atomicAdd(&dL_conic_another[global_id].z, 0.5f* dL_dalpha*(con_o1.w*dG1_ds8 + con_o2.w*dG2_ds8)); //sigma8

		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const float3* normal,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	glm::vec3* dL_dnormal,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float3* dL_conic_another)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		(float3*)dL_dnormal,
		dL_dcov3D,
		(float3*)dL_conic_another);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity1,
	const float4* conic_opacity2,
	const float4* conic_opacity4,
	const float3* conic_opacity5,
	const float3* normal,
	const float* viewmatrix,
	const float* projmatrix,
	const float* colors,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity1,
	float* dL_dopacity2,
	float3* dL_dnormal,
	float* dL_dcolors,
	float3* dL_conic_another)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity1,
		conic_opacity2,
		conic_opacity4,
		conic_opacity5,
		normal,
		viewmatrix,
		projmatrix,
		colors,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity1,
		dL_dopacity2,
		dL_dnormal,
		dL_dcolors,
		dL_conic_another);
}