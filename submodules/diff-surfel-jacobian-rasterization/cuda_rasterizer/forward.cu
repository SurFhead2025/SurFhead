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
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdexcept>//for debug
#include <iostream>//debug
#include <cmath>
#include <cfloat>
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

__device__ void normalize(float3 &normal, float &length) {
    // Calculate the length of the vector
    length = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);

    // Check to avoid division by zero using FLT_EPSILON
	normal.x /= length;
	normal.y /= length;
	normal.z /= length;
    // if (length > FLT_EPSILON) {
    //     normal.x /= length;
    //     normal.y /= length;
    //     normal.z /= length;
    // } else {
    //     // Optionally set normal to a default value, e.g., a unit vector along the z-axis
    //     normal.x = 0.0f;
    //     normal.y = 0.0f;
    //     normal.z = 1.0f;
    //     length = 1.0f; // The length of the default unit vector
    // }
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void compute_transmat(
	const float3& p_orig,
	const glm::vec2 scale,
	const glm::vec4 rot,
	const glm::mat3 Jacobian,
	const glm::mat3 Jacobian_inv,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H, 
	glm::mat3 &T,
	float3 &normal,
	const int index
) {

	glm::mat3 R = quat_to_rotmat(rot);
	glm::mat3 S = scale_to_mat(scale, 1.0f);
	// glm::mat3 L = R * S;
	glm::mat3 L = Jacobian * R * S;//*
	glm::mat3 L_inv = Jacobian_inv * R * S;//*


	// if (index == 0){
	// 		printf("L fw:\n");
	// 		for (int i = 0; i < 3; ++i) {
	// 			for (int j = 0; j < 3; ++j) {
	// 				printf("%f ", L[j][i]);
	// 			}
	// 			printf("\n");
	// 		}
	// 	}
	// if (index == 0){
	// 		printf("R fw:\n");
	// 		for (int i = 0; i < 3; ++i) {
	// 			for (int j = 0; j < 3; ++j) {
	// 				printf("%f ", R[j][i]);
	// 			}
	// 			printf("\n");
	// 		}
	// 	}
	// if (index == 0){
	// 		printf("Jinv fw:\n");
	// 		for (int i = 0; i < 3; ++i) {
	// 			for (int j = 0; j < 3; ++j) {
	// 				printf("%f ", Jacobian_inv[j][i]);
	// 			}
	// 			printf("\n");
	// 		}
	// 	}
	// if (index == 0) {
    //     printf("Index: %d\n", index);
	// 	// printf("Forward:: %d\n");
    //     printf("Jacobian in FW:\n");
    //     for (int i = 0; i < 3; ++i) {
    //         for (int j = 0; j < 3; ++j) {
    //             printf("%f ", Jacobian[j][i]);
    //         }
    //         printf("\n");
    //     }
	// }

    //     printf("R:\n");
    //     for (int i = 0; i < 3; ++i) {
    //         for (int j = 0; j < 3; ++j) {
    //             printf("%f ", R[j][i]);
    //         }
    //         printf("\n");
    //     }

    //     printf("S:\n");
    //     for (int i = 0; i < 3; ++i) {
    //         for (int j = 0; j < 3; ++j) {
    //             printf("%f ", S[j][i]);
    //         }
    //         printf("\n");
    //     }

    //     printf("L:\n");
    //     for (int i = 0; i < 3; ++i) {
    //         for (int j = 0; j < 3; ++j) {
    //             printf("%f ", L[j][i]);
    //         }
    //         printf("\n");
    //     }
    // }

	// glm::mat3 L = R * S;
	// glm::mat3 L = Jacobian * R * S;
	//! glm is column major 3x4 means 3 columns and 4 rows.;;
	// center of Gaussians in the camera coordinate
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
	); //! same with H matrix

	glm::mat4 world2ndc = glm::mat4(
		projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
		projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
		projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
		projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
	);
	// if (index == 0){
	// 		printf("world2ndc fw:\n");
	// 		for (int i = 0; i < 4; ++i) {
	// 			for (int j = 0; j < 4; ++j) {
	// 				printf("%f ", world2ndc[j][i]);
	// 			}
	// 			printf("\n");
	// 		}
	// 	}
	glm::mat3x4 ndc2pix = glm::mat3x4(
		glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
		glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
		glm::vec4(0.0, 0.0, 0.0, 1.0)
	);
	//!지금 모든 메트릭스가 트랜스포즈가 아님
	//! our case, actually, splat2mesh @ mesh2world @ world2ndc @ ndc2pix
	T = glm::transpose(splat2world) * world2ndc * ndc2pix; //? This is (WH)^T, not WH
	//! 3x4 @ 4x4 @ 4x3
	//! change it to (WJH)^T 
	//! == (JH)^T @ W^T 
	//! == H^T @ J^T @ W^T 
	//! == H^T @ J^T @ (world2cam@cam2pix)^T 
	//! == H^T @ J^T @ cam2pix^T @world2cam^T

	//! TODO: normalize normal
	
	// //? TW
	float3 normal_nn;
	float length;

	normal_nn.x = L_inv[2].x;
    normal_nn.y = L_inv[2].y;
    normal_nn.z = L_inv[2].z;
	// normalize(normal_nn, length);
	normal = transformVec4x3({normal_nn.x, normal_nn.y, normal_nn.z}, viewmatrix);
	// //?
	// normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);//*
	// normal = transformVec4x3({L_inv[2].x, L_inv[2].y, L_inv[2].z}, viewmatrix);//*
	//? normal in camera space
#if DUAL_VISIABLE
	float multiplier = normal.z < 0 ? 1: -1;
	normal = multiplier * normal;
#endif
}

// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
__device__ bool compute_aabb(
	glm::mat3 T, 
	float2& point_image,
	float2 & extent
) {
	float3 T0 = {T[0][0], T[0][1], T[0][2]};
	float3 T1 = {T[1][0], T[1][1], T[1][2]};
	float3 T3 = {T[2][0], T[2][1], T[2][2]};

	// Compute AABB
	float3 temp_point = {1.0f, 1.0f, -1.0f};
	float distance = sumf3(T3 * T3 * temp_point);
	float3 f = (1 / distance) * temp_point;
	if (distance == 0.0) return false;

	point_image = {
		sumf3(f * T0 * T3),
		sumf3(f * T1 * T3)
	};  
	
	float2 temp = {
		sumf3(f * T0 * T0),
		sumf3(f * T1 * T1)
	};
	float2 half_extend = point_image * point_image - temp;
	extent = sqrtf2(maxf2(1e-4, half_extend));
	return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const glm::mat3* Jacobians,
	const glm::mat3* Jacobians_inv,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
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
	
	// Compute transformation matrix
	glm::mat3 T;
	float3 normal;
	if (transMat_precomp == nullptr)
	{
		compute_transmat(((float3*)orig_points)[idx], scales[idx], rotations[idx], Jacobians[idx], Jacobians_inv[idx], projmatrix, viewmatrix, W, H, T, normal, idx);
		float3 *T_ptr = (float3*)transMats;
		T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
		T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
		T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};
	} else {
		glm::vec3 *T_ptr = (glm::vec3*)transMat_precomp;
		T = glm::mat3(
			T_ptr[idx * 3 + 0], 
			T_ptr[idx * 3 + 1],
			T_ptr[idx * 3 + 2]
		);
		normal = make_float3(0.0, 0.0, 1.0);
	}

	// Compute center and radius
	float2 point_image;
	float radius;
	{
		float2 extent;
		bool ok = compute_aabb(T, point_image, extent);
		if (!ok) return;
		radius = 3.0f * ceil(max(extent.x, extent.y));
	}

	uint2 rect_min, rect_max;
	getRect(point_image, radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Compute colors 
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view.z;
	radii[idx] = (int)radius;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
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
	float focal_x, float focal_y,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ transMats,
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others,
	// int* __restrict__ gaussian_indices,  //? Add this parameter
    int* __restrict__ n_contrib_pixel,   //? Add this parameter
    int max_gaussians_per_pixel,
	float* __restrict__ top_weights,
	float* __restrict__ top_depths,
	bool* __restrict__ visible_points,
	float tight_pruning_threshold) //?
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};

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
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };


#if RENDER_AXUTILITY
	// render axutility ouput
	float N[3] = {0};
	float D = { 0 };
	float M1 = {0};
	float M2 = {0};
	float distortion = {0};
	float median_depth = {0};
	// float median_weight = {0};
	float median_contributor = {-1};

#endif

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) //? for a single tile
	{
		// End if entire block votes that it is done rasterizing
		
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress]; //? mayb global id
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			//?  camera space에서 받아서 여기서 소팅
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
		}
		block.sync();

		// Iterate over current batch (pixel)
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// if (block.thread_rank() == 0 && j == 0) {
			// 	printf("Thread %d: Starting processing for batch %d\n", block.thread_rank(), i);
			// }//?
			// Keep track of current position in range
			contributor++;

			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			float3 k = pix.x * Tw - Tu; //? T is WH!! and pix.x is hx(screen)
			float3 l = pix.y * Tw - Tv; //? hy
			float3 p = cross(k, l); //? same with u(x), v(x)
			if (p.z == 0.0) continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 

			// compute intersection and depth
			float rho = min(rho3d, rho2d);
			float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; 
			if (depth < near_n) continue;
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w; //! real opacity

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, opa * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			float w = alpha * T;
#if RENDER_AXUTILITY
			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			float A = 1-T;
			float m = far_n / (far_n - near_n) * (1 - near_n / depth);
			distortion += (m * m * A + M2 - 2 * m * M1) * w;
			D  += depth * w;
			M1 += m * w;
			M2 += m * m * w;

			if (T > 0.5) {
				median_depth = depth;
				// median_weight = w;
				median_contributor = contributor;
			}
			// Render normal map
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w;
#endif

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
			// Store the gaussian index that influences this pixel
            if (inside) {
                // if (pix_id < 0 || pix_id >= W * H) {
                //     printf("Error: Invalid pix_id %d at (pix.x, pix.y) = (%d, %d)\n", pix_id, pix.x, pix.y);
                // }
                // int index = atomicAdd(&n_contrib_pixel[pix_id], 1);
				int curr_index = n_contrib_pixel[pix_id];
				if (curr_index < max_gaussians_per_pixel) {
					top_weights[pix_id * max_gaussians_per_pixel + curr_index] = w;
					top_depths[pix_id * max_gaussians_per_pixel + curr_index] = depth;
				}
                if (curr_index == 0 || w > tight_pruning_threshold) {
                    // if (range.x + progress < 0 || range.x + progress >= P) {
                    //     printf("Error: Invalid point_list index %d at progress %d\n", range.x + progress, progress);
                    // }
					// visible_points[curr_index] = true;
                    visible_points[collected_id[j]] = true;	
                }
				atomicAdd(&n_contrib_pixel[pix_id], 1); //? always
			}
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

#if RENDER_AXUTILITY
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
		// out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
#endif
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* means2D,
	const float* colors,
	const float* transMats,
	const float* depths,
	const float4* normal_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_others,
	// int* __restrict__ gaussian_indices,  //? Add this parameter
    int* __restrict__ n_contrib_pixel,   //? Add this parameter
    int max_gaussians_per_pixel,    //?
	float* __restrict__ top_weights,
	float* __restrict__ top_depths,
	bool* __restrict__ visible_points,
	float tight_pruning_threshold
	) //?
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		means2D,
		colors,
		transMats,
		depths,
		normal_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_others,
		// gaussian_indices,  //? Pass the new parameter
		n_contrib_pixel,
        max_gaussians_per_pixel,
		top_weights,
		top_depths,
		visible_points,
		tight_pruning_threshold);  //? Pass the new parameter
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			throw std::runtime_error(cudaGetErrorString(err));
		}
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const glm::mat3* Jacobians,
	const glm::mat3* Jacobians_inv,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, const int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		Jacobians,
		Jacobians_inv,
		opacities,
		shs,
		clamped,
		transMat_precomp,
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
		transMats,
		rgb,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}
