/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Template project which demonstrates the basics on how to setup a project 
 * example application.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil_inline.h>
#include <shrQATest.h>

//======================== THRUST Libs
// http://code.google.com/p/thrust/
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
// includes, kernels
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int runMultiGPUExampleForSVM( ) {
	// Query device properties
	cudaDeviceProp prop[64];
	int gpuid_tesla[64]; // we want to find the first two GPU's that can support P2P
	int gpu_count = 0; // GPUs that meet the criteria
	int gpu_n;
	cutilSafeCall(cudaGetDeviceCount(&gpu_n));
	printf("CUDA-capable device count: %i\n", gpu_n);
	for (int i = 0; i < gpu_n; i++) {
		cutilSafeCall(cudaGetDeviceProperties(&prop[i], i));
		// Only Tesla boards based on Fermi can support P2P
		if ((!STRNCASECMP(prop[i].name, "Tesla", 5)) && (prop[i].major >= 2)
#ifdef _WIN32
		// on Windows (64-bit), the Tesla Compute Cluster driver for windows must be enabled
		&& prop[i].tccDriver
#endif
		)
		{
			// This is an array of P2P capable GPUs
			gpuid_tesla[gpu_count++] = i;
		}
		printf("> GPU%d = \"%15s\"   capable of Peer-to-Peer (P2P)\n", i,
				prop[i].name);
	}

	// Enable peer access
	printf("Enabling peer access between GPU%d and GPU%d...\n", gpuid_tesla[0],
			gpuid_tesla[1]);
	cutilSafeCall(cudaSetDevice(gpuid_tesla[0]));
	cutilSafeCall(cudaDeviceEnablePeerAccess(gpuid_tesla[1], gpuid_tesla[0]));
	cutilSafeCall(cudaSetDevice(gpuid_tesla[1]));
	cutilSafeCall(cudaDeviceEnablePeerAccess(gpuid_tesla[0], gpuid_tesla[0]));

	// Check that we got UVA on both devices
	printf("Checking GPU%d and GPU%d for UVA capabilities...\n",
			gpuid_tesla[0], gpuid_tesla[1]);
	const bool has_uva = (prop[gpuid_tesla[0]].unifiedAddressing
			&& prop[gpuid_tesla[1]].unifiedAddressing);

	printf("> %s (GPU%d) supports UVA: %s\n", prop[gpuid_tesla[0]].name,
			gpuid_tesla[0], (prop[gpuid_tesla[0]].unifiedAddressing ? "Yes"
					: "No"));
	printf("> %s (GPU%d) supports UVA: %s\n", prop[gpuid_tesla[1]].name,
			gpuid_tesla[1], (prop[gpuid_tesla[1]].unifiedAddressing ? "Yes"
					: "No"));

	if (has_uva) {
		printf("Both GPUs can support UVA, enabling...\n");
	} else {
		printf(
				"At least one of the two GPUs does NOT support UVA, waiving test.\n");
		printf("PASSED\n");
		exit(EXIT_SUCCESS);
	}

	// Allocate buffers
	const size_t buf_size = 1024 * 1024 * 512 * sizeof(float);
	printf("Allocating buffers (%iMB on GPU%d, GPU%d and CPU Host)...\n", int(buf_size / 1024 / 1024), gpuid_tesla[0], gpuid_tesla[1]);
	cutilSafeCall(cudaSetDevice(gpuid_tesla[0]));
	printf("choosing device 0\n");

	// manipulate memory

	// deallocate with device_free

	thrust::device_ptr<float> d_dev0 = thrust::device_malloc<float>(1024 * 1024 * 512);
	printf("choosing device 1\n");
	cutilSafeCall(cudaSetDevice(gpuid_tesla[1]));
	thrust::device_ptr<float> d_dev1 = thrust::device_malloc<float>(1024 * 1024 * 512);
	printf("done\n");

	//	thrust::host_vector<float> h_res0 = d_dev0;
	//	thrust::host_vector<float> h_res1 = d_dev1;
	//
	//	printf("suma: %f\n",h_res0[0]+h_res1[0]);


	//       cutilSafeCall(cudaSetDevice(gpuid_tesla[1]));
	//       float* g1;
	//       cutilSafeCall(cudaMalloc(&g1, buf_size));
	//       float* h0;
	//       cutilSafeCall(cudaMallocHost(&h0, buf_size)); // Automatically portable with UVA

	cutilSafeCall(cudaSetDevice(gpuid_tesla[0]));
	thrust::device_free(d_dev0 );
	cutilSafeCall(cudaSetDevice(gpuid_tesla[1]));
	thrust::device_free(d_dev1 );

	return 1;
}

