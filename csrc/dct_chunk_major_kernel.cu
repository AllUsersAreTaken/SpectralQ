#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Kernel 1: project token into frequency domain
template<int BS>
__global__ void project_token_kernel(
    const half* __restrict__ x,
    const half* __restrict__ dct_basis,
    half* __restrict__ x_freq,
    int n_chunks)
{
    int chunk = blockIdx.x;
    int tid = threadIdx.x;
    if (chunk >= n_chunks) return;
    float sum = 0.0f;
    int base = chunk * BS;
    for (int j = 0; j < BS; j++)
        sum += __half2float(x[base + j]) * __half2float(dct_basis[j * BS + tid]);
    x_freq[base + tid] = __float2half(sum);
}

// Kernel 2: GEMV with chunk-major packed uint32 coefficients
template<int BS>
__global__ void gemv_chunk_major_kernel(
    const half* __restrict__ x_freq,
    const uint32_t* __restrict__ qcoeff_packed,
    const half* __restrict__ qscale,
    const half* __restrict__ qzero,
    float* __restrict__ out,
    int out_features,
    int n_chunks,
    int pk)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= out_features) return;

    float acc = 0.0f;
    int pack_group = tid / 5;
    int slot = tid % 5;

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        int idx = chunk * BS + tid;
        float xf = __half2float(x_freq[idx]);

        int pidx = chunk * out_features * pk + row * pk + pack_group;
        uint32_t pack = qcoeff_packed[pidx];
        int c6 = (pack >> (slot * 6)) & 0x3F;

        float c = (float)c6 * __half2float(qscale[tid]) + __half2float(qzero[tid]);
        acc += c * xf;
    }

    __shared__ float red[BS];
    red[tid] = acc;
    __syncthreads();

    for (int s = BS >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[row] = red[0];
}

extern "C" {

__declspec(dllexport)
void launch_dct_forward(
    const half* x,
    const uint32_t* qcoeff_packed,
    const half* qscale,
    const half* qzero,
    const half* dct_basis,
    half* x_freq,
    float* out,
    int out_features,
    int in_features,
    int block_size,
    int packs_per_chunk,
    cudaStream_t stream)
{
    int n_chunks = in_features / block_size;
    dim3 gp(n_chunks);
    dim3 bp(block_size);
    dim3 gg(out_features);
    dim3 bg(block_size);

    switch (block_size) {
        case 64:
            project_token_kernel<64><<<gp, bp, 0, stream>>>(x, dct_basis, x_freq, n_chunks);
            gemv_chunk_major_kernel<64><<<gg, bp, 0, stream>>>(x_freq, qcoeff_packed, qscale, qzero, out, out_features, n_chunks, packs_per_chunk);
            break;
        case 128:
            project_token_kernel<128><<<gp, bp, 0, stream>>>(x, dct_basis, x_freq, n_chunks);
            gemv_chunk_major_kernel<128><<<gg, bp, 0, stream>>>(x_freq, qcoeff_packed, qscale, qzero, out, out_features, n_chunks, packs_per_chunk);
            break;
        case 256:
            project_token_kernel<256><<<gp, bp, 0, stream>>>(x, dct_basis, x_freq, n_chunks);
            gemv_chunk_major_kernel<256><<<gg, bp, 0, stream>>>(x_freq, qcoeff_packed, qscale, qzero, out, out_features, n_chunks, packs_per_chunk);
            break;
        case 512:
            project_token_kernel<512><<<gp, bp, 0, stream>>>(x, dct_basis, x_freq, n_chunks);
            gemv_chunk_major_kernel<512><<<gg, bp, 0, stream>>>(x_freq, qcoeff_packed, qscale, qzero, out, out_features, n_chunks, packs_per_chunk);
            break;
    }
}

}
