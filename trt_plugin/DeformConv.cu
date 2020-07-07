#include "DeformConv.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu

// [batch gemm]
// https://github.com/pytorch/pytorch/blob/master/aten/src/THC/generic/THCTensorMathBlas.cu

namespace
{
#define CUDA_KERNEL_LOOP(i, n)                                   \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
         i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define CUDA_ALIGN 256

template <typename T>
inline size_t get_size_aligned(size_t num_elem)
{
    size_t size = num_elem * sizeof(T);
    size_t extra_align = 0;
    if (size % CUDA_ALIGN != 0)
    {
        extra_align = CUDA_ALIGN - size % CUDA_ALIGN;
    }
    return size + extra_align;
}

template <typename T>
inline T *get_next_ptr(size_t num_elem, void *&workspace,
                       size_t &workspace_size)
{
    size_t size = get_size_aligned<T>(num_elem);
    if (size > workspace_size)
    {
        throw std::runtime_error("Workspace is too small!");
    }
    workspace_size -= size;
    T *ptr = reinterpret_cast<T *>(workspace);
    workspace =
        reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(workspace) + size);
    return ptr;
}

__device__ float dmcn_im2col_bilinear(const float *bottom_data,
                                      const int data_width, const int height,
                                      const int width, float h, float w)
{
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1 - lh, hw = 1 - lw;

    float v1 = 0;
    if (h_low >= 0 && w_low >= 0)
        v1 = bottom_data[h_low * data_width + w_low];
    float v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
        v2 = bottom_data[h_low * data_width + w_high];
    float v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
        v3 = bottom_data[h_high * data_width + w_low];
    float v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
        v4 = bottom_data[h_high * data_width + w_high];

    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

__device__ float dmcn_get_gradient_weight(float argmax_h, float argmax_w,
                                          const int h, const int w,
                                          const int height, const int width)
{
    if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
        argmax_w >= width)
    {
        // empty
        return 0;
    }

    int argmax_h_low = floor(argmax_h);
    int argmax_w_low = floor(argmax_w);
    int argmax_h_high = argmax_h_low + 1;
    int argmax_w_high = argmax_w_low + 1;

    float weight = 0;
    if (h == argmax_h_low && w == argmax_w_low)
        weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
    if (h == argmax_h_low && w == argmax_w_high)
        weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
    if (h == argmax_h_high && w == argmax_w_low)
        weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
    if (h == argmax_h_high && w == argmax_w_high)
        weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
    return weight;
}

__device__ float dmcn_get_coordinate_weight(float argmax_h, float argmax_w,
                                            const int height, const int width,
                                            const float *im_data,
                                            const int data_width,
                                            const int bp_dir)
{
    if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
        argmax_w >= width)
    {
        // empty
        return 0;
    }

    int argmax_h_low = floor(argmax_h);
    int argmax_w_low = floor(argmax_w);
    int argmax_h_high = argmax_h_low + 1;
    int argmax_w_high = argmax_w_low + 1;

    float weight = 0;

    if (bp_dir == 0)
    {
        if (argmax_h_low >= 0 && argmax_w_low >= 0)
            weight += -1 * (argmax_w_low + 1 - argmax_w) *
                      im_data[argmax_h_low * data_width + argmax_w_low];
        if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
            weight += -1 * (argmax_w - argmax_w_low) *
                      im_data[argmax_h_low * data_width + argmax_w_high];
        if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
            weight += (argmax_w_low + 1 - argmax_w) *
                      im_data[argmax_h_high * data_width + argmax_w_low];
        if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
            weight += (argmax_w - argmax_w_low) *
                      im_data[argmax_h_high * data_width + argmax_w_high];
    }
    else if (bp_dir == 1)
    {
        if (argmax_h_low >= 0 && argmax_w_low >= 0)
            weight += -1 * (argmax_h_low + 1 - argmax_h) *
                      im_data[argmax_h_low * data_width + argmax_w_low];
        if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
            weight += (argmax_h_low + 1 - argmax_h) *
                      im_data[argmax_h_low * data_width + argmax_w_high];
        if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
            weight += -1 * (argmax_h - argmax_h_low) *
                      im_data[argmax_h_high * data_width + argmax_w_low];
        if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
            weight += (argmax_h - argmax_h_low) *
                      im_data[argmax_h_high * data_width + argmax_w_high];
    }

    return weight;
}

__global__ void modulated_deformable_im2col_gpu_kernel(
    const int n, const float *data_im, const float *data_offset,
    const float *data_mask, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int channel_per_deformable_group,
    const int batch_size, const int num_channels, const int deformable_group,
    const int height_col, const int width_col, float *data_col)
{
    // launch channels * batch_size * height_col * width_col cores
    CUDA_KERNEL_LOOP(index, n)
    {
        // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation,
        // col_buffer is of shape (c*kw*kh, N, oh, ow) here columns is of shape (N,
        // c*kw*kh, oh * ow), need to adapt axis

        // index index of output matrix
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        // const int b_col = (index / width_col / height_col) % batch_size;
        const int b_col =
            (index / width_col / height_col / num_channels) % batch_size;
        // const int c_im = (index / width_col / height_col) / batch_size;
        const int c_im = (index / width_col / height_col) % num_channels;
        // const int c_col = c_im * kernel_h * kernel_w;
        const int c_col = c_im * kernel_h * kernel_w;

        // compute deformable group index
        const int deformable_group_index = c_im / channel_per_deformable_group;

        const int h_in = h_col * stride_h - pad_h;
        const int w_in = w_col * stride_w - pad_w;

        //  float *data_col_ptr = data_col + ((c_col * batch_size + b_col) *
        //  height_col + h_col) * width_col + w_col;
        float *data_col_ptr =
            data_col +
            ((b_col * num_channels * kernel_w * kernel_h + c_col) * height_col +
             h_col) *
                width_col +
            w_col;
        // const float* data_im_ptr = data_im + ((b_col * num_channels + c_im) *
        // height + h_in) * width + w_in;
        const float *data_im_ptr =
            data_im + (b_col * num_channels + c_im) * height * width;
        const float *data_offset_ptr =
            data_offset + (b_col * deformable_group + deformable_group_index) * 2 *
                              kernel_h * kernel_w * height_col * width_col;

        const float *data_mask_ptr =
            data_mask + (b_col * deformable_group + deformable_group_index) *
                            kernel_h * kernel_w * height_col * width_col;

        for (int i = 0; i < kernel_h; ++i)
        {
            for (int j = 0; j < kernel_w; ++j)
            {
                const int data_offset_h_ptr =
                    ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
                const int data_offset_w_ptr =
                    ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
                    w_col;
                const int data_mask_hw_ptr =
                    ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
                const float offset_h = data_offset_ptr[data_offset_h_ptr];
                const float offset_w = data_offset_ptr[data_offset_w_ptr];
                const float mask = data_mask_ptr[data_mask_hw_ptr];
                float val = static_cast<float>(0);
                const float h_im = h_in + i * dilation_h + offset_h;
                const float w_im = w_in + j * dilation_w + offset_w;
                // if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
                if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
                {
                    // const float map_h = i * dilation_h + offset_h;
                    // const float map_w = j * dilation_w + offset_w;
                    // const int cur_height = height - h_in;
                    // const int cur_width = width - w_in;
                    // val = dmcn_im2col_bilinear(data_im_ptr, width, cur_height,
                    // cur_width, map_h, map_w);
                    val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im,
                                               w_im);
                }
                *data_col_ptr = val * mask;
                // data_col_ptr += batch_size * height_col * width_col;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

__global__ void modulated_deformable_col2im_gpu_kernel(
    const int n, const float *data_col, const float *data_offset,
    const float *data_mask, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int deformable_group, const int height_col, const int width_col,
    float *grad_im)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        const int j = (index / width_col / height_col / batch_size) % kernel_w;
        const int i =
            (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
        const int c =
            index / width_col / height_col / batch_size / kernel_w / kernel_h;
        // compute the start and end of the output

        const int deformable_group_index = c / channel_per_deformable_group;

        int w_out = index % width_col;
        int h_out = (index / width_col) % height_col;
        int b = (index / width_col / height_col) % batch_size;
        int w_in = w_out * stride_w - pad_w;
        int h_in = h_out * stride_h - pad_h;

        const float *data_offset_ptr =
            data_offset + (b * deformable_group + deformable_group_index) * 2 *
                              kernel_h * kernel_w * height_col * width_col;
        const float *data_mask_ptr =
            data_mask + (b * deformable_group + deformable_group_index) * kernel_h *
                            kernel_w * height_col * width_col;
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
        const int data_mask_hw_ptr =
            ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
        const float offset_h = data_offset_ptr[data_offset_h_ptr];
        const float offset_w = data_offset_ptr[data_offset_w_ptr];
        const float mask = data_mask_ptr[data_mask_hw_ptr];
        const float cur_inv_h_data = h_in + i * dilation_h + offset_h;
        const float cur_inv_w_data = w_in + j * dilation_w + offset_w;

        const float cur_top_grad = data_col[index] * mask;
        const int cur_h = (int)cur_inv_h_data;
        const int cur_w = (int)cur_inv_w_data;
        for (int dy = -2; dy <= 2; dy++)
        {
            for (int dx = -2; dx <= 2; dx++)
            {
                if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
                    cur_w + dx < width && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
                    abs(cur_inv_w_data - (cur_w + dx)) < 1)
                {
                    int cur_bottom_grad_pos =
                        ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
                    float weight =
                        dmcn_get_gradient_weight(cur_inv_h_data, cur_inv_w_data,
                                                 cur_h + dy, cur_w + dx, height, width);
                    atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
                }
            }
        }
    }
}

__global__ void modulated_deformable_col2im_coord_gpu_kernel(
    const int n, const float *data_col, const float *data_im,
    const float *data_offset, const float *data_mask, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int offset_channels, const int deformable_group, const int height_col,
    const int width_col, float *grad_offset, float *grad_mask)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        float val = 0, mval = 0;
        int w = index % width_col;
        int h = (index / width_col) % height_col;
        int c = (index / width_col / height_col) % offset_channels;
        int b = (index / width_col / height_col) / offset_channels;
        // compute the start and end of the output

        const int deformable_group_index = c / (2 * kernel_h * kernel_w);
        const int col_step = kernel_h * kernel_w;
        int cnt = 0;
        const float *data_col_ptr =
            data_col + deformable_group_index * channel_per_deformable_group *
                           batch_size * width_col * height_col;
        const float *data_im_ptr =
            data_im + (b * deformable_group + deformable_group_index) *
                          channel_per_deformable_group / kernel_h / kernel_w *
                          height * width;
        const float *data_offset_ptr =
            data_offset + (b * deformable_group + deformable_group_index) * 2 *
                              kernel_h * kernel_w * height_col * width_col;
        const float *data_mask_ptr =
            data_mask + (b * deformable_group + deformable_group_index) * kernel_h *
                            kernel_w * height_col * width_col;

        const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

        for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group;
             col_c += col_step)
        {
            const int col_pos =
                (((col_c * batch_size + b) * height_col) + h) * width_col + w;
            const int bp_dir = offset_c % 2;

            int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
            int i =
                (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
            int w_out = col_pos % width_col;
            int h_out = (col_pos / width_col) % height_col;
            int w_in = w_out * stride_w - pad_w;
            int h_in = h_out * stride_h - pad_h;
            const int data_offset_h_ptr =
                (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
            const int data_offset_w_ptr =
                (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col +
                 w_out);
            const int data_mask_hw_ptr =
                (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
            const float offset_h = data_offset_ptr[data_offset_h_ptr];
            const float offset_w = data_offset_ptr[data_offset_w_ptr];
            const float mask = data_mask_ptr[data_mask_hw_ptr];
            float inv_h = h_in + i * dilation_h + offset_h;
            float inv_w = w_in + j * dilation_w + offset_w;
            if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
            {
                inv_h = inv_w = -2;
            }
            else
            {
                mval += data_col_ptr[col_pos] *
                        dmcn_im2col_bilinear(data_im_ptr + cnt * height * width, width,
                                             height, width, inv_h, inv_w);
            }
            const float weight = dmcn_get_coordinate_weight(
                inv_h, inv_w, height, width, data_im_ptr + cnt * height * width,
                width, bp_dir);
            val += weight * data_col_ptr[col_pos] * mask;
            cnt += 1;
        }
        // KERNEL_ASSIGN(grad_offset[index], offset_req, val);
        grad_offset[index] = val;
        if (offset_c % 2 == 0)
            // KERNEL_ASSIGN(grad_mask[(((b * deformable_group +
            // deformable_group_index) * kernel_h * kernel_w + offset_c / 2) *
            // height_col + h) * width_col + w], mask_req, mval);
            grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h *
                            kernel_w +
                        offset_c / 2) *
                           height_col +
                       h) *
                          width_col +
                      w] = mval;
    }
}

void modulated_deformable_im2col_cuda(
    cudaStream_t stream, const float *data_im, const float *data_offset,
    const float *data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    float *data_col)
{
    // num_axes should be smaller than block size
    const int channel_per_deformable_group = channels / deformable_group;
    const int num_kernels = channels * batch_size * height_col * width_col;
    modulated_deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels),
                                             CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels, data_im, data_offset, data_mask, height_im, width_im,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
        dilation_w, channel_per_deformable_group, batch_size, channels,
        deformable_group, height_col, width_col, data_col);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in modulated_deformable_im2col_cuda: %s\n",
               cudaGetErrorString(err));
    }
}

void modulated_deformable_col2im_cuda(
    cudaStream_t stream, const float *data_col, const float *data_offset,
    const float *data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    float *grad_im)
{

    const int channel_per_deformable_group = channels / deformable_group;
    const int num_kernels =
        channels * kernel_h * kernel_w * batch_size * height_col * width_col;
    modulated_deformable_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels),
                                             CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels, data_col, data_offset, data_mask, channels, height_im,
        width_im, kernel_h, kernel_w, pad_h, pad_h, stride_h, stride_w,
        dilation_h, dilation_w, channel_per_deformable_group, batch_size,
        deformable_group, height_col, width_col, grad_im);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in modulated_deformable_col2im_cuda: %s\n",
               cudaGetErrorString(err));
    }
}

void modulated_deformable_col2im_coord_cuda(
    cudaStream_t stream, const float *data_col, const float *data_im,
    const float *data_offset, const float *data_mask, const int batch_size,
    const int channels, const int height_im, const int width_im,
    const int height_col, const int width_col, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int deformable_group, float *grad_offset, float *grad_mask)
{
    const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h *
                            kernel_w * deformable_group;
    const int channel_per_deformable_group =
        channels * kernel_h * kernel_w / deformable_group;
    modulated_deformable_col2im_coord_gpu_kernel<<<GET_BLOCKS(num_kernels),
                                                   CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels, data_col, data_im, data_offset, data_mask, channels,
        height_im, width_im, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, channel_per_deformable_group, batch_size,
        2 * kernel_h * kernel_w * deformable_group, deformable_group, height_col,
        width_col, grad_offset, grad_mask);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in modulated_deformable_col2im_coord_cuda: %s\n",
               cudaGetErrorString(err));
    }
}

__global__ void createBatchGemmBuffer(
    const float **input_b, float **output_b, float **columns_b,
    const float **ones_b, const float **weight_b, const float **bias_b,
    float *input, float *output, float *columns, float *ones, float *weight,
    float *bias, const int input_stride, const int output_stride,
    const int columns_stride, const int ones_stride, const int num_batches)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_batches)
    {
        input_b[idx] = input + idx * input_stride;
        output_b[idx] = output + idx * output_stride;
        columns_b[idx] = columns + idx * columns_stride;
        ones_b[idx] = ones + idx * ones_stride;
        // share weights and bias within a Mini-Batch
        weight_b[idx] = weight;
        bias_b[idx] = bias;
    }
}

int dcn_v2_cuda_forward(int batch_size, const void *const *inputs,
                        void **outputs, const float *weight, const float *bias,
                        int input_c, int input_h, int input_w,
                        int out_c, int kernel_h, int kernel_w, int stride_h,
                        int stride_w, int pad_h, int pad_w, int dilation_h,
                        int dilation_w, int deformable_group,
                        cublasHandle_t cublas_handle, void *workspace,
                        size_t workspace_size, cudaStream_t stream)
{
    // assume i/o tensors are all float
    const int out_h = (input_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int out_w = (input_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int input_area = input_h * input_w;
    const int kernel_area = kernel_h * kernel_w;
    const int out_area = out_h * out_w;
    const int matrices_size = batch_size * sizeof(float *);
    if (!workspace || workspace_size == 0)
    {
        workspace_size = 0;
        workspace_size += get_size_aligned<float>(batch_size * out_area);                         // ones
        workspace_size += get_size_aligned<float>(batch_size * input_c * kernel_area * out_area); // columns
        for (int i = 0; i < 6; ++i)
        {
            workspace_size += get_size_aligned<float *>(matrices_size);
        }
        return workspace_size;
    }

    const auto *input = static_cast<const float *>(inputs[0]);
    const auto *offset = static_cast<const float *>(inputs[1]);
    const auto *mask = static_cast<const float *>(inputs[2]);

    // output is already allocated by tensorrt runtime
    auto *output = static_cast<float *>(outputs[0]);

    // thrust policy
    auto on_stream = thrust::cuda::par.on(stream);

    auto *ones = get_next_ptr<float>(batch_size * out_area, workspace, workspace_size);
    thrust::fill_n(on_stream, thrust::device_pointer_cast(ones), batch_size * out_area, 1.f);
    auto *columns = get_next_ptr<float>(batch_size * input_c * kernel_area * out_area, workspace, workspace_size);

    const float **input_b = const_cast<const float **>(
        get_next_ptr<float *>(matrices_size, workspace, workspace_size));
    const float **weight_b = const_cast<const float **>(
        get_next_ptr<float *>(matrices_size, workspace, workspace_size));
    const float **bias_b = const_cast<const float **>(
        get_next_ptr<float *>(matrices_size, workspace, workspace_size));

    float **output_b =
        get_next_ptr<float *>(matrices_size, workspace, workspace_size);

    const float **ones_b = const_cast<const float **>(
        get_next_ptr<float *>(matrices_size, workspace, workspace_size));
    float **columns_b =
        get_next_ptr<float *>(matrices_size, workspace, workspace_size);

    const int block = 128;
    const int grid = (batch_size + block - 1) / block;

    createBatchGemmBuffer<<<grid, block, 0, stream>>>(
        input_b, output_b, columns_b, ones_b, weight_b, bias_b,
        const_cast<float *>(input), const_cast<float *>(output), columns, ones,
        const_cast<float *>(weight), const_cast<float *>(bias),
        input_c * input_area, out_c * out_area, input_c * kernel_area * out_area,
        out_area, batch_size);

    auto check_cublas = [](cublasStatus_t r) {
        if (r != CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error("cublas runtime error: " +
                                     std::to_string(static_cast<int>(r)));
        }
    };
    check_cublas(cublasSetStream(cublas_handle, stream));

    auto adjustLdLevel3 = [](cublasOperation_t opa, cublasOperation_t opb,
                             int m, int n, int k, int &lda, int &ldb, int &ldc) {
        bool transa = opa == CUBLAS_OP_T;
        bool transb = opb == CUBLAS_OP_T;
        if (n <= 1)
        {
            ldc = std::max(m, 1);
        }
        if (transa)
        {
            if (m <= 1)
            {
                lda = std::max(k, 1);
            }
        }
        else
        {
            if (k <= 1)
            {
                lda = std::max(m, 1);
            }
        }
        if (transb)
        {
            if (k <= 1)
            {
                ldb = std::max(n, 1);
            }
        }
        else
        {
            if (n <= 1)
            {
                ldb = std::max(k, 1);
            }
        }
    };

    auto sgemm_batched = [&](cublasOperation_t opa, cublasOperation_t opb,
                             int m, int n, int k,
                             float alpha,
                             const float **a, int lda,
                             const float **b, int ldb,
                             float beta,
                             float **c, int ldc) {
        adjustLdLevel3(opa, opb, m, n, k, lda, ldb, ldc);
        check_cublas(cublasSgemmBatched(cublas_handle, opa, opb,
                                        m, n, k,
                                        &alpha,
                                        a, lda,
                                        b, ldb,
                                        &beta,
                                        c, ldc,
                                        batch_size));
    };

    int m_ = out_c;
    int n_ = out_area;
    int k_ = 1;
    sgemm_batched(CUBLAS_OP_T, CUBLAS_OP_N,
                  n_, m_, k_,
                  1.f,
                  ones_b, k_,
                  bias_b, k_,
                  0.f,
                  output_b, n_);

    modulated_deformable_im2col_cuda(
        stream, input, offset, mask, batch_size, input_c, input_h, input_w, out_h,
        out_w, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
        dilation_w, deformable_group, columns);

    int m = out_c;
    int n = out_area;
    int k = input_c * kernel_area;
    sgemm_batched(CUBLAS_OP_N, CUBLAS_OP_N,
                  n, m, k,
                  1.f,
                  const_cast<const float **>(columns_b), n,
                  weight_b, k,
                  1.f,
                  output_b, n);

    return 0;
}
} // namespace

size_t DCNPlugin::getWorkspaceSize(int maxBatchSize) const
{
    // multiple instances should be reentrent, don't use static variable
    auto size = dcn_v2_cuda_forward(
        maxBatchSize, nullptr, nullptr, nullptr, nullptr,
        input_c, input_h, input_w, output_c, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, deformable_group,
        nullptr, nullptr, 0, nullptr);
    return size;
}

int DCNPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs,
                       void *workspace, cudaStream_t stream)
{
    return dcn_v2_cuda_forward(
        batchSize, inputs, outputs, weight_dev, bias_dev,
        input_c, input_h, input_w, output_c, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, deformable_group,
        cublas_handle, workspace, getWorkspaceSize(batchSize), stream);
}
