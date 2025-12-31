#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include <ATen/cuda/CUDAContext.h>   // ✅ getCurrentCUDAStream
#include <vector>

namespace extension_cpp {

std::tuple<torch::Tensor, torch::Tensor>
sort_by_keys(const at::Tensor keys,
             const at::Tensor values,
             const int highest_tile_id_msb)  // 你传 bit_length（位数）
{
    TORCH_CHECK(keys.sizes() == values.sizes());
    TORCH_CHECK(keys.dtype() == torch::kLong);
    TORCH_CHECK(values.dtype() == torch::kInt32);
    TORCH_CHECK(keys.is_cuda() && values.is_cuda());


    at::Tensor keys_sorted   = torch::empty_like(keys);
    at::Tensor values_sorted = torch::empty_like(values);

    at::Tensor keys_contig   = keys.contiguous();
    at::Tensor values_contig = values.contiguous();

    const int64_t* keys_ptr   = keys_contig.data_ptr<int64_t>();
    const int32_t* vals_ptr   = values_contig.data_ptr<int32_t>();
    int64_t* keys_out_ptr     = keys_sorted.data_ptr<int64_t>();
    int32_t* vals_out_ptr     = values_sorted.data_ptr<int32_t>();

    const int num_items = (int)keys_contig.numel();

    // ✅ 你的 key = (tile_id << 32) | depth_key(32bit)
    // tile_id 占 highest_tile_id_bits 位
    int begin_bit = 0;
    int end_bit   = 32 + highest_tile_id_msb;  // 注意：bit_length 是“位数”，这里通常不需要 +1
    if (end_bit > 64) end_bit = 64;

    // ✅ 关键：用 PyTorch 当前 stream（不是 stream 0）
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    void*  d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // ✅ 关键：query temp bytes 时也要传同样的 begin/end + stream
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        keys_ptr, keys_out_ptr,
        vals_ptr, vals_out_ptr,
        num_items, begin_bit, end_bit,
        stream);

    cudaError_t err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    TORCH_CHECK(err == cudaSuccess, "cudaMalloc failed: ", cudaGetErrorString(err));

    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        keys_ptr, keys_out_ptr,
        vals_ptr, vals_out_ptr,
        num_items, begin_bit, end_bit,
        stream);

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUB SortPairs failed: ", cudaGetErrorString(err));

    cudaFree(d_temp_storage);

    return {keys_sorted, values_sorted};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sort_by_keys", &sort_by_keys);
}

} // namespace extension_cpp
