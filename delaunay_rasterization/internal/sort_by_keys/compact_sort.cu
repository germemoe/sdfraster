#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <limits>
#include <torch/extension.h>

namespace {
using SortResult = std::tuple<at::Tensor, at::Tensor, int64_t>;

SortResult sort_in_place(const at::Tensor& keys, const at::Tensor& values, int tile_bits) {
    TORCH_CHECK(keys.sizes() == values.sizes());
    TORCH_CHECK(keys.dim() == 1 && keys.is_contiguous() && values.is_contiguous());
    TORCH_CHECK(keys.scalar_type() == at::kLong && values.scalar_type() == at::kInt);
    TORCH_CHECK(keys.is_cuda() && keys.device() == values.device());
    TORCH_CHECK(tile_bits >= 0 && tile_bits <= 32);
    TORCH_CHECK(keys.numel() <= std::numeric_limits<int>::max());
    if (keys.numel() == 0) {
        return {keys, values, 0};
    }

    const c10::cuda::CUDAGuard guard(keys.device());
    auto alternate_keys = at::empty_like(keys);
    auto alternate_values = at::empty_like(values);
    cub::DoubleBuffer<int64_t> key_buffers(keys.data_ptr<int64_t>(), alternate_keys.data_ptr<int64_t>());
    cub::DoubleBuffer<int32_t> value_buffers(values.data_ptr<int32_t>(), alternate_values.data_ptr<int32_t>());
    const auto stream = at::cuda::getCurrentCUDAStream(keys.get_device()).stream();
    const int count = static_cast<int>(keys.numel());
    size_t temporary_bytes = 0;

    auto status = cub::DeviceRadixSort::SortPairs(
        nullptr, temporary_bytes, key_buffers, value_buffers, count, 0, 32 + tile_bits, stream);
    TORCH_CHECK(status == cudaSuccess, "CUB SortPairs query failed: ", cudaGetErrorString(status));

    auto workspace = at::empty({static_cast<int64_t>(temporary_bytes)}, keys.options().dtype(at::kByte));
    status = cub::DeviceRadixSort::SortPairs(
        workspace.data_ptr(), temporary_bytes, key_buffers, value_buffers, count, 0, 32 + tile_bits, stream);
    TORCH_CHECK(status == cudaSuccess, "CUB SortPairs failed: ", cudaGetErrorString(status));
    TORCH_CHECK(key_buffers.selector == value_buffers.selector);
    return key_buffers.selector == 0
        ? SortResult{keys, values, static_cast<int64_t>(temporary_bytes)}
        : SortResult{alternate_keys, alternate_values, static_cast<int64_t>(temporary_bytes)};
}

int64_t stock_workspace_bytes(const at::Tensor& keys, const at::Tensor& values, int tile_bits) {
    const c10::cuda::CUDAGuard guard(keys.device());
    size_t bytes = 0;
    const auto stream = at::cuda::getCurrentCUDAStream(keys.get_device()).stream();
    // Null input/output arrays are allowed for the size query. Use distinct
    // tensor addresses below so the query describes the existing out-of-place call.
    const auto status = cub::DeviceRadixSort::SortPairs(
        nullptr, bytes,
        static_cast<const int64_t*>(keys.data_ptr<int64_t>()), static_cast<int64_t*>(nullptr),
        static_cast<const int32_t*>(values.data_ptr<int32_t>()), static_cast<int32_t*>(nullptr),
        static_cast<int>(keys.numel()), 0, 32 + tile_bits, stream);
    TORCH_CHECK(status == cudaSuccess, "CUB stock workspace query failed: ", cudaGetErrorString(status));
    return static_cast<int64_t>(bytes);
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("sort_in_place", &sort_in_place, "Stable packed radix sort, consuming inputs");
    module.def("stock_workspace_bytes", &stock_workspace_bytes, "Query original out-of-place CUB temporary storage");
}
