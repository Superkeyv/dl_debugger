#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <unordered_set>

#include <torch/extension.h>

std::unordered_set<void*> _global_memory_record;

extern "C" {
    void* ds_malloc(ssize_t size, int device, cudaStream_t stream) {
        void *ptr;
        cudaMalloc(&ptr, size);
        _global_memory_record.insert(ptr);
        return ptr;
    }

    void* ds_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
        cudaFree(ptr);
        auto iter = _global_memory_record.find(ptr);
        if (iter != _global_memory_record.end())
            _global_memory_record.erase(iter);
    }

    void free_all_mem() {
        for (auto ptr : _global_memory_record) {
            cudaFree(ptr);
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("free_all_mem", &free_all_mem, "release all mem allocted by torch.");
}