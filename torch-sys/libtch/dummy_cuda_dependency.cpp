// Restored after commit 16e8f59 ("Attempt at removing the cuda hack.") so
// torch_cuda.dll's import survives MSVC's linker on Windows. Without a static
// reference to *any* CUDA symbol the linker drops `torch_cuda.lib` (the .lib
// supplied by libtorch is an import lib with no statically-required symbols
// from our point of view), the OS loader never brings `torch_cuda.dll` in at
// startup, its DllMain never runs, and `at::globalContext().hasCUDA()` stays
// false at runtime — making `tch::Cuda::is_available()` always return false.
//
// We reference low-level CUDA symbols here rather than `c10::cuda::*` because
// they're stable across libtorch minor versions and live in libraries the
// linker is going to pull in anyway when CUDA is present (cublas + the CUDA
// warp helpers in torch_cuda.dll).
#include <stdio.h>
#include <stdint.h>
#include <stdexcept>
#include <iostream>

extern "C" {
    void dummy_cuda_dependency();
}

namespace at {
namespace cuda {
// Stable since PyTorch 2.3 — the original `getCurrentCUDABlasHandle` /
// `warp_size` symbols used in the pre-16e8f59 stub were renamed/removed
// in later libtorch releases, breaking the mangled-name match. The host
// allocator API has held its export across recent libtorch versions and is
// cheap to invoke (no-op when the allocator hasn't been used).
void CachingHostAllocator_emptyCache();
} // namespace cuda
} // namespace at

void dummy_cuda_dependency() {
    try {
        at::cuda::CachingHostAllocator_emptyCache();
    } catch (std::exception& e) {
        if (getenv("TCH_PRINT_CUDA_INIT_ERROR") != nullptr) {
            std::cerr << "error initializing cuda: " << e.what() << std::endl;
        }
    }
}
