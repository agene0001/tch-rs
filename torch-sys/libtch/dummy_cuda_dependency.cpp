// Restored after commit 16e8f59 ("Attempt at removing the cuda hack.") so
// torch_cuda.dll's import survives MSVC's linker on Windows. Without a static
// reference to *any* CUDA symbol the linker drops `torch_cuda.lib` (the .lib
// supplied by libtorch is an import lib with no statically-required symbols
// from our point of view), the OS loader never brings `torch_cuda.dll` in at
// startup, its DllMain never runs, and `at::globalContext().hasCUDA()` stays
// false at runtime — making `tch::Cuda::is_available()` always return false.
//
// The anchor symbol must be a *non-inline* exported function so its mangled
// name is a real DLL export rather than an artifact of MSVC exporting
// dllexport-marked inline functions instantiated inside the torch build.
// `at::cuda::warp_size` is a plain TORCH_CUDA_CPP_API export with a stable
// signature. (Earlier revisions anchored on `CachingHostAllocator_emptyCache`,
// deprecated upstream, and then `getCurrentCUDABlasHandle`, which grew a
// defaulted `bool` parameter in libtorch 2.13 and so changed its mangled
// name.)
//
// Note this function is never called at runtime: the Rust side only takes its
// address in a `#[used]` static (src/tensor/mod.rs) / a black_box reference,
// which is all the linker needs.
#include <stdio.h>
#include <stdint.h>
#include <stdexcept>
#include <iostream>

extern "C" {
    void dummy_cuda_dependency();
}

namespace at {
namespace cuda {
int warp_size();
} // namespace cuda
} // namespace at

void dummy_cuda_dependency() {
    try {
        at::cuda::warp_size();
    } catch (std::exception& e) {
        if (getenv("TCH_PRINT_CUDA_INIT_ERROR") != nullptr) {
            std::cerr << "error initializing cuda: " << e.what() << std::endl;
        }
    }
}
