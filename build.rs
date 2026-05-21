fn main() {
    let os = std::env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    match os.as_str() {
        "linux" | "windows" => {
            if let Some(lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
                println!("cargo:rustc-link-arg=-Wl,-rpath={}", lib_path.to_string_lossy());
            }
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            println!("cargo:rustc-link-arg=-ltorch");
        }
        _ => {}
    }
    // Propagate the CUDA-presence cfg set by `torch-sys/build.rs` (via the
    // `links = "tch"` metadata channel) so we can gate the link anchor that
    // forces MSVC to keep the `torch_cuda.lib` import.
    println!("cargo:rustc-check-cfg=cfg(use_cuda)");
    if std::env::var_os("DEP_TCH_CUDA").is_some() {
        println!("cargo:rustc-cfg=use_cuda");
    }
}
