//! CUDA graph capture and replay (`at::cuda::CUDAGraph`).
//!
//! A captured graph records the full DAG of kernel launches from one training
//! step and replays it with a single API call â€” eliminating the per-kernel CPU
//! dispatch cost that dominates launch-bound workloads (many tiny kernels).
//! Replay executes the IDENTICAL kernels with identical arguments on fixed
//! memory addresses: given the same inputs and RNG state it is bitwise
//! equivalent to eager execution. libtorch captures the CUDA RNG generator and
//! advances its Philox offset per replay, so dropout draws fresh masks.
//!
//! # Capture contract (CUDA requirements, not tch inventions)
//!
//! * Everything the graph touches must live at stable addresses: inputs come
//!   from persistent buffers the caller `copy_`s into before each [`CudaGraph::replay`].
//! * No host synchronization inside the captured region.
//! * Capture happens on a dedicated side stream â€” [`CudaGraph::side_enter`] /
//!   [`CudaGraph::side_exit`] manage that; run a few eager warmup steps inside
//!   the side-stream scope first so lazily-created state (optimizer moments,
//!   cuDNN autotune results) exists at stable addresses before recording.
//! * The recorded step is NOT executed during capture â€” the first
//!   [`CudaGraph::replay`] performs it.
//! * Scalar arguments (a learning rate, a dropout probability) are BAKED into
//!   the recording; changing them afterwards requires re-capturing.
//! * libtorch's C++ Adam/AdamW are NOT capture-safe: their step counter lives
//!   on the CPU and the bias-correction terms are baked as scalars, so
//!   replays silently reuse the capture step's correction (measured: weights
//!   diverge from eager within a handful of steps). Use SGD, or a
//!   "capturable" Adam implementation whose state (step count, lr) lives in
//!   device tensors.
//!
//! ```no_run
//! # use tch::CudaGraph;
//! # fn step() {}
//! let mut graph = CudaGraph::new().unwrap();
//! graph.side_enter().unwrap();
//! for _ in 0..3 { step(); }            // warmup, eager
//! graph.capture_begin().unwrap();
//! step();                              // recorded, not executed
//! graph.capture_end().unwrap();
//! graph.side_exit().unwrap();
//! loop {
//!     // copy fresh batch data into the persistent buffers, then:
//!     graph.replay().unwrap();
//! # break;
//! }
//! ```

use crate::TchError;

pub struct CudaGraph {
    c_graph: *mut torch_sys::C_cuda_graph,
}

unsafe impl Send for CudaGraph {}

impl std::fmt::Debug for CudaGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "cuda_graph")
    }
}

impl CudaGraph {
    /// Create an empty graph handle (allocates the capture side stream).
    /// Fails with a descriptive error on a libtorch built without CUDA.
    pub fn new() -> Result<CudaGraph, TchError> {
        let c_graph = unsafe_torch_err!(torch_sys::atcg_new());
        if c_graph.is_null() {
            return Err(TchError::Torch("atcg_new returned null".to_string()));
        }
        Ok(CudaGraph { c_graph })
    }

    /// Switch the current thread to the capture side stream (synchronizing the
    /// displaced stream first). Warmup steps and the capture itself must run
    /// inside this scope.
    pub fn side_enter(&mut self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::atcg_side_enter(self.c_graph));
        Ok(())
    }

    /// Leave the side stream (synchronizing it) and restore the previous one.
    pub fn side_exit(&mut self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::atcg_side_exit(self.c_graph));
        Ok(())
    }

    /// Start recording. Subsequent CUDA work on this thread is captured into
    /// the graph instead of executing, until [`Self::capture_end`].
    pub fn capture_begin(&mut self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::atcg_capture_begin(self.c_graph));
        Ok(())
    }

    /// Stop recording.
    pub fn capture_end(&mut self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::atcg_capture_end(self.c_graph));
        Ok(())
    }

    /// Execute the recorded step on the current stream.
    pub fn replay(&self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::atcg_replay(self.c_graph));
        Ok(())
    }

    /// Free and total device memory in bytes. Lets a caller verify that
    /// destroying graphs actually returns their private pools.
    pub fn mem_info() -> Result<(i64, i64), TchError> {
        let (mut free, mut total) = (0i64, 0i64);
        unsafe_torch_err!(torch_sys::atcg_mem_info(&mut free, &mut total));
        Ok((free, total))
    }

    /// Execute the recorded step on this graph's own side stream,
    /// event-bracketed against the caller's current stream: pending
    /// current-stream work (buffer fills) happens-before the replay, and the
    /// replay happens-before subsequent current-stream work (loss reads).
    /// Semantically identical to [`Self::replay`] for a single caller; the
    /// point is that DIFFERENT graphs replay on DIFFERENT streams, letting
    /// concurrent workers' small-kernel graphs overlap on the GPU.
    pub fn replay_concurrent(&self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::atcg_replay_concurrent(self.c_graph));
        Ok(())
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe { torch_sys::atcg_free(self.c_graph) }
    }
}
