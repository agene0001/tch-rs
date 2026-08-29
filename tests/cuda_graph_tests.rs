//! CUDA graph capture/replay correctness.
//!
//! The acceptance gate for the whole feature: a graphed training step must be
//! BITWISE identical to the eager step it recorded — same weights after the
//! same number of steps from the same seed. Anything less means the capture
//! contract (stable addresses, RNG handling) is being violated somewhere.
//!
//! Skipped without CUDA: graphs are a CUDA-runtime feature.
//!
//! Uses SGD+momentum, deliberately: libtorch's C++ Adam/AdamW keep their step
//! counter on the CPU and bake the bias-correction terms into the captured
//! kernels as scalars, so replays reuse the capture-step's correction — a
//! CAPTURE-SAFETY property of those optimizers, not of the graph machinery.
//! (Measured here before the swap: max weight diff 0.51 after 8 steps.)
//! Consumers that want Adam-family optimizers inside a graph need a
//! "capturable" implementation with device-tensor state.

use std::sync::Mutex;

use tch::nn::{self, Module, OptimizerConfig};

/// Serializes the tests in this file. They all seed and draw from the GLOBAL
/// CUDA generator (weight init + batch synthesis), and cargo runs test fns in
/// parallel threads — a concurrent test's `manual_seed` landing between this
/// test's seed and its init makes the two arms diverge for reasons that have
/// nothing to do with graphs. (Diagnosed 2026-08-28: solo runs passed 3/3
/// while full-file runs failed probabilistically.) Production is unaffected —
/// trainers seed once per fit and own their generator usage.
static RNG_LOCK: Mutex<()> = Mutex::new(());
use tch::{CudaGraph, Device, Kind, Tensor};

fn build(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(nn::linear(vs / "l1", 16, 32, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(vs / "l2", 32, 1, Default::default()))
}

/// One SGD-family step on persistent buffers; identical code for both arms.
fn step(model: &impl Module, opt: &mut nn::Optimizer, x: &Tensor, y: &Tensor) {
    opt.zero_grad();
    let loss = (model.forward(x) - y).pow_tensor_scalar(2.0).mean(Kind::Float);
    loss.backward();
    opt.step();
}

fn weights_of(vs: &nn::VarStore) -> Vec<(String, Tensor)> {
    let mut v: Vec<(String, Tensor)> = vs
        .variables()
        .into_iter()
        .map(|(k, t)| (k, t.detach().to_device(Device::Cpu)))
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

/// Deterministic per-step batch, written into the persistent buffers.
fn fill_batch(x: &mut Tensor, y: &mut Tensor, step_idx: i64) {
    tch::no_grad(|| {
        tch::manual_seed(1000 + step_idx);
        let device = x.device();
        let xb = Tensor::randn([64, 16], (Kind::Float, device));
        let yb = Tensor::randn([64, 1], (Kind::Float, device));
        x.copy_(&xb);
        y.copy_(&yb);
    });
}

#[test]
fn graphed_steps_are_bitwise_identical_to_eager() {
    let _guard = RNG_LOCK.lock().unwrap();
    if !tch::Cuda::is_available() {
        eprintln!("CUDA unavailable - skipping cuda graph test");
        return;
    }
    let device = Device::Cuda(0);
    const WARMUP: i64 = 3;
    const TOTAL: i64 = 8;

    // ---- arm 1: fully eager --------------------------------------------
    tch::manual_seed(42);
    let vs_e = nn::VarStore::new(device);
    let root_e = vs_e.root();
    let model_e = build(&root_e);
    let mut opt_e = nn::Sgd { momentum: 0.9, ..Default::default() }.build(&vs_e, 1e-2).unwrap();
    let mut x = Tensor::zeros([64, 16], (Kind::Float, device));
    let mut y = Tensor::zeros([64, 1], (Kind::Float, device));
    for i in 0..TOTAL {
        fill_batch(&mut x, &mut y, i);
        step(&model_e, &mut opt_e, &x, &y);
    }
    tch::Cuda::synchronize(0);
    let eager = weights_of(&vs_e);

    // ---- arm 2: warmup eager on the side stream, then capture + replay --
    tch::manual_seed(42);
    let vs_g = nn::VarStore::new(device);
    let root_g = vs_g.root();
    let model_g = build(&root_g);
    let mut opt_g = nn::Sgd { momentum: 0.9, ..Default::default() }.build(&vs_g, 1e-2).unwrap();
    let mut xg = Tensor::zeros([64, 16], (Kind::Float, device));
    let mut yg = Tensor::zeros([64, 1], (Kind::Float, device));

    let mut graph = CudaGraph::new().expect("graph handle");
    graph.side_enter().expect("side stream");
    for i in 0..WARMUP {
        fill_batch(&mut xg, &mut yg, i);
        step(&model_g, &mut opt_g, &xg, &yg);
    }
    // The captured step is RECORDED, not executed: fill the buffers for the
    // step that the FIRST replay will perform, then record.
    fill_batch(&mut xg, &mut yg, WARMUP);
    graph.capture_begin().expect("capture begin");
    step(&model_g, &mut opt_g, &xg, &yg);
    graph.capture_end().expect("capture end");
    graph.side_exit().expect("side exit");

    for i in WARMUP..TOTAL {
        fill_batch(&mut xg, &mut yg, i);
        graph.replay().expect("replay");
    }
    tch::Cuda::synchronize(0);
    let graphed = weights_of(&vs_g);

    // ---- bitwise comparison --------------------------------------------
    assert_eq!(eager.len(), graphed.len());
    for ((name_e, w_e), (name_g, w_g)) in eager.iter().zip(graphed.iter()) {
        assert_eq!(name_e, name_g);
        let equal = bool::try_from(w_e.eq_tensor(w_g).all()).unwrap();
        assert!(
            equal,
            "weight '{name_e}' diverged between eager and graphed training \
             (max abs diff {})",
            f64::try_from((w_e - w_g).abs().max()).unwrap()
        );
    }
}

/// Replay must read FRESH buffer contents each call — if the graph had baked
/// the batch data instead of the buffer address, every replay would repeat the
/// captured batch and weights would evolve differently for different data.
#[test]
fn replay_reads_fresh_buffer_contents() {
    let _guard = RNG_LOCK.lock().unwrap();
    if !tch::Cuda::is_available() {
        eprintln!("CUDA unavailable - skipping cuda graph test");
        return;
    }
    let device = Device::Cuda(0);
    tch::manual_seed(7);
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let model = build(&root);
    let mut opt = nn::Sgd { momentum: 0.9, ..Default::default() }.build(&vs, 1e-2).unwrap();
    let mut x = Tensor::zeros([64, 16], (Kind::Float, device));
    let mut y = Tensor::zeros([64, 1], (Kind::Float, device));

    let mut graph = CudaGraph::new().unwrap();
    graph.side_enter().unwrap();
    fill_batch(&mut x, &mut y, 0);
    for _ in 0..2 {
        step(&model, &mut opt, &x, &y);
    }
    graph.capture_begin().unwrap();
    step(&model, &mut opt, &x, &y);
    graph.capture_end().unwrap();
    graph.side_exit().unwrap();

    // Two replays on DIFFERENT data must produce different weights than two
    // replays on the SAME data would (i.e., the second batch matters).
    let snap = |vs: &nn::VarStore| {
        let v = vs.variables();
        let t = v.get("l1.weight").unwrap();
        f64::try_from(t.detach().sum(Kind::Float)).unwrap()
    };
    fill_batch(&mut x, &mut y, 1);
    graph.replay().unwrap();
    tch::Cuda::synchronize(0);
    let after_b1 = snap(&vs);
    fill_batch(&mut x, &mut y, 2);
    graph.replay().unwrap();
    tch::Cuda::synchronize(0);
    let after_b2 = snap(&vs);
    assert!(
        (after_b1 - after_b2).abs() > 0.0,
        "weights did not change across replays - graph is not executing"
    );
}

/// Diagnostic: per-step weight trace for both arms — localizes WHERE the
/// divergence enters (warmup = harness bug; capture = capture corrupts state;
/// first replay = replay executes different math).
#[test]
fn diag_per_step_divergence_trace() {
    let _guard = RNG_LOCK.lock().unwrap();
    if !tch::Cuda::is_available() {
        return;
    }
    let device = Device::Cuda(0);
    let snap = |vs: &nn::VarStore| -> f64 {
        tch::Cuda::synchronize(0);
        let v = vs.variables();
        let t = v.get("l1.bias").unwrap();
        f64::try_from(t.detach().sum(Kind::Float)).unwrap()
    };

    // eager
    tch::manual_seed(42);
    let vs_e = nn::VarStore::new(device);
    let root_e = vs_e.root();
    let model_e = build(&root_e);
    let mut opt_e = nn::Sgd { momentum: 0.9, ..Default::default() }.build(&vs_e, 1e-2).unwrap();
    let mut x = Tensor::zeros([64, 16], (Kind::Float, device));
    let mut y = Tensor::zeros([64, 1], (Kind::Float, device));
    let mut eager_trace = vec![snap(&vs_e)];
    for i in 0..6 {
        fill_batch(&mut x, &mut y, i);
        step(&model_e, &mut opt_e, &x, &y);
        eager_trace.push(snap(&vs_e));
    }

    // graphed
    tch::manual_seed(42);
    let vs_g = nn::VarStore::new(device);
    let root_g = vs_g.root();
    let model_g = build(&root_g);
    let mut opt_g = nn::Sgd { momentum: 0.9, ..Default::default() }.build(&vs_g, 1e-2).unwrap();
    let mut xg = Tensor::zeros([64, 16], (Kind::Float, device));
    let mut yg = Tensor::zeros([64, 1], (Kind::Float, device));
    let mut graphed_trace = vec![snap(&vs_g)];

    let mut graph = CudaGraph::new().unwrap();
    graph.side_enter().unwrap();
    for i in 0..3 {
        fill_batch(&mut xg, &mut yg, i);
        step(&model_g, &mut opt_g, &xg, &yg);
        graphed_trace.push(snap(&vs_g));
    }
    fill_batch(&mut xg, &mut yg, 3);
    graph.capture_begin().unwrap();
    step(&model_g, &mut opt_g, &xg, &yg);
    graph.capture_end().unwrap();
    graph.side_exit().unwrap();
    eprintln!("after capture (should equal after-step-3): {}", snap(&vs_g));
    for i in 3..6 {
        fill_batch(&mut xg, &mut yg, i);
        graph.replay().unwrap();
        graphed_trace.push(snap(&vs_g));
    }

    eprintln!("step |        eager |      graphed |    diff");
    for (i, (e, g)) in eager_trace.iter().zip(graphed_trace.iter()).enumerate() {
        eprintln!("{i:4} | {e:12.6} | {g:12.6} | {:9.2e}", (e - g).abs());
    }
}
