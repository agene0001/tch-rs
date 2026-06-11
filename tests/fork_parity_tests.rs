//! Regression tests for this fork's accuracy and performance changes:
//! - `scalar / tensor` computed as a single exact division (no `pow(-1)`),
//! - PyTorch-parity layer initializations (linear/conv/batch-norm/RNN),
//! - on-device (sync-free) `clip_grad_norm`,
//! - lazy `Iter2::shuffle`,
//! - fused `cross_entropy_for_logits`,
//! - `Tensor::copy` without the redundant zero-fill,
//! - mutex-free imagenet normalization.
use std::convert::TryFrom;
use tch::nn::{self, Module, OptimizerConfig, RNN};
use tch::{data::Iter2, Device, Kind, Tensor};

#[test]
fn scalar_div_float_tensor_is_exact() {
    let values: Vec<f32> = vec![3.0, -7.5, 0.25, 1e-8, 123456.0, 0.1];
    let t = Tensor::from_slice(&values);
    for s in [2.0f32, -1.0, 0.5, 7.3] {
        let out = Vec::<f32>::try_from(s / &t).unwrap();
        let expected: Vec<f32> = values.iter().map(|&x| s / x).collect();
        // A single f32 division must match the scalar computation bit for bit.
        assert_eq!(out, expected, "scalar {s}");
    }
}

#[test]
fn scalar_div_int_tensor_is_true_division() {
    // The previous pow(-1) formulation panicked on integer tensors.
    let t = Tensor::from_slice(&[1i64, 2, 4, -8]);
    let out = 2i64 / &t;
    assert_eq!(out.kind(), Kind::Float);
    assert_eq!(Vec::<f32>::try_from(out).unwrap(), vec![2.0, 1.0, 0.5, -0.25]);
}

#[test]
fn scalar_div_keeps_tensor_float_kind() {
    let t = Tensor::from_slice(&[2.0f64, 8.0]);
    let out = 1.0f64 / &t;
    assert_eq!(out.kind(), Kind::Double);
    assert_eq!(Vec::<f64>::try_from(out).unwrap(), vec![0.5, 0.125]);
}

#[test]
fn tensor_div_scalar_unchanged() {
    let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    let out = Vec::<f32>::try_from(&t / 2.0).unwrap();
    assert_eq!(out, vec![0.5, 1.0, 1.5]);
}

#[test]
fn clip_grad_norm_matches_reference() {
    let vs = nn::VarStore::new(Device::Cpu);
    let root = vs.root();
    let a = root.var("a", &[4], nn::Init::Const(1.0));
    let b = root.var("b", &[3], nn::Init::Const(-2.0));
    let mut opt = nn::Sgd::default().build(&vs, 0.1).unwrap();

    let run_backward = |opt: &mut nn::Optimizer| {
        opt.zero_grad();
        let loss = (&a * &a).sum(Kind::Float) + (&b * &b).sum(Kind::Float);
        loss.backward();
    };

    // grads: a -> 2.0 (4 elements), b -> -4.0 (3 elements)
    let total_norm = (4.0 * 4.0f64 + 3.0 * 16.0).sqrt();

    // Clipping branch: gradients are rescaled by max / (norm + 1e-6).
    run_backward(&mut opt);
    let max = 1.0;
    opt.clip_grad_norm(max);
    let coef = max / (total_norm + 1e-6);
    for g in Vec::<f32>::try_from(a.grad()).unwrap() {
        assert!((f64::from(g) - 2.0 * coef).abs() < 1e-6, "got {g}");
    }
    for g in Vec::<f32>::try_from(b.grad()).unwrap() {
        assert!((f64::from(g) + 4.0 * coef).abs() < 1e-6, "got {g}");
    }

    // Non-clipping branch: multiplying by the clamped coefficient of exactly
    // 1.0 must leave gradients bit-identical.
    run_backward(&mut opt);
    opt.clip_grad_norm(1e9);
    assert_eq!(Vec::<f32>::try_from(a.grad()).unwrap(), vec![2.0; 4]);
    assert_eq!(Vec::<f32>::try_from(b.grad()).unwrap(), vec![-4.0; 3]);
}

#[test]
fn default_linear_init_matches_pytorch() {
    let vs = nn::VarStore::new(Device::Cpu);
    let linear = nn::linear(vs.root(), 128, 256, Default::default());

    // PyTorch: kaiming_uniform_(a=sqrt(5)) => U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
    let bound = 1.0 / 128f64.sqrt();
    let w_max = f64::try_from(linear.ws.abs().max()).unwrap();
    assert!(w_max <= bound * (1.0 + 1e-5), "weight max {w_max} exceeds bound {bound}");
    // U(-b, b) has std b/sqrt(3); 128*256 samples keep the estimate tight.
    let w_std = f64::try_from(linear.ws.std(true)).unwrap();
    let expected_std = bound / 3f64.sqrt();
    assert!(
        (w_std - expected_std).abs() / expected_std < 0.05,
        "weight std {w_std} vs expected {expected_std}"
    );

    let bs = linear.bs.as_ref().unwrap();
    let b_max = f64::try_from(bs.abs().max()).unwrap();
    assert!(b_max <= bound * (1.0 + 1e-5), "bias max {b_max} exceeds bound {bound}");
    assert!(b_max > 0.0, "bias should not be all zeros");
}

#[test]
fn default_conv_init_matches_pytorch() {
    let vs = nn::VarStore::new(Device::Cpu);
    let conv = nn::conv2d(vs.root(), 8, 16, 3, Default::default());

    let fan_in = 8 * 3 * 3;
    let bound = 1.0 / (fan_in as f64).sqrt();
    let w_max = f64::try_from(conv.ws.abs().max()).unwrap();
    assert!(w_max <= bound * (1.0 + 1e-5), "weight max {w_max} exceeds bound {bound}");

    let bs = conv.bs.as_ref().unwrap();
    let b_max = f64::try_from(bs.abs().max()).unwrap();
    assert!(b_max <= bound * (1.0 + 1e-5), "bias max {b_max} exceeds bound {bound}");
    assert!(b_max > 0.0, "bias should not be all zeros");
}

#[test]
fn default_batch_norm_init_matches_pytorch() {
    let vs = nn::VarStore::new(Device::Cpu);
    let bn = nn::batch_norm2d(vs.root(), 16, Default::default());
    assert_eq!(Vec::<f32>::try_from(bn.ws.as_ref().unwrap()).unwrap(), vec![1.0; 16]);
    assert_eq!(Vec::<f32>::try_from(bn.bs.as_ref().unwrap()).unwrap(), vec![0.0; 16]);
}

#[test]
fn default_lstm_init_matches_pytorch() {
    let vs = nn::VarStore::new(Device::Cpu);
    let _lstm = nn::lstm(vs.root(), 10, 32, Default::default());

    let bound = 1.0 / 32f64.sqrt();
    let variables = vs.variables();
    for name in ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"] {
        let tensor = variables.get(name).unwrap_or_else(|| panic!("missing {name}"));
        let max = f64::try_from(tensor.abs().max()).unwrap();
        assert!(max <= bound * (1.0 + 1e-5), "{name} max {max} exceeds bound {bound}");
        assert!(max > 0.0, "{name} should not be all zeros");
    }
}

#[test]
fn lstm_zero_state_not_aliased() {
    let vs = nn::VarStore::new(Device::Cpu);
    let lstm = nn::lstm(vs.root(), 4, 8, Default::default());
    let nn::LSTMState((mut h, c)) = lstm.zero_state(2);
    let _ = h.fill_(1.0);
    // c must stay zero when h is updated in place.
    assert_eq!(f64::try_from(c.sum(Kind::Float)).unwrap(), 0.0);
    assert_eq!(f64::try_from(h.sum(Kind::Float)).unwrap(), 16.0);
}

#[test]
fn iter2_shuffle_preserves_pairs_and_covers_dataset() {
    let xs = Tensor::arange(100, (Kind::Int64, Device::Cpu)).view([100, 1]);
    let ys = &xs * 2;

    let mut seen = vec![];
    for (bx, by) in Iter2::new(&xs, &ys, 10).shuffle() {
        assert_eq!(bx.size(), [10, 1]);
        let bx = Vec::<i64>::try_from(bx.view(-1)).unwrap();
        let by = Vec::<i64>::try_from(by.view(-1)).unwrap();
        // x/y pairing must survive the shuffle.
        for (x, y) in bx.iter().zip(by.iter()) {
            assert_eq!(*y, 2 * x);
        }
        seen.extend(bx);
    }
    // The shuffle must actually permute (P(identity) = 1/100! in a fair shuffle).
    assert_ne!(seen, (0..100).collect::<Vec<_>>());
    // Every sample must appear exactly once.
    seen.sort_unstable();
    assert_eq!(seen, (0..100).collect::<Vec<_>>());
}

#[test]
fn iter2_without_shuffle_keeps_order() {
    let xs = Tensor::arange(30, (Kind::Int64, Device::Cpu)).view([30, 1]);
    let ys = xs.shallow_clone();
    let mut seen = vec![];
    for (bx, _by) in Iter2::new(&xs, &ys, 10) {
        seen.extend(Vec::<i64>::try_from(bx.view(-1)).unwrap());
    }
    assert_eq!(seen, (0..30).collect::<Vec<_>>());
}

#[test]
fn cross_entropy_matches_log_softmax_nll() {
    tch::manual_seed(42);
    let logits = Tensor::randn([8, 5], (Kind::Float, Device::Cpu));
    let targets = Tensor::from_slice(&[0i64, 1, 2, 3, 4, 0, 1, 2]);
    let fused = f64::try_from(logits.cross_entropy_for_logits(&targets)).unwrap();
    let reference =
        f64::try_from(logits.log_softmax(-1, Kind::Float).nll_loss(&targets)).unwrap();
    assert!((fused - reference).abs() < 1e-6, "{fused} vs {reference}");
}

#[test]
fn cross_entropy_keeps_double_precision() {
    let logits = Tensor::randn([4, 3], (Kind::Double, Device::Cpu));
    let targets = Tensor::from_slice(&[0i64, 1, 2, 0]);
    // The previous implementation downcast Double logits to Float.
    assert_eq!(logits.cross_entropy_for_logits(&targets).kind(), Kind::Double);
}

#[test]
fn copy_is_deep_and_equal() {
    let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    let mut c = t.copy();
    assert_eq!(Vec::<f32>::try_from(&c).unwrap(), vec![1.0, 2.0, 3.0]);
    let _ = c.fill_(0.0);
    assert_eq!(Vec::<f32>::try_from(&t).unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn init_const_uniform_randn_values() {
    // Const now uses a single full() call.
    let t = nn::init(nn::Init::Const(2.5), &[16], Device::Cpu);
    assert_eq!(Vec::<f32>::try_from(&t).unwrap(), vec![2.5; 16]);

    // Uniform now fills uninitialized memory in place: every value must be
    // freshly sampled within bounds.
    let t = nn::init(nn::Init::Uniform { lo: -0.25, up: 0.25 }, &[10_000], Device::Cpu);
    let max = f64::try_from(t.abs().max()).unwrap();
    assert!(max <= 0.25 && max > 0.0);

    // Randn{mean, stdev} now samples normal_(mean, stdev) directly.
    let t = nn::init(nn::Init::Randn { mean: 3.0, stdev: 2.0 }, &[100_000], Device::Cpu);
    let mean = f64::try_from(t.mean(Kind::Float)).unwrap();
    let std = f64::try_from(t.std(true)).unwrap();
    assert!((mean - 3.0).abs() < 0.05, "mean {mean}");
    assert!((std - 2.0).abs() < 0.05, "std {std}");
}

#[test]
fn kaiming_gain_matches_pytorch() {
    // PyTorch: gain = sqrt(2 / (1 + a^2)) with a = sqrt(5) for the default
    // Linear/Conv init, i.e. sqrt(1/3).
    let gain = nn::init::NonLinearity::LeakyReLU(5f64.sqrt()).gain();
    assert!((gain - (1f64 / 3.).sqrt()).abs() < 1e-12);
}

#[test]
fn imagenet_normalize_roundtrip() {
    let values: Vec<u8> = vec![0, 128, 255, 64, 32, 200, 10, 90, 180, 250, 5, 77];
    let img = Tensor::from_slice(&values).view([3, 2, 2]);
    let normalized = tch::vision::imagenet::normalize(&img).unwrap();
    assert_eq!(normalized.kind(), Kind::Float);
    let restored = tch::vision::imagenet::unnormalize(&normalized).unwrap();
    assert_eq!(restored.kind(), Kind::Uint8);
    let restored = Vec::<u8>::try_from(restored.view(-1)).unwrap();
    for (orig, back) in values.iter().zip(restored.iter()) {
        let diff = (i16::from(*orig) - i16::from(*back)).abs();
        assert!(diff <= 1, "roundtrip {orig} -> {back}");
    }
}

/// Print-only timing comparisons; run with `cargo test -- --ignored --nocapture`.
/// These document the speedups rather than asserting on wall-clock time,
/// which would be flaky across machines.
#[test]
#[ignore]
fn bench_shuffle_and_clip() {
    use std::time::Instant;

    // Lazy shuffle vs materializing the whole dataset up front.
    let n = 200_000i64;
    let xs = Tensor::randn([n, 64], (Kind::Float, Device::Cpu));
    let ys = Tensor::randn([n, 1], (Kind::Float, Device::Cpu));

    let start = Instant::now();
    let mut batches = 0;
    for (_bx, _by) in Iter2::new(&xs, &ys, 256).shuffle() {
        batches += 1;
    }
    println!("lazy shuffle epoch ({batches} batches): {:?}", start.elapsed());

    let start = Instant::now();
    let index = Tensor::randperm(n, (Kind::Int64, Device::Cpu));
    let xs_mat = xs.index_select(0, &index);
    let ys_mat = ys.index_select(0, &index);
    let mut batches = 0;
    for (_bx, _by) in Iter2::new(&xs_mat, &ys_mat, 256) {
        batches += 1;
    }
    println!("materialized shuffle epoch ({batches} batches): {:?}", start.elapsed());

    // Same comparison with real per-batch work: one epoch of training an MLP.
    // The lazy shuffle costs a fixed per-batch gather, so its relative
    // overhead shrinks as per-batch compute grows (vary `hidden` to see it).
    let train_epoch = |xs: &Tensor, ys: &Tensor, hidden: i64, lazy: bool| {
        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let net = nn::seq()
            .add(nn::linear(&root / "l1", 64, hidden, Default::default()))
            .add_fn(Tensor::relu)
            .add(nn::linear(&root / "l2", hidden, 1, Default::default()));
        let mut opt = nn::Sgd::default().build(&vs, 1e-3).unwrap();
        let start = Instant::now();
        let (xs_mat, ys_mat);
        let mut iter = if lazy {
            let mut it = Iter2::new(xs, ys, 256);
            it.shuffle();
            it
        } else {
            let index = Tensor::randperm(xs.size()[0], (Kind::Int64, Device::Cpu));
            xs_mat = xs.index_select(0, &index);
            ys_mat = ys.index_select(0, &index);
            Iter2::new(&xs_mat, &ys_mat, 256)
        };
        for (bx, by) in &mut iter {
            let loss = net.forward(&bx).mse_loss(&by, tch::Reduction::Mean);
            opt.backward_step(&loss);
        }
        start.elapsed()
    };
    for hidden in [128, 1024] {
        let lazy = train_epoch(&xs, &ys, hidden, true);
        let mat = train_epoch(&xs, &ys, hidden, false);
        println!("training epoch (hidden={hidden}), lazy shuffle: {lazy:?}, materialized: {mat:?}");
    }

    // clip_grad_norm throughput (the win is removing the device sync, which
    // shows up on CUDA rather than CPU; this is a smoke test).
    let vs = nn::VarStore::new(Device::Cpu);
    let root = vs.root();
    let vars: Vec<_> =
        (0..16).map(|i| root.var(&format!("v{i}"), &[256, 256], nn::Init::Const(1.0))).collect();
    let mut opt = nn::Sgd::default().build(&vs, 0.1).unwrap();
    let start = Instant::now();
    for _ in 0..50 {
        opt.zero_grad();
        let loss = vars.iter().map(|v| (v * v).sum(Kind::Float)).sum::<Tensor>();
        loss.backward();
        opt.clip_grad_norm(1.0);
    }
    println!("50 backward+clip_grad_norm steps: {:?}", start.elapsed());
}
