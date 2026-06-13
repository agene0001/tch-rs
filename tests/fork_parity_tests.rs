//! Regression tests for this fork's accuracy and performance changes:
//! - `scalar / tensor` computed as a single exact division (no `pow(-1)`),
//! - PyTorch-parity layer initializations (linear/conv/batch-norm/RNN),
//! - on-device (sync-free) `clip_grad_norm`,
//! - lazy `Iter2::shuffle`,
//! - fused `cross_entropy_for_logits`,
//! - `Tensor::copy` without the redundant zero-fill,
//! - mutex-free imagenet normalization,
//! - `Entry::or_kaiming_uniform` actually sampling a uniform distribution,
//! - `Entry::or_{zeros,ones}_no_train` not registering trainable variables,
//! - `Init::Orthogonal` preserving the requested shape for rank > 2,
//! - embedding `padding_idx` rows zeroed at init,
//! - vectorized `random_flip` / inclusive-offset `random_crop`,
//! - bulk reads in `Tensor::iter` and stack-allocated `sizeN()`.
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

#[test]
fn or_kaiming_uniform_is_uniform() {
    let vs = nn::VarStore::new(Device::Cpu);
    let w = vs.root().entry("w").or_kaiming_uniform(&[256, 128]);
    assert_eq!(w.size(), [256, 128]);
    // Kaiming uniform with a=sqrt(5) over fan_in=128 is U(-b, b) with
    // b = 1/sqrt(128). The kaiming *normal* distribution this used to sample
    // has std = sqrt(2/128) and would exceed the bound almost surely over
    // 32768 samples.
    let bound = 1.0 / 128f64.sqrt();
    let max = f64::try_from(w.abs().max()).unwrap();
    assert!(max <= bound * (1.0 + 1e-5), "max {max} exceeds uniform bound {bound}");
    assert!(max > 0.0);
}

#[test]
fn no_train_entries_are_not_trainable() {
    let vs = nn::VarStore::new(Device::Cpu);
    let root = vs.root();
    let z = root.entry("z").or_zeros_no_train(&[4]);
    let o = root.entry("o").or_ones_no_train(&[4]);
    assert!(!z.requires_grad(), "or_zeros_no_train must not require grad");
    assert!(!o.requires_grad(), "or_ones_no_train must not require grad");
    assert!(vs.trainable_variables().is_empty());
    // The values themselves are still as requested.
    assert_eq!(Vec::<f32>::try_from(&z).unwrap(), vec![0.0; 4]);
    assert_eq!(Vec::<f32>::try_from(&o).unwrap(), vec![1.0; 4]);
}

#[test]
fn orthogonal_init_keeps_shape_and_orthogonality() {
    let vs = nn::VarStore::new(Device::Cpu);

    // Conv-style rank-4 weight: the variable must keep its 4d shape (it used
    // to come back flattened to [16, 36]).
    let w = vs.root().orthogonal("w", &[16, 4, 3, 3], 1.0);
    assert_eq!(w.size(), [16, 4, 3, 3]);
    // rows(16) <= cols(36): the flattened rows are orthonormal.
    let flat = w.view([16, 36]);
    let prod = flat.matmul(&flat.tr());
    let eye = Tensor::eye(16, (Kind::Float, Device::Cpu));
    assert!(prod.allclose(&eye, 1e-4, 1e-5, false), "rows are not orthonormal");

    // Tall matrix: columns are orthonormal, and the gain scales them.
    let t = vs.root().orthogonal("t", &[8, 4], 2.0);
    assert_eq!(t.size(), [8, 4]);
    let prod = t.tr().matmul(&t);
    let expected = Tensor::eye(4, (Kind::Float, Device::Cpu)) * 4.0;
    assert!(prod.allclose(&expected, 1e-4, 1e-5, false), "gain^2 * I expected");

    // Re-initializing an existing (grad-tracked) variable in place.
    let mut v = vs.root().var("v", &[8, 4], nn::Init::Const(0.0));
    v.init(nn::Init::Orthogonal { gain: 1.0 });
    assert_eq!(v.size(), [8, 4]);
    let prod = v.tr().matmul(&v);
    let eye = Tensor::eye(4, (Kind::Float, Device::Cpu));
    assert!(prod.allclose(&eye, 1e-4, 1e-5, false), "in-place orthogonal re-init");
}

#[test]
fn embedding_padding_row_is_zero() {
    let vs = nn::VarStore::new(Device::Cpu);
    let config = nn::EmbeddingConfig { padding_idx: 2, ..Default::default() };
    let emb = nn::embedding(vs.root(), 5, 8, config);

    // PyTorch zeroes weight[padding_idx] after init.
    assert_eq!(f64::try_from(emb.ws.get(2).abs().sum(Kind::Float)).unwrap(), 0.0);
    // The other rows keep their N(0, 1) init.
    assert!(f64::try_from(emb.ws.abs().sum(Kind::Float)).unwrap() > 0.0);

    // Padding tokens embed to exact zero vectors.
    let out = emb.forward(&Tensor::from_slice(&[2i64, 0]));
    assert_eq!(f64::try_from(out.get(0).abs().sum(Kind::Float)).unwrap(), 0.0);
    assert!(f64::try_from(out.get(1).abs().sum(Kind::Float)).unwrap() > 0.0);

    // The default config (padding_idx = -1) means no padding handling: no row
    // is zeroed, in particular not the last one.
    let emb = nn::embedding(vs.root(), 5, 8, Default::default());
    assert!(f64::try_from(emb.ws.get(4).abs().sum(Kind::Float)).unwrap() > 0.0);
}

#[test]
fn rnn_batch_first_false_layouts_agree() {
    let cfg_sf = nn::RNNConfig { batch_first: false, ..Default::default() };

    // Copy the weights across rather than re-seeding the global RNG: tests
    // run in parallel threads, so two seed+init sequences are not guaranteed
    // to consume the same random draws.
    let vs1 = nn::VarStore::new(Device::Cpu);
    let lstm_bf = nn::lstm(vs1.root(), 4, 6, Default::default());
    let mut vs2 = nn::VarStore::new(Device::Cpu);
    let lstm_sf = nn::lstm(vs2.root(), 4, 6, cfg_sf);
    vs2.copy(&vs1).unwrap();

    let input = Tensor::randn([3, 5, 4], (Kind::Float, Device::Cpu)); // [batch, seq, feat]
    let (out_bf, nn::LSTMState((h_bf, c_bf))) = lstm_bf.seq(&input);
    // seq() used to size the zero state from dim 0 even when the layout is
    // [seq, batch, features], which made this error out (or silently mix the
    // axes for square inputs).
    let (out_sf, nn::LSTMState((h_sf, c_sf))) = lstm_sf.seq(&input.transpose(0, 1).contiguous());
    assert_eq!(out_sf.size(), [5, 3, 6]);
    assert!(out_bf.transpose(0, 1).allclose(&out_sf, 1e-5, 1e-7, false));
    assert!(h_bf.allclose(&h_sf, 1e-5, 1e-7, false));
    assert!(c_bf.allclose(&c_sf, 1e-5, 1e-7, false));

    // step() must insert the singleton sequence axis, keeping the batch on
    // the layout's batch axis.
    let step_in = Tensor::randn([3, 4], (Kind::Float, Device::Cpu));
    let nn::LSTMState((h, c)) = lstm_sf.step(&step_in, &lstm_sf.zero_state(3));
    assert_eq!(h.size(), [1, 3, 6]);
    assert_eq!(c.size(), [1, 3, 6]);

    // Same checks for the GRU layer.
    let vs3 = nn::VarStore::new(Device::Cpu);
    let gru_bf = nn::gru(vs3.root(), 4, 6, Default::default());
    let mut vs4 = nn::VarStore::new(Device::Cpu);
    let gru_sf = nn::gru(vs4.root(), 4, 6, cfg_sf);
    vs4.copy(&vs3).unwrap();
    let (gout_bf, nn::GRUState(gh_bf)) = gru_bf.seq(&input);
    let (gout_sf, nn::GRUState(gh_sf)) = gru_sf.seq(&input.transpose(0, 1).contiguous());
    assert!(gout_bf.transpose(0, 1).allclose(&gout_sf, 1e-5, 1e-7, false));
    assert!(gh_bf.allclose(&gh_sf, 1e-5, 1e-7, false));
    let nn::GRUState(gh) = gru_sf.step(&step_in, &gru_sf.zero_state(3));
    assert_eq!(gh.size(), [1, 3, 6]);
}

#[test]
fn random_flip_is_seeded_and_per_sample() {
    let t = Tensor::arange(4 * 3 * 5 * 5, (Kind::Float, Device::Cpu)).view([4, 3, 5, 5]);

    // Each output sample must be either the original or its horizontal flip.
    let out = tch::vision::dataset::random_flip(&t);
    assert_eq!(out.size(), [4, 3, 5, 5]);
    for i in 0..4 {
        let sample = out.get(i);
        let orig = t.get(i);
        let flipped = orig.flip([2]);
        assert!(sample == orig || sample == flipped, "sample {i} is neither");
    }

    // The flip mask now comes from the torch RNG, so it is reproducible
    // under manual_seed. Tests in this binary run in parallel and share the
    // global RNG, so allow a few attempts in case another test draws random
    // values between the two seedings.
    let deterministic = (0..3).any(|_| {
        tch::manual_seed(7);
        let a = tch::vision::dataset::random_flip(&t);
        tch::manual_seed(7);
        let b = tch::vision::dataset::random_flip(&t);
        a == b
    });
    assert!(deterministic, "same seed must give the same flips");

    // Over many samples both outcomes occur.
    let big = Tensor::arange(64 * 4, (Kind::Float, Device::Cpu)).view([64, 1, 2, 2]);
    let out = tch::vision::dataset::random_flip(&big);
    let flipped_count = i64::try_from(
        out.eq_tensor(&big.flip([3])).all_dims([1, 2, 3].as_slice(), false).sum(Kind::Int64),
    )
    .unwrap();
    assert!(flipped_count > 0 && flipped_count < 64, "got {flipped_count} flips out of 64");
}

#[test]
fn random_crop_samples_all_offsets() {
    // 3x3 image with distinct values, pad=1: the padded image is 5x5 and the
    // 9 crop offsets (0..=2)^2 each produce a distinct window. The previous
    // exclusive bound could only ever produce 4 of them.
    let t = Tensor::arange(9, (Kind::Float, Device::Cpu)).view([1, 1, 3, 3]);
    let mut seen = std::collections::HashSet::new();
    for _ in 0..300 {
        let crop = tch::vision::dataset::random_crop(&t, 1);
        assert_eq!(crop.size(), [1, 1, 3, 3]);
        let values: Vec<i64> =
            Vec::<f32>::try_from(crop.view(-1)).unwrap().iter().map(|&v| v as i64).collect();
        seen.insert(values);
    }
    assert_eq!(seen.len(), 9, "all 9 crop offsets should be reachable");
}

#[test]
fn random_crop_pad_zero_is_identity() {
    // pad=0 used to panic on an empty sampling range.
    let t = Tensor::arange(2 * 3 * 4 * 4, (Kind::Float, Device::Cpu)).view([2, 3, 4, 4]);
    let out = tch::vision::dataset::random_crop(&t, 0);
    assert!(out == t);
}

#[test]
fn random_cutout_zeroes_one_square() {
    let t = Tensor::ones([2, 3, 8, 8], (Kind::Float, Device::Cpu));
    let out = tch::vision::dataset::random_cutout(&t, 3);
    assert_eq!(out.size(), [2, 3, 8, 8]);
    // Exactly one 3x3 square is zeroed across all channels of each sample.
    let expected_sum = (2 * 3 * 8 * 8 - 2 * 3 * 9) as f64;
    assert_eq!(f64::try_from(out.sum(Kind::Float)).unwrap(), expected_sum);
    // The input is untouched.
    assert_eq!(f64::try_from(t.sum(Kind::Float)).unwrap(), (2 * 3 * 8 * 8) as f64);
}

#[test]
fn tensor_iter_bulk_read_matches() {
    let t = Tensor::from_slice(&[1.5f32, -2.5, 3.0]);
    // Reads through a kind conversion, matching the old per-element casts.
    let v: Vec<f64> = t.iter::<f64>().unwrap().collect();
    assert_eq!(v, vec![1.5, -2.5, 3.0]);
    let it = t.iter::<i64>().unwrap();
    assert_eq!(it.len(), 3);
    assert_eq!(it.collect::<Vec<_>>(), vec![1, -2, 3]);
    // Multi-dimensional tensors still error out.
    assert!(t.view([3, 1]).iter::<i64>().is_err());
}

#[test]
fn size_n_accessors() {
    let t = Tensor::from_slice(&[1f32, 2.0, 3.0, 4.0]).view([2, 2]);
    assert_eq!(t.size2().unwrap(), (2, 2));
    assert!(t.size1().is_err());
    let err = t.size3().unwrap_err().to_string();
    assert!(err.contains("three dims") && err.contains("[2, 2]"), "unexpected message: {err}");
}

#[test]
fn no_grad_restores_after_panic() {
    let result = std::panic::catch_unwind(|| tch::no_grad(|| panic!("boom")));
    assert!(result.is_err());
    // Gradient tracking must be re-enabled once the panic has been caught
    // (grad mode is thread local, so parallel tests do not interfere).
    let t = Tensor::from_slice(&[1f32]).set_requires_grad(true);
    let y = &t * &t;
    assert!(y.requires_grad(), "grad mode must be restored after a panic in no_grad");
}

#[test]
fn from_slice2_matches_rows() {
    let t = Tensor::from_slice2(&[[1i64, 2, 3], [4, 5, 6]]);
    assert_eq!(t.size(), [2, 3]);
    assert_eq!(Vec::<i64>::try_from(t.view(-1)).unwrap(), vec![1, 2, 3, 4, 5, 6]);
    let ragged = std::panic::catch_unwind(|| Tensor::from_slice2(&[vec![1f32, 2.0], vec![3.0]]));
    assert!(ragged.is_err(), "ragged rows must panic");
}

#[test]
fn var_store_read_safetensors_trainable() {
    // read_safetensors/fill_safetensors used to fail on trainable variables:
    // the copy was not wrapped in no_grad, which autograd rejects for
    // requires-grad leaves.
    let dir = std::env::temp_dir().join(format!("tch_st_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vars.safetensors");

    let src = nn::VarStore::new(Device::Cpu);
    let _w = src.root().var("w", &[4, 3], nn::Init::Uniform { lo: -1.0, up: 1.0 });
    src.save(&path).unwrap();

    let dst = nn::VarStore::new(Device::Cpu);
    let w2 = dst.root().var("w", &[4, 3], nn::Init::Const(0.0));
    dst.read_safetensors(&path).unwrap();
    assert!(w2.requires_grad());
    assert!(f64::try_from(w2.abs().sum(Kind::Float)).unwrap() > 0.0);

    let dst2 = nn::VarStore::new(Device::Cpu);
    let w3 = dst2.root().var("w", &[4, 3], nn::Init::Const(0.0));
    dst2.fill_safetensors(&path).unwrap();
    assert!(f64::try_from(w3.abs().sum(Kind::Float)).unwrap() > 0.0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn from_data_size_roundtrip() {
    // from_slice/from_data_size now go through torch::empty + memcpy; every
    // element must still arrive intact.
    let bytes: Vec<u8> = (0..24).collect();
    let t = Tensor::from_data_size(&bytes, &[2, 3, 4], Kind::Uint8);
    assert_eq!(t.size(), [2, 3, 4]);
    assert_eq!(Vec::<u8>::try_from(t.view(-1)).unwrap(), bytes);

    let floats: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5 - 250.0).collect();
    let t = Tensor::from_slice(&floats);
    assert_eq!(Vec::<f32>::try_from(&t).unwrap(), floats);
}

/// Print-only timing comparisons for this round of optimizations; run with
/// `cargo test -- --ignored --nocapture`. The `from_slice` number measures the
/// `torch::empty` change in torch_api.cpp: rebuild with the previous
/// `torch::zeros` version to compare baselines.
#[test]
#[ignore]
fn bench_fork_optimizations() {
    use std::hint::black_box;
    use std::time::Instant;
    use tch::IndexOp;

    // Tensor creation from host data (64MB of f32 per call).
    let data: Vec<f32> = vec![1.0; 16 * 1024 * 1024];
    let start = Instant::now();
    for _ in 0..20 {
        let _ = black_box(Tensor::from_slice(&data));
    }
    println!("from_slice 64MB x20:           {:?}", start.elapsed());

    // Vectorized random_flip vs the previous per-sample loop, same workload.
    let imgs = Tensor::randn([128, 3, 224, 224], (Kind::Float, Device::Cpu));
    let start = Instant::now();
    for _ in 0..5 {
        let _ = black_box(tch::vision::dataset::random_flip(&imgs));
    }
    println!("random_flip vectorized x5:     {:?}", start.elapsed());

    let start = Instant::now();
    for _ in 0..5 {
        let size = imgs.size();
        let output = imgs.zeros_like();
        for i in 0..size[0] {
            let mut output_view = output.i(i);
            let view = imgs.i(i);
            let src = if rand::random() { view } else { view.flip([2]) };
            output_view.copy_(&src);
        }
        black_box(&output);
    }
    println!("random_flip per-sample x5:     {:?}", start.elapsed());

    // Scalar-op overhead: each `tensor op scalar` creates and frees a
    // heap-allocated torch::Scalar through the FFI.
    let t = Tensor::from_slice(&[1.0f32]);
    let start = Instant::now();
    for _ in 0..200_000 {
        let _ = black_box(&t + 1.0);
    }
    println!("scalar add x200k:              {:?}", start.elapsed());

    // Text-style batching: gather every window in one unfold+index_select vs
    // one narrow per sample plus a stack.
    let data = Tensor::arange(1_000_000, (Kind::Int64, Device::Cpu));
    let (seq_len, batch_size, nbatches) = (256i64, 64i64, 50i64);
    let indexes = Tensor::randperm(1_000_000 - seq_len + 1, (Kind::Int64, Device::Cpu));
    let start = Instant::now();
    for b in 0..nbatches {
        let batch_indexes = indexes.i(b * batch_size..(b + 1) * batch_size);
        let _ = black_box(data.unfold(0, seq_len, 1).index_select(0, &batch_indexes));
    }
    println!("text batches unfold x50:       {:?}", start.elapsed());
    let start = Instant::now();
    for b in 0..nbatches {
        let batch_indexes =
            Vec::<i64>::try_from(indexes.i(b * batch_size..(b + 1) * batch_size)).unwrap();
        let batch: Vec<_> = batch_indexes.iter().map(|&i| data.i(i..i + seq_len)).collect();
        let batch: Vec<_> = batch.iter().collect();
        let _ = black_box(Tensor::stack(&batch, 0));
    }
    println!("text batches narrow+stack x50: {:?}", start.elapsed());

    // CIFAR-style batch decoding: vectorized slicing vs the previous
    // per-sample copy loop (10k records of 1 label byte + 3072 image bytes).
    let data: Vec<u8> = (0..10_000usize * 3073).map(|i| (i % 256) as u8).collect();
    let start = Instant::now();
    let content = Tensor::from_slice(&data).view([10_000, 3073]);
    let labels = content.select(1, 0).to_kind(Kind::Int64);
    let images =
        content.narrow(1, 1, 3072).reshape([10_000, 3, 32, 32]).to_kind(Kind::Float) / 255.0;
    black_box((&images, &labels));
    println!("cifar decode vectorized:       {:?}", start.elapsed());

    let start = Instant::now();
    let content = Tensor::from_slice(&data);
    let images = Tensor::zeros([10_000, 3, 32, 32], (Kind::Float, Device::Cpu));
    let labels = Tensor::zeros([10_000], (Kind::Int64, Device::Cpu));
    for index in 0..10_000i64 {
        let offset = 3073 * index;
        let mut label_view = labels.i(index);
        label_view.copy_(&content.i(offset));
        let mut image_view = images.i(index);
        image_view
            .copy_(&content.narrow(0, 1 + offset, 3072).view((3i64, 32, 32)).to_kind(Kind::Float));
    }
    let images = images.to_kind(Kind::Float) / 255.0;
    black_box((&images, &labels));
    println!("cifar decode per-sample loop:  {:?}", start.elapsed());

    // Bulk Iter reads vs one FFI value extraction per element (the old
    // implementation; on CUDA each of those is also a device sync).
    let t = Tensor::arange(1_000_000, (Kind::Int64, Device::Cpu));
    let start = Instant::now();
    let bulk: i64 = t.iter::<i64>().unwrap().sum();
    println!("iter bulk 1M elements:         {:?}", start.elapsed());
    let start = Instant::now();
    let mut per_elem = 0i64;
    for i in 0..1_000_000 {
        per_elem += t.int64_value(&[i]);
    }
    println!("per-element reads 1M elements: {:?}", start.elapsed());
    assert_eq!(bulk, per_elem);
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
