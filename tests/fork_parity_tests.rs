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

// ---------------------------------------------------------------------------
// Round-2 fixes: regression tests.
// ---------------------------------------------------------------------------

#[test]
fn bf16_elements_roundtrip() {
    // bf16 was mis-tagged as Kind::Half (fp16), silently reinterpreting the
    // bits; from_slice must produce a BFloat16 tensor that round-trips.
    let vals: Vec<half::bf16> =
        [0.5f32, -1.25, 3.0, 100.0].iter().map(|&v| half::bf16::from_f32(v)).collect();
    let t = Tensor::from_slice(&vals);
    assert_eq!(t.kind(), Kind::BFloat16);
    assert_eq!(Vec::<half::bf16>::try_from(&t).unwrap(), vals);
    // And the values must be numerically right, not just bit-preserved.
    assert_eq!(f64::try_from(t.sum(Kind::Double)).unwrap(), 102.25);
}

#[test]
fn set_momentum_group_works_for_adam() {
    // ato_set_momentum_group was missing an `else`, so any non-SGD optimizer
    // set the value and then threw "unexpected optimizer".
    let vs = nn::VarStore::new(Device::Cpu);
    let _w = vs.root().var("w", &[4], nn::Init::Const(0.));
    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
    opt.set_momentum_group(0, 0.8);
}

#[test]
fn text_data_iter_yields_contiguous_windows() {
    // The vectorized TextDataIter must still return [batch, seq_len] windows
    // of consecutive positions. Labels are assigned in first-appearance order
    // over a cyclic alphabet, so consecutive positions differ by 1 mod 26.
    let path = std::env::temp_dir().join("tch_test_text_windows.txt");
    let bytes: Vec<u8> = (0..10_000u32).map(|i| b'a' + (i % 26) as u8).collect();
    std::fs::write(&path, &bytes).unwrap();
    let text = tch::data::TextData::new(&path).unwrap();
    let (seq_len, batch_size) = (17i64, 8i64);
    let mut nbatches = 0;
    for batch in text.iter_shuffle(seq_len, batch_size).take(10) {
        assert_eq!(batch.size(), [batch_size, seq_len]);
        for row in Vec::<Vec<i64>>::try_from(&batch.to_kind(Kind::Int64)).unwrap() {
            for j in 1..row.len() {
                assert_eq!((row[j] - row[j - 1]).rem_euclid(26), 1, "row {row:?}");
            }
        }
        nbatches += 1;
    }
    assert_eq!(nbatches, 10);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn forward_all_zero_layers_requested() {
    // forward_all(_, Some(0)) used to run layers[0] anyway.
    let vs = nn::VarStore::new(Device::Cpu);
    let seq = nn::seq().add(nn::linear(vs.root() / "l", 4, 4, Default::default()));
    let xs = Tensor::zeros([2, 4], (Kind::Float, Device::Cpu));
    assert!(seq.forward_all(&xs, Some(0)).is_empty());
    assert_eq!(seq.forward_all(&xs, Some(1)).len(), 1);
    assert_eq!(seq.forward_all(&xs, None).len(), 1);
}

#[test]
fn entry_or_var_honors_store_kind() {
    // Entry::or_var hardcoded Kind::Float, ignoring vs.half()/bfloat16().
    let mut vs = nn::VarStore::new(Device::Cpu);
    vs.half();
    let t = vs.root().entry("w").or_var(&[4], nn::Init::Const(1.));
    assert_eq!(t.kind(), Kind::Half);
}

#[test]
#[should_panic(expected = "divisible")]
fn group_norm_validates_channels() {
    // PyTorch raises at construction; we now assert instead of failing with
    // an opaque aten error at the first forward.
    let vs = nn::VarStore::new(Device::Cpu);
    let _gn = nn::group_norm(vs.root(), 3, 10, Default::default());
}

#[test]
fn indexing_matches_previous_semantics() {
    // The .i() fast path (no upfront shallow clone) must keep identical
    // results across single and mixed index specs.
    use tch::IndexOp;
    let t = Tensor::arange(24, (Kind::Int64, Device::Cpu)).view([2, 3, 4]);
    assert_eq!(t.i(1).size(), [3, 4]);
    assert_eq!(t.i((.., 1..3)).size(), [2, 2, 4]);
    assert_eq!(t.i((.., .., ..)).size(), [2, 3, 4]);
    assert_eq!(i64::try_from(t.i((1, 2, 3))).unwrap(), 23);
    let idx = Tensor::from_slice(&[2i64, 0]);
    let sel = t.i((.., &idx));
    assert_eq!(sel.size(), [2, 2, 4]);
    assert_eq!(i64::try_from(sel.i((0, 0, 0))).unwrap(), 8);
    // Full-range slice of a 1-d tensor still returns an equal tensor.
    let v = Tensor::from_slice(&[1i64, 2, 3]);
    assert_eq!(Vec::<i64>::try_from(v.i(..)).unwrap(), vec![1, 2, 3]);
}

/// Print-only timing comparisons for the round-2 optimizations; run with
/// `cargo test --release -- --ignored --nocapture`. Items marked "rebuild"
/// measure changes whose old version no longer exists in the tree: check out
/// the previous commit for the baseline number.
#[test]
#[ignore]
fn bench_round2_optimizations() {
    use std::hint::black_box;
    use std::time::Instant;
    use tch::IndexOp;

    // Single-narrow .i(): no upfront shallow clone per call (rebuild).
    let t = Tensor::arange(4096, (Kind::Float, Device::Cpu));
    let start = Instant::now();
    for _ in 0..100_000 {
        let _ = black_box(t.i(5..100));
    }
    println!("i(5..100) x100k:               {:?}", start.elapsed());

    // Stack-allocated stride1 vs the Vec-allocating stride().
    let start = Instant::now();
    for _ in 0..200_000 {
        let _ = black_box(t.stride1().unwrap());
    }
    println!("stride1 x200k:                 {:?}", start.elapsed());
    let start = Instant::now();
    for _ in 0..200_000 {
        let _ = black_box(t.stride()[0]);
    }
    println!("stride() vec x200k:            {:?}", start.elapsed());

    // flat_view via size_at (rebuild for the size()[0] baseline).
    let images = Tensor::randn([64, 3, 32, 32], (Kind::Float, Device::Cpu));
    let start = Instant::now();
    for _ in 0..100_000 {
        let _ = black_box(images.flat_view());
    }
    println!("flat_view x100k:               {:?}", start.elapsed());

    // onehot shape built in place vs the two-Vec concat.
    let labels = Tensor::randint(10, [4096], (Kind::Int64, Device::Cpu));
    let start = Instant::now();
    for _ in 0..2_000 {
        let _ = black_box(labels.onehot(10));
    }
    println!("onehot x2k:                    {:?}", start.elapsed());
    let start = Instant::now();
    for _ in 0..2_000 {
        let z = Tensor::zeros([labels.size(), vec![10]].concat(), (Kind::Float, Device::Cpu))
            .scatter_value_(-1, &labels.unsqueeze(-1).to_kind(Kind::Int64), 1.0);
        let _ = black_box(z);
    }
    println!("onehot concat-shape x2k:       {:?}", start.elapsed());

    // TextDataIter through the real API vs the old host-roundtrip gather.
    let path = std::env::temp_dir().join("tch_bench_text_data.txt");
    let bytes: Vec<u8> = (0..1_000_000u32).map(|i| b'a' + (i % 26) as u8).collect();
    std::fs::write(&path, &bytes).unwrap();
    let text = tch::data::TextData::new(&path).unwrap();
    let (seq_len, batch_size) = (256i64, 64i64);
    let start = Instant::now();
    for batch in text.iter_shuffle(seq_len, batch_size).take(50) {
        black_box(&batch);
    }
    println!("TextDataIter x50:              {:?}", start.elapsed());
    let data = text.data();
    let indexes = Tensor::randperm(1_000_000 - seq_len + 1, (Kind::Int64, Device::Cpu));
    let start = Instant::now();
    for b in 0..50i64 {
        let batch_indexes =
            Vec::<i64>::try_from(indexes.i(b * batch_size..(b + 1) * batch_size)).unwrap();
        let batch: Vec<_> = batch_indexes.iter().map(|&i| data.i(i..i + seq_len)).collect();
        let batch: Vec<_> = batch.iter().collect();
        let _ = black_box(Tensor::stack(&batch, 0));
    }
    println!("old narrow+stack x50:          {:?}", start.elapsed());
    let _ = std::fs::remove_file(&path);

    // Same-dtype VarStore::load: to_kind/set_data per variable now skipped
    // (rebuild).
    let vs_path = std::env::temp_dir().join("tch_bench_vs.safetensors");
    let vs = nn::VarStore::new(Device::Cpu);
    for i in 0..200 {
        let _ = vs.root().var(&format!("w{i}"), &[64, 64], nn::Init::Const(0.5));
    }
    vs.save(&vs_path).unwrap();
    let mut vs2 = nn::VarStore::new(Device::Cpu);
    for i in 0..200 {
        let _ = vs2.root().var(&format!("w{i}"), &[64, 64], nn::Init::Const(0.));
    }
    let start = Instant::now();
    for _ in 0..20 {
        vs2.load(&vs_path).unwrap();
    }
    println!("VarStore::load same-kind x20:  {:?}", start.elapsed());
    let _ = std::fs::remove_file(&vs_path);

    // Image resize: output buffer is torch::empty instead of torch::zeros in
    // torch_api.cpp; also covers at_load_image (rebuild).
    let img =
        Tensor::randint(256, [3, 1080, 1920], (Kind::Int64, Device::Cpu)).to_kind(Kind::Uint8);
    let start = Instant::now();
    for _ in 0..50 {
        let _ = black_box(tch::vision::image::resize(&img, 224, 224).unwrap());
    }
    println!("image resize 1080p->224 x50:   {:?}", start.elapsed());
}

#[test]
fn load_preserves_model_dtype_like_pytorch() {
    // PyTorch load_state_dict casts checkpoint values into each parameter's
    // dtype; adopting the checkpoint's dtype takes the explicit variant.
    let filename = std::env::temp_dir().join(format!("tch-load-dtype-{}", std::process::id()));
    let mut vs1 = nn::VarStore::new(Device::Cpu);
    let _w1 = vs1.root().var("w", &[8], nn::Init::Const(1.5));
    vs1.half();
    vs1.save(&filename).unwrap();

    let mut vs2 = nn::VarStore::new(Device::Cpu);
    let w2 = vs2.root().var("w", &[8], nn::Init::Const(0.));
    vs2.load(&filename).unwrap();
    assert_eq!(w2.kind(), Kind::Float);
    assert_eq!(f64::try_from(w2.mean(Kind::Float)).unwrap(), 1.5);

    let mut vs3 = nn::VarStore::new(Device::Cpu);
    let w3 = vs3.root().var("w", &[8], nn::Init::Const(0.));
    vs3.load_with_precision_update(&filename).unwrap();
    assert_eq!(w3.kind(), Kind::Half);
    assert_eq!(f64::try_from(w3.mean(Kind::Half)).unwrap(), 1.5);
    let _ = std::fs::remove_file(&filename);
}

#[test]
fn negative_slice_bounds_match_python() {
    use tch::IndexOp;
    let t = Tensor::arange(5, (Kind::Int64, Device::Cpu));
    // t[:-1]
    assert_eq!(Vec::<i64>::try_from(t.i((..-1,))).unwrap(), vec![0, 1, 2, 3]);
    // t[-2:]
    assert_eq!(Vec::<i64>::try_from(t.i((-2..,))).unwrap(), vec![3, 4]);
    // t[1:-1] (bounds computed so clippy doesn't flag the literal as reversed)
    let (start, end) = (1, -1);
    assert_eq!(Vec::<i64>::try_from(t.i((start..end,))).unwrap(), vec![1, 2, 3]);
    // t[-2:-1]
    assert_eq!(Vec::<i64>::try_from(t.i((-2..-1,))).unwrap(), vec![3]);
    // Inclusive negative end reaches through the last element: t[1:] here.
    let end = -1;
    assert_eq!(Vec::<i64>::try_from(t.i((1..=end,))).unwrap(), vec![1, 2, 3, 4]);
    // Out-of-range bounds clamp instead of erroring, as in Python.
    assert_eq!(Vec::<i64>::try_from(t.i((-100..,))).unwrap(), vec![0, 1, 2, 3, 4]);
    assert_eq!(Vec::<i64>::try_from(t.i((1..100,))).unwrap(), vec![1, 2, 3, 4]);
    // Reversed ranges stay empty.
    let (start, end) = (3, 1);
    assert_eq!(t.i((start..end,)).numel(), 0);
    // Negative bounds on a non-leading dimension.
    let t2 = Tensor::arange(6, (Kind::Int64, Device::Cpu)).view([2, 3]);
    let r = t2.i((.., ..-1));
    assert_eq!(r.size(), vec![2, 2]);
    assert_eq!(Vec::<i64>::try_from(r.reshape(-1)).unwrap(), vec![0, 1, 3, 4]);
}

#[test]
fn fallible_scalar_op_returns_err_not_panic() {
    // Integers to negative integer powers raise an error in libtorch. The
    // Scalar temporary used to run an error check in its Drop, which picked up
    // the pending error and panicked before f_* could return Err.
    let t = Tensor::from_slice(&[1i64, 2, 4]);
    let result = t.f_pow_tensor_scalar(-1);
    assert!(result.is_err(), "expected Err, got {result:?}");
    // The error must have been consumed: the next op works normally.
    let ok = t.f_pow_tensor_scalar(2).unwrap();
    assert_eq!(Vec::<i64>::try_from(ok).unwrap(), vec![1, 4, 16]);
}

#[test]
fn relu6_single_kernel_matches_reference() {
    let t = Tensor::arange_start(-3, 9, (Kind::Float, Device::Cpu));
    let fused = t.clamp(0., 6.);
    let reference = t.relu().clamp_max(6.);
    assert_eq!(
        Vec::<f32>::try_from(fused).unwrap(),
        Vec::<f32>::try_from(reference).unwrap()
    );
}

#[test]
fn load_rejects_shape_mismatch() {
    // PyTorch's load_state_dict errors on shape mismatch; f_copy_ used to
    // silently broadcast the checkpoint tensor into the variable.
    let filename = std::env::temp_dir().join(format!("tch-load-shape-{}", std::process::id()));
    let vs1 = nn::VarStore::new(Device::Cpu);
    let _w1 = vs1.root().var("w", &[1, 4], nn::Init::Const(1.0));
    vs1.save(&filename).unwrap();

    let mut vs2 = nn::VarStore::new(Device::Cpu);
    let _w2 = vs2.root().var("w", &[3, 4], nn::Init::Const(0.));
    assert!(vs2.load(&filename).is_err(), "a [1, 4] tensor must not broadcast into [3, 4]");
    let _ = std::fs::remove_file(&filename);
}

#[test]
fn truncated_normal_init_matches_pytorch() {
    // trunc_normal_(mean=0, std=0.1, a=-2, b=2): every draw lies inside the
    // bounds and the sample std matches the requested one (truncation at
    // +/-20 sigma is a no-op for the variance).
    let t = tch::nn::init(
        nn::Init::TruncatedNormal { mean: 0., stdev: 0.1, lo: -2., up: 2. },
        &[64, 1024],
        Device::Cpu,
    );
    let max = f64::try_from(t.abs().max()).unwrap();
    assert!(max <= 2.0, "draw outside truncation bounds: {max}");
    let std = f64::try_from(t.std(true)).unwrap();
    assert!((std - 0.1).abs() / 0.1 < 0.05, "std {std} vs expected 0.1");

    // Tight bounds actually truncate: N(0, 1) restricted to [-0.5, 0.5].
    let t = tch::nn::init(
        nn::Init::TruncatedNormal { mean: 0., stdev: 1., lo: -0.5, up: 0.5 },
        &[16, 1024],
        Device::Cpu,
    );
    let max = f64::try_from(t.abs().max()).unwrap();
    assert!(max <= 0.5, "draw outside truncation bounds: {max}");
}

#[test]
fn inception_aux_logits() {
    use tch::vision::inception;
    let vs = nn::VarStore::new(Device::Cpu);
    let net = inception::v3(&vs.root(), 10);

    // The aux branch registers torchvision-named variables.
    let variables = vs.variables();
    assert!(variables.contains_key("AuxLogits.conv0.conv.weight"));
    assert!(variables.contains_key("AuxLogits.conv1.conv.weight"));
    assert!(variables.contains_key("AuxLogits.fc.weight"));

    let xs = Tensor::zeros([1, 3, 299, 299], (Kind::Float, Device::Cpu));
    // Training mode returns both towers; eval mode skips the aux branch,
    // mirroring torchvision.
    let (main, aux) = net.forward_t_with_aux(&xs, true);
    assert_eq!(main.size(), vec![1, 10]);
    assert_eq!(aux.expect("aux logits in training mode").size(), vec![1, 10]);
    let (main, aux) = net.forward_t_with_aux(&xs, false);
    assert_eq!(main.size(), vec![1, 10]);
    assert!(aux.is_none());
}

// Runs a small deterministic training loop and returns the per-step losses
// plus the final flattened weights, so the foreach (multi-tensor) optimizers
// can be checked against libtorch's C++ reference implementations.
fn train_tiny_net(opt_of: impl FnOnce(&nn::VarStore) -> nn::Optimizer) -> (Vec<f64>, Vec<f64>) {
    tch::manual_seed(42);
    let vs = nn::VarStore::new(Device::Cpu);
    let root = vs.root();
    let l1 = nn::linear(&root / "l1", 4, 8, Default::default());
    let l2 = nn::linear(&root / "l2", 8, 1, Default::default());
    let mut opt = opt_of(&vs);
    let xs = (Tensor::arange(20, (Kind::Float, Device::Cpu)).view([5, 4]) - 10.0) / 7.0;
    let ys = (Tensor::arange(5, (Kind::Float, Device::Cpu)).view([5, 1]) - 2.0) / 3.0;
    let mut losses = Vec::new();
    for _ in 0..8 {
        let loss = xs.apply(&l1).relu().apply(&l2).mse_loss(&ys, tch::Reduction::Mean);
        opt.backward_step(&loss);
        losses.push(f64::try_from(&loss).unwrap());
    }
    let mut weights = Vec::new();
    // HashMap iteration order is nondeterministic: sort by name so the two
    // runs being compared serialize their weights identically.
    let mut named: Vec<_> = vs.variables().into_iter().collect();
    named.sort_by(|a, b| a.0.cmp(&b.0));
    for (_name, t) in named {
        weights.extend(Vec::<f64>::try_from(t.to_kind(Kind::Double).view(-1)).unwrap());
    }
    (losses, weights)
}

fn assert_all_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() <= tol * (1.0 + x.abs()), "{what}[{i}]: {x} vs {y}");
    }
}

#[test]
fn foreach_adamw_matches_cpp_adamw() {
    let (l_ref, w_ref) = train_tiny_net(|vs| nn::AdamW::default().build(vs, 1e-2).unwrap());
    let (l_new, w_new) =
        train_tiny_net(|vs| nn::ForeachAdamW::default().build(vs, 1e-2).unwrap());
    assert_all_close(&l_ref, &l_new, 1e-6, "losses");
    assert_all_close(&w_ref, &w_new, 1e-6, "weights");
}

#[test]
fn foreach_adam_matches_cpp_adam_with_l2() {
    // wd != 0 exercises the L2 (non-decoupled) branch, amsgrad the running
    // max of the second moment.
    let cfg = |amsgrad| nn::Adam { wd: 0.1, amsgrad, ..Default::default() };
    let fcfg =
        |amsgrad| nn::ForeachAdam { wd: 0.1, amsgrad, ..Default::default() };
    for amsgrad in [false, true] {
        let (l_ref, w_ref) = train_tiny_net(|vs| cfg(amsgrad).build(vs, 1e-2).unwrap());
        let (l_new, w_new) = train_tiny_net(|vs| fcfg(amsgrad).build(vs, 1e-2).unwrap());
        assert_all_close(&l_ref, &l_new, 1e-6, "losses");
        assert_all_close(&w_ref, &w_new, 1e-6, "weights");
    }
}

#[test]
fn foreach_adamw_respects_group_lr_and_zero_grad() {
    tch::manual_seed(7);
    let vs = nn::VarStore::new(Device::Cpu);
    let root = vs.root();
    // Group 1 gets lr 0: its weights must not move.
    let frozen = root.set_group(1).var("frozen", &[3], nn::Init::Const(1.0));
    let live = root.var("live", &[3], nn::Init::Const(1.0));
    let mut opt = nn::ForeachAdamW::default().build(&vs, 1e-1).unwrap();
    opt.set_lr_group(1, 0.0);
    for _ in 0..3 {
        let loss = (&frozen + &live).sum(Kind::Float);
        opt.backward_step(&loss);
    }
    let frozen_v = Vec::<f64>::try_from(frozen.to_kind(Kind::Double)).unwrap();
    let live_v = Vec::<f64>::try_from(live.to_kind(Kind::Double)).unwrap();
    // AdamW's decoupled decay also scales by lr, so lr 0 leaves the weights
    // bit-identical.
    assert_eq!(frozen_v, vec![1.0, 1.0, 1.0]);
    assert!(live_v.iter().all(|&v| v < 1.0), "live weights should have moved: {live_v:?}");
    // zero_grad through the foreach path actually zeroes the grads.
    opt.zero_grad();
    let g = frozen.grad();
    assert!(g.defined());
    assert_eq!(f64::try_from(g.abs().sum(Kind::Double)).unwrap(), 0.0);
}
