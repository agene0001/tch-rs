//! VGG models.
//!
//! Pre-trained weights for the vgg-13/16/19 models can be found here:
//! <https://github.com/LaurentMazare/tch-rs/releases/download/mw/vgg13.ot>
//! <https://github.com/LaurentMazare/tch-rs/releases/download/mw/vgg16.ot>
//! <https://github.com/LaurentMazare/tch-rs/releases/download/mw/vgg19.ot>
use crate::{nn, nn::Conv2D, nn::SequentialT};

// Each list element contains multiple convolutions with some specified number
// of features followed by a single max-pool layer.
fn layers_a() -> Vec<Vec<i64>> {
    vec![vec![64], vec![128], vec![256, 256], vec![512, 512], vec![512, 512]]
}

fn layers_b() -> Vec<Vec<i64>> {
    vec![vec![64, 64], vec![128, 128], vec![256, 256], vec![512, 512], vec![512, 512]]
}
fn layers_d() -> Vec<Vec<i64>> {
    vec![
        vec![64, 64],
        vec![128, 128],
        vec![256, 256, 256],
        vec![512, 512, 512],
        vec![512, 512, 512],
    ]
}
fn layers_e() -> Vec<Vec<i64>> {
    vec![
        vec![64, 64],
        vec![128, 128],
        vec![256, 256, 256, 256],
        vec![512, 512, 512, 512],
        vec![512, 512, 512, 512],
    ]
}

fn conv2d(p: nn::Path, c_in: i64, c_out: i64) -> Conv2D {
    // torchvision's VGG `_initialize_weights` uses
    // kaiming_normal_(mode="fan_out", nonlinearity="relu") for conv weights
    // and zeros the biases; this only affects from-scratch training
    // (pretrained loads overwrite it).
    let ws_init = nn::Init::Kaiming {
        dist: nn::init::NormalOrUniform::Normal,
        fan: nn::init::FanInOut::FanOut,
        non_linearity: nn::init::NonLinearity::ReLU,
    };
    let conv2d_cfg = nn::ConvConfig {
        stride: 1,
        padding: 1,
        ws_init,
        bs_init: Some(nn::Init::Const(0.)),
        ..Default::default()
    };
    nn::conv2d(p, c_in, c_out, 3, conv2d_cfg)
}

fn vgg(p: &nn::Path, cfg: Vec<Vec<i64>>, nclasses: i64, batch_norm: bool) -> SequentialT {
    let c = p / "classifier";
    let mut seq = nn::seq_t();
    let f = p / "features";
    let mut c_in = 3;
    for channels in cfg.into_iter() {
        for &c_out in channels.iter() {
            let l = seq.len();
            seq = seq.add(conv2d(&f / &l.to_string(), c_in, c_out));
            if batch_norm {
                let l = seq.len();
                seq = seq.add(nn::batch_norm2d(&f / &l.to_string(), c_out, Default::default()));
            };
            seq = seq.add_fn(|xs| xs.relu());
            c_in = c_out;
        }
        seq = seq.add_fn(|xs| xs.max_pool2d_default(2));
    }
    // torchvision's `_initialize_weights` uses normal_(0, 0.01) for linear
    // weights and zeros the biases (from-scratch training only).
    let linear_cfg = nn::LinearConfig {
        ws_init: nn::Init::Randn { mean: 0., stdev: 0.01 },
        bs_init: Some(nn::Init::Const(0.)),
        ..Default::default()
    };
    // torchvision applies an adaptive 7x7 average pooling between the
    // features and the classifier; this is the identity for 224x224 inputs
    // but keeps other input sizes working.
    seq.add_fn(|xs| xs.adaptive_avg_pool2d([7, 7]).flat_view())
        .add(nn::linear(&c / "0", 512 * 7 * 7, 4096, linear_cfg))
        .add_fn(|xs| xs.relu())
        .add_fn_t(|xs, train| xs.dropout(0.5, train))
        .add(nn::linear(&c / "3", 4096, 4096, linear_cfg))
        .add_fn(|xs| xs.relu())
        .add_fn_t(|xs, train| xs.dropout(0.5, train))
        .add(nn::linear(&c / "6", 4096, nclasses, linear_cfg))
}

pub fn vgg11(p: &nn::Path, nclasses: i64) -> SequentialT {
    vgg(p, layers_a(), nclasses, false)
}

pub fn vgg11_bn(p: &nn::Path, nclasses: i64) -> SequentialT {
    vgg(p, layers_a(), nclasses, true)
}

pub fn vgg13(p: &nn::Path, nclasses: i64) -> SequentialT {
    vgg(p, layers_b(), nclasses, false)
}

pub fn vgg13_bn(p: &nn::Path, nclasses: i64) -> SequentialT {
    vgg(p, layers_b(), nclasses, true)
}

pub fn vgg16(p: &nn::Path, nclasses: i64) -> SequentialT {
    vgg(p, layers_d(), nclasses, false)
}

pub fn vgg16_bn(p: &nn::Path, nclasses: i64) -> SequentialT {
    vgg(p, layers_d(), nclasses, true)
}

pub fn vgg19(p: &nn::Path, nclasses: i64) -> SequentialT {
    vgg(p, layers_e(), nclasses, false)
}

pub fn vgg19_bn(p: &nn::Path, nclasses: i64) -> SequentialT {
    vgg(p, layers_e(), nclasses, true)
}
