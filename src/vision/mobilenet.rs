//! MobileNet V2 implementation.
//! <https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.htmla>
use crate::nn::{self, ModuleT};

// torchvision initializes every MobileNetV2 conv with
// kaiming_normal_(mode="fan_out") — the default a=0 leaky_relu gain is the
// same sqrt(2) as relu — and zeros any bias; this only affects from-scratch
// training (pretrained loads overwrite it).
const CONV_WS_INIT: nn::Init = nn::Init::Kaiming {
    dist: nn::init::NormalOrUniform::Normal,
    fan: nn::init::FanInOut::FanOut,
    non_linearity: nn::init::NonLinearity::ReLU,
};

#[allow(clippy::identity_op)]
// Conv2D + BatchNorm2D + ReLU6
fn cbr(p: nn::Path, c_in: i64, c_out: i64, ks: i64, stride: i64, g: i64) -> impl ModuleT + use<> {
    let conv2d = nn::ConvConfig {
        stride,
        padding: (ks - 1) / 2,
        groups: g,
        bias: false,
        ws_init: CONV_WS_INIT,
        ..Default::default()
    };
    nn::seq_t()
        .add(nn::conv2d(&p / 0, c_in, c_out, ks, conv2d))
        .add(nn::batch_norm2d(&p / 1, c_out, Default::default()))
        // ReLU6 in a single kernel.
        .add_fn(|xs| xs.clamp(0., 6.))
}

// Inverted Residual block.
fn inv(p: nn::Path, c_in: i64, c_out: i64, stride: i64, er: i64) -> impl ModuleT + use<> {
    let c_hidden = er * c_in;
    let mut conv = nn::seq_t();
    let mut id = 0;
    if er != 1 {
        conv = conv.add(cbr(&p / id, c_in, c_hidden, 1, 1, 1));
        id += 1;
    }
    conv = conv
        .add(cbr(&p / id, c_hidden, c_hidden, 3, stride, c_hidden))
        .add(nn::conv2d(
            &p / (id + 1),
            c_hidden,
            c_out,
            1,
            nn::ConvConfig { bias: false, ws_init: CONV_WS_INIT, ..Default::default() },
        ))
        .add(nn::batch_norm2d(&p / (id + 2), c_out, Default::default()));
    nn::func_t(move |xs, train| {
        let ys = xs.apply_t(&conv, train);
        if stride == 1 && c_in == c_out {
            xs + ys
        } else {
            ys
        }
    })
}

const INVERTED_RESIDUAL_SETTINGS: [(i64, i64, i64, i64); 7] = [
    (1, 16, 1, 1),
    (6, 24, 2, 2),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
];

#[allow(clippy::identity_op)]
pub fn v2(p: &nn::Path, nclasses: i64) -> impl ModuleT + use<> {
    let f_p = p / "features";
    let c_p = p / "classifier";
    let mut c_in = 32;
    let mut features = nn::seq_t().add(cbr(&f_p / "0", 3, c_in, 3, 2, 1));
    let mut layer_id = 1;
    for &(er, c_out, n, stride) in INVERTED_RESIDUAL_SETTINGS.iter() {
        for i in 0..n {
            let stride = if i == 0 { stride } else { 1 };
            let f_p = &f_p / layer_id;
            features = features.add(inv(&f_p / "conv", c_in, c_out, stride, er));
            c_in = c_out;
            layer_id += 1;
        }
    }
    features = features.add(cbr(&f_p / layer_id, c_in, 1280, 1, 1, 1));
    // torchvision uses normal_(0, 0.01) for the classifier weight and zeros
    // its bias (from-scratch training only).
    let classifier = nn::seq_t().add_fn_t(|xs, train| xs.dropout(0.2, train)).add(nn::linear(
        &c_p / 1,
        1280,
        nclasses,
        nn::LinearConfig {
            ws_init: nn::Init::Randn { mean: 0., stdev: 0.01 },
            bs_init: Some(nn::Init::Const(0.)),
            ..Default::default()
        },
    ));
    nn::func_t(move |xs, train| {
        // Dtype-preserving pooling, matching torchvision's
        // adaptive_avg_pool2d + flatten (the previous two-pass mean upcast
        // half-precision activations to f32).
        xs.apply_t(&features, train)
            .adaptive_avg_pool2d([1, 1])
            .flat_view()
            .apply_t(&classifier, train)
    })
}
