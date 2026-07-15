//! SqueezeNet implementation.

use crate::{nn, nn::Module, nn::ModuleT, Tensor};

fn max_pool2d(xs: &Tensor) -> Tensor {
    xs.max_pool2d([3, 3], [2, 2], [0, 0], [1, 1], true)
}

// torchvision's from-scratch init: plain kaiming_uniform_ (gain sqrt(2),
// wider than tch's a=sqrt(5) default) for every conv except the final
// classifier, and all conv biases zeroed. Pretrained loads overwrite this.
fn conv_cfg(padding: i64, stride: i64) -> nn::ConvConfig {
    nn::ConvConfig {
        padding,
        stride,
        ws_init: nn::Init::Kaiming {
            dist: nn::init::NormalOrUniform::Uniform,
            fan: nn::init::FanInOut::FanIn,
            non_linearity: nn::init::NonLinearity::ReLU,
        },
        bs_init: Some(nn::Init::Const(0.)),
        ..Default::default()
    }
}

fn fire(p: nn::Path, c_in: i64, c_squeeze: i64, c_exp1: i64, c_exp3: i64) -> impl Module {
    let squeeze = nn::conv2d(&p / "squeeze", c_in, c_squeeze, 1, conv_cfg(0, 1));
    let exp1 = nn::conv2d(&p / "expand1x1", c_squeeze, c_exp1, 1, conv_cfg(0, 1));
    let exp3 = nn::conv2d(&p / "expand3x3", c_squeeze, c_exp3, 3, conv_cfg(1, 1));
    nn::func(move |xs| {
        let xs = xs.apply(&squeeze).relu();
        Tensor::cat(&[xs.apply(&exp1).relu(), xs.apply(&exp3).relu()], 1)
    })
}

fn squeezenet(p: &nn::Path, v1_0: bool, nclasses: i64) -> impl ModuleT {
    let f_p = p / "features";
    let c_p = p / "classifier";
    let initial_conv_cfg = conv_cfg(0, 2);
    // torchvision initializes the final classifier conv with normal(0, 0.01).
    let final_conv_cfg = nn::ConvConfig {
        ws_init: nn::Init::Randn { mean: 0., stdev: 0.01 },
        bs_init: Some(nn::Init::Const(0.)),
        ..Default::default()
    };
    let features = if v1_0 {
        nn::seq_t()
            .add(nn::conv2d(&f_p / "0", 3, 96, 7, initial_conv_cfg))
            .add_fn(|xs| xs.relu())
            .add_fn(max_pool2d)
            .add(fire(&f_p / "3", 96, 16, 64, 64))
            .add(fire(&f_p / "4", 128, 16, 64, 64))
            .add(fire(&f_p / "5", 128, 32, 128, 128))
            .add_fn(max_pool2d)
            .add(fire(&f_p / "7", 256, 32, 128, 128))
            .add(fire(&f_p / "8", 256, 48, 192, 192))
            .add(fire(&f_p / "9", 384, 48, 192, 192))
            .add(fire(&f_p / "10", 384, 64, 256, 256))
            .add_fn(max_pool2d)
            .add(fire(&f_p / "12", 512, 64, 256, 256))
    } else {
        nn::seq_t()
            .add(nn::conv2d(&f_p / "0", 3, 64, 3, initial_conv_cfg))
            .add_fn(|xs| xs.relu())
            .add_fn(max_pool2d)
            .add(fire(&f_p / "3", 64, 16, 64, 64))
            .add(fire(&f_p / "4", 128, 16, 64, 64))
            .add_fn(max_pool2d)
            .add(fire(&f_p / "6", 128, 32, 128, 128))
            .add(fire(&f_p / "7", 256, 32, 128, 128))
            .add_fn(max_pool2d)
            .add(fire(&f_p / "9", 256, 48, 192, 192))
            .add(fire(&f_p / "10", 384, 48, 192, 192))
            .add(fire(&f_p / "11", 384, 64, 256, 256))
            .add(fire(&f_p / "12", 512, 64, 256, 256))
    };
    features
        .add_fn_t(|xs, train| xs.dropout(0.5, train))
        .add(nn::conv2d(&c_p / "1", 512, nclasses, 1, final_conv_cfg))
        .add_fn(|xs| xs.relu().adaptive_avg_pool2d([1, 1]).flat_view())
}

pub fn v1_0(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    squeezenet(p, true, nclasses)
}

pub fn v1_1(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    squeezenet(p, false, nclasses)
}
