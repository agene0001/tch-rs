//! InceptionV3.
use crate::{nn, nn::ModuleT, Tensor};

// torchvision initializes every Inception conv/linear weight with
// trunc_normal_(mean=0, std=<module stddev, default 0.1>, a=-2, b=2); the aux
// branch overrides the std on two of its modules. Only affects from-scratch
// training (pretrained loads overwrite the weights).
fn trunc_normal(stdev: f64) -> nn::Init {
    nn::Init::TruncatedNormal { mean: 0., stdev, lo: -2., up: 2. }
}

fn conv_bn_std(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    ksize: i64,
    pad: i64,
    stride: i64,
    stddev: f64,
) -> impl ModuleT {
    let conv2d_cfg = nn::ConvConfig {
        stride,
        padding: pad,
        bias: false,
        ws_init: trunc_normal(stddev),
        ..Default::default()
    };
    let bn_cfg = nn::BatchNormConfig { eps: 0.001, ..Default::default() };
    nn::seq_t()
        .add(nn::conv2d(&p / "conv", c_in, c_out, ksize, conv2d_cfg))
        .add(nn::batch_norm2d(&p / "bn", c_out, bn_cfg))
        .add_fn(|xs| xs.relu())
}

fn conv_bn(p: nn::Path, c_in: i64, c_out: i64, ksize: i64, pad: i64, stride: i64) -> impl ModuleT {
    conv_bn_std(p, c_in, c_out, ksize, pad, stride, 0.1)
}

fn conv_bn2(p: nn::Path, c_in: i64, c_out: i64, ksize: [i64; 2], pad: [i64; 2]) -> impl ModuleT {
    let conv2d_cfg = nn::ConvConfigND::<[i64; 2]> {
        padding: pad,
        bias: false,
        ws_init: trunc_normal(0.1),
        ..Default::default()
    };
    let bn_cfg = nn::BatchNormConfig { eps: 0.001, ..Default::default() };
    nn::seq_t()
        .add(nn::conv(&p / "conv", c_in, c_out, ksize, conv2d_cfg))
        .add(nn::batch_norm2d(&p / "bn", c_out, bn_cfg))
        .add_fn(|xs| xs.relu())
}

fn max_pool2d(xs: &Tensor, ksize: i64, stride: i64) -> Tensor {
    xs.max_pool2d([ksize, ksize], [stride, stride], [0, 0], [1, 1], false)
}

fn inception_a(p: nn::Path, c_in: i64, c_pool: i64) -> impl ModuleT {
    let b1 = conv_bn(&p / "branch1x1", c_in, 64, 1, 0, 1);
    let b2_1 = conv_bn(&p / "branch5x5_1", c_in, 48, 1, 0, 1);
    let b2_2 = conv_bn(&p / "branch5x5_2", 48, 64, 5, 2, 1);
    let b3_1 = conv_bn(&p / "branch3x3dbl_1", c_in, 64, 1, 0, 1);
    let b3_2 = conv_bn(&p / "branch3x3dbl_2", 64, 96, 3, 1, 1);
    let b3_3 = conv_bn(&p / "branch3x3dbl_3", 96, 96, 3, 1, 1);
    let bpool = conv_bn(&p / "branch_pool", c_in, c_pool, 1, 0, 1);
    nn::func_t(move |xs, tr| {
        let b1 = xs.apply_t(&b1, tr);
        let b2 = xs.apply_t(&b2_1, tr).apply_t(&b2_2, tr);
        let b3 = xs.apply_t(&b3_1, tr).apply_t(&b3_2, tr).apply_t(&b3_3, tr);
        let bpool = xs.avg_pool2d([3, 3], [1, 1], [1, 1], false, true, 9).apply_t(&bpool, tr);
        Tensor::cat(&[b1, b2, b3, bpool], 1)
    })
}

fn inception_b(p: nn::Path, c_in: i64) -> impl ModuleT {
    let b1 = conv_bn(&p / "branch3x3", c_in, 384, 3, 0, 2);
    let b2_1 = conv_bn(&p / "branch3x3dbl_1", c_in, 64, 1, 0, 1);
    let b2_2 = conv_bn(&p / "branch3x3dbl_2", 64, 96, 3, 1, 1);
    let b2_3 = conv_bn(&p / "branch3x3dbl_3", 96, 96, 3, 0, 2);
    nn::func_t(move |xs, tr| {
        let b1 = xs.apply_t(&b1, tr);
        let b2 = xs.apply_t(&b2_1, tr).apply_t(&b2_2, tr).apply_t(&b2_3, tr);
        let bpool = max_pool2d(xs, 3, 2);
        Tensor::cat(&[b1, b2, bpool], 1)
    })
}

fn inception_c(p: nn::Path, c_in: i64, c7: i64) -> impl ModuleT {
    let b1 = conv_bn(&p / "branch1x1", c_in, 192, 1, 0, 1);

    let b2_1 = conv_bn(&p / "branch7x7_1", c_in, c7, 1, 0, 1);
    let b2_2 = conv_bn2(&p / "branch7x7_2", c7, c7, [1, 7], [0, 3]);
    let b2_3 = conv_bn2(&p / "branch7x7_3", c7, 192, [7, 1], [3, 0]);

    let b3_1 = conv_bn(&p / "branch7x7dbl_1", c_in, c7, 1, 0, 1);
    let b3_2 = conv_bn2(&p / "branch7x7dbl_2", c7, c7, [7, 1], [3, 0]);
    let b3_3 = conv_bn2(&p / "branch7x7dbl_3", c7, c7, [1, 7], [0, 3]);
    let b3_4 = conv_bn2(&p / "branch7x7dbl_4", c7, c7, [7, 1], [3, 0]);
    let b3_5 = conv_bn2(&p / "branch7x7dbl_5", c7, 192, [1, 7], [0, 3]);

    let bpool = conv_bn(&p / "branch_pool", c_in, 192, 1, 0, 1);

    nn::func_t(move |xs, tr| {
        let b1 = xs.apply_t(&b1, tr);
        let b2 = xs.apply_t(&b2_1, tr).apply_t(&b2_2, tr).apply_t(&b2_3, tr);
        let b3 = xs
            .apply_t(&b3_1, tr)
            .apply_t(&b3_2, tr)
            .apply_t(&b3_3, tr)
            .apply_t(&b3_4, tr)
            .apply_t(&b3_5, tr);
        let bpool = xs.avg_pool2d([3, 3], [1, 1], [1, 1], false, true, 9).apply_t(&bpool, tr);
        Tensor::cat(&[b1, b2, b3, bpool], 1)
    })
}

fn inception_d(p: nn::Path, c_in: i64) -> impl ModuleT {
    let b1_1 = conv_bn(&p / "branch3x3_1", c_in, 192, 1, 0, 1);
    let b1_2 = conv_bn(&p / "branch3x3_2", 192, 320, 3, 0, 2);

    let b2_1 = conv_bn(&p / "branch7x7x3_1", c_in, 192, 1, 0, 1);
    let b2_2 = conv_bn2(&p / "branch7x7x3_2", 192, 192, [1, 7], [0, 3]);
    let b2_3 = conv_bn2(&p / "branch7x7x3_3", 192, 192, [7, 1], [3, 0]);
    let b2_4 = conv_bn(&p / "branch7x7x3_4", 192, 192, 3, 0, 2);

    nn::func_t(move |xs, tr| {
        let b1 = xs.apply_t(&b1_1, tr).apply_t(&b1_2, tr);
        let b2 = xs.apply_t(&b2_1, tr).apply_t(&b2_2, tr).apply_t(&b2_3, tr).apply_t(&b2_4, tr);
        let bpool = max_pool2d(xs, 3, 2);
        Tensor::cat(&[b1, b2, bpool], 1)
    })
}

fn inception_e(p: nn::Path, c_in: i64) -> impl ModuleT {
    let b1 = conv_bn(&p / "branch1x1", c_in, 320, 1, 0, 1);

    let b2_1 = conv_bn(&p / "branch3x3_1", c_in, 384, 1, 0, 1);
    let b2_2a = conv_bn2(&p / "branch3x3_2a", 384, 384, [1, 3], [0, 1]);
    let b2_2b = conv_bn2(&p / "branch3x3_2b", 384, 384, [3, 1], [1, 0]);

    let b3_1 = conv_bn(&p / "branch3x3dbl_1", c_in, 448, 1, 0, 1);
    let b3_2 = conv_bn(&p / "branch3x3dbl_2", 448, 384, 3, 1, 1);
    let b3_3a = conv_bn2(&p / "branch3x3dbl_3a", 384, 384, [1, 3], [0, 1]);
    let b3_3b = conv_bn2(&p / "branch3x3dbl_3b", 384, 384, [3, 1], [1, 0]);

    let bpool = conv_bn(&p / "branch_pool", c_in, 192, 1, 0, 1);

    nn::func_t(move |xs, tr| {
        let b1 = xs.apply_t(&b1, tr);

        let b2 = xs.apply_t(&b2_1, tr);
        let b2 = Tensor::cat(&[b2.apply_t(&b2_2a, tr), b2.apply_t(&b2_2b, tr)], 1);

        let b3 = xs.apply_t(&b3_1, tr).apply_t(&b3_2, tr);
        let b3 = Tensor::cat(&[b3.apply_t(&b3_3a, tr), b3.apply_t(&b3_3b, tr)], 1);

        let bpool = xs.avg_pool2d([3, 3], [1, 1], [1, 1], false, true, 9).apply_t(&bpool, tr);

        Tensor::cat(&[b1, b2, b3, bpool], 1)
    })
}

/// Remaps ImageNet-normalized inputs to the TF (-1, 1) convention, as done by
/// torchvision's `Inception3._transform_input` (enabled on the pretrained model).
fn transform_input(xs: &Tensor) -> Tensor {
    let (scale, shift) = transform_constants(xs.device(), xs.kind());
    xs * scale + shift
}

/// The broadcastable scale/shift constants used by [`transform_input`].
fn transform_constants(device: crate::Device, kind: crate::Kind) -> (Tensor, Tensor) {
    let scale = Tensor::from_slice(&[0.229 / 0.5, 0.224 / 0.5, 0.225 / 0.5])
        .view([3, 1, 1])
        .to_kind(kind)
        .to_device(device);
    let shift =
        Tensor::from_slice(&[(0.485 - 0.5) / 0.5, (0.456 - 0.5) / 0.5, (0.406 - 0.5) / 0.5])
            .view([3, 1, 1])
            .to_kind(kind)
            .to_device(device);
    (scale, shift)
}

/// The auxiliary classifier hanging off the middle of the network
/// (torchvision's `InceptionAux`), used during training as
/// `loss = main_loss + 0.4 * aux_loss`.
fn inception_aux(p: nn::Path, nclasses: i64) -> nn::SequentialT {
    let fc_cfg = nn::LinearConfig { ws_init: trunc_normal(0.001), ..Default::default() };
    nn::seq_t()
        // F.avg_pool2d(x, kernel_size=5, stride=3)
        .add_fn(|xs| xs.avg_pool2d([5, 5], [3, 3], [0, 0], false, true, 25))
        .add(conv_bn(&p / "conv0", 768, 128, 1, 0, 1))
        .add(conv_bn_std(&p / "conv1", 128, 768, 5, 0, 1, 0.01))
        .add_fn(|xs| xs.adaptive_avg_pool2d([1, 1]).flat_view())
        .add(nn::linear(&p / "fc", 768, nclasses, fc_cfg))
}

/// InceptionV3 configuration.
#[derive(Debug, Clone, Copy)]
pub struct InceptionV3Config {
    /// Remap ImageNet-normalized inputs to the TF (-1, 1) convention before
    /// the first conv. torchvision's pretrained builder enables this.
    pub transform_input: bool,
    /// Build the auxiliary classifier branch. Matches torchvision's default;
    /// the standard pretrained checkpoint contains its weights.
    pub aux_logits: bool,
}

impl Default for InceptionV3Config {
    fn default() -> Self {
        InceptionV3Config { transform_input: true, aux_logits: true }
    }
}

/// An InceptionV3 model.
///
/// `ModuleT` runs the main tower only; use
/// [`InceptionV3::forward_t_with_aux`] to also get the auxiliary logits when
/// training with the torchvision recipe (`loss = main + 0.4 * aux`).
#[derive(Debug)]
pub struct InceptionV3 {
    /// Cached transform-input constants (built once on the var-store device):
    /// rebuilding them per forward would cost two host-to-device copies per
    /// call on CUDA. `None` when `transform_input` is disabled.
    transform: Option<(Tensor, Tensor)>,
    /// Stem through Mixed_6e, where the aux branch taps in.
    pre_aux: nn::SequentialT,
    /// Mixed_7a through the final linear layer.
    post_aux: nn::SequentialT,
    aux: Option<nn::SequentialT>,
}

impl InceptionV3 {
    fn apply_transform(&self, xs: &Tensor) -> Tensor {
        match &self.transform {
            None => xs.shallow_clone(),
            Some((scale, shift)) => {
                if scale.device() == xs.device() && scale.kind() == xs.kind() {
                    xs * scale + shift
                } else {
                    // The input lives elsewhere than the build-time var-store
                    // device (or in another dtype): fall back to building the
                    // constants for it.
                    transform_input(xs)
                }
            }
        }
    }

    /// Runs the forward pass, also returning the auxiliary logits when the
    /// model was built with `aux_logits` and `train` is set — mirroring
    /// torchvision, which only computes the branch in training mode.
    pub fn forward_t_with_aux(&self, xs: &Tensor, train: bool) -> (Tensor, Option<Tensor>) {
        let xs = self.apply_transform(xs);
        let mid = xs.apply_t(&self.pre_aux, train);
        let aux = match (&self.aux, train) {
            (Some(aux), true) => Some(mid.apply_t(aux, train)),
            _ => None,
        };
        (mid.apply_t(&self.post_aux, train), aux)
    }
}

impl ModuleT for InceptionV3 {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = self.apply_transform(xs);
        xs.apply_t(&self.pre_aux, train).apply_t(&self.post_aux, train)
    }
}

/// InceptionV3 matching torchvision's pretrained `inception_v3`
/// (`transform_input=True`, `aux_logits=True`). Use this when loading weights
/// converted from torchvision together with ImageNet normalization.
pub fn v3(p: &nn::Path, nclasses: i64) -> InceptionV3 {
    v3_with(p, nclasses, Default::default())
}

/// InceptionV3 without input transformation, matching torchvision's
/// from-scratch `Inception3()` constructor (`transform_input=False`,
/// `aux_logits=True`).
pub fn v3_no_transform_input(p: &nn::Path, nclasses: i64) -> InceptionV3 {
    v3_with(p, nclasses, InceptionV3Config { transform_input: false, aux_logits: true })
}

/// InceptionV3 with an explicit configuration.
pub fn v3_with(p: &nn::Path, nclasses: i64, config: InceptionV3Config) -> InceptionV3 {
    let pre_aux = nn::seq_t()
        .add(conv_bn(p / "Conv2d_1a_3x3", 3, 32, 3, 0, 2))
        .add(conv_bn(p / "Conv2d_2a_3x3", 32, 32, 3, 0, 1))
        .add(conv_bn(p / "Conv2d_2b_3x3", 32, 64, 3, 1, 1))
        .add_fn(|xs| max_pool2d(xs, 3, 2))
        .add(conv_bn(p / "Conv2d_3b_1x1", 64, 80, 1, 0, 1))
        .add(conv_bn(p / "Conv2d_4a_3x3", 80, 192, 3, 0, 1))
        .add_fn(|xs| max_pool2d(xs, 3, 2))
        .add(inception_a(p / "Mixed_5b", 192, 32))
        .add(inception_a(p / "Mixed_5c", 256, 64))
        .add(inception_a(p / "Mixed_5d", 288, 64))
        .add(inception_b(p / "Mixed_6a", 288))
        .add(inception_c(p / "Mixed_6b", 768, 128))
        .add(inception_c(p / "Mixed_6c", 768, 160))
        .add(inception_c(p / "Mixed_6d", 768, 160))
        .add(inception_c(p / "Mixed_6e", 768, 192));
    let aux = config.aux_logits.then(|| inception_aux(p / "AuxLogits", nclasses));
    let post_aux = nn::seq_t()
        .add(inception_d(p / "Mixed_7a", 768))
        .add(inception_e(p / "Mixed_7b", 1280))
        .add(inception_e(p / "Mixed_7c", 2048))
        .add_fn_t(|xs, train| xs.adaptive_avg_pool2d([1, 1]).dropout(0.5, train).flat_view())
        // The init loop covers Linear weights too (default stddev 0.1); the
        // bias keeps the standard Linear init, as in torchvision.
        .add(nn::linear(
            p / "fc",
            2048,
            nclasses,
            nn::LinearConfig { ws_init: trunc_normal(0.1), ..Default::default() },
        ));
    let transform = config
        .transform_input
        .then(|| transform_constants(p.device(), crate::Kind::Float));
    InceptionV3 { transform, pre_aux, post_aux, aux }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_input_matches_torchvision() {
        let xs = Tensor::ones([1, 3, 2, 2], (crate::Kind::Float, crate::Device::Cpu));
        let ys = transform_input(&xs);
        // Per-channel: x * (std_c / 0.5) + (mean_c - 0.5) / 0.5 for x = 1.
        let expected = [0.229f64 / 0.5 - 0.03, 0.224 / 0.5 - 0.088, 0.225 / 0.5 - 0.188];
        for (c, want) in expected.iter().enumerate() {
            let got: f64 = ys.select(1, c as i64).mean(crate::Kind::Double).try_into().unwrap();
            assert!((got - want).abs() < 1e-6, "channel {c}: got {got}, want {want}");
        }
    }
}
