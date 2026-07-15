//! Variable initialization.
use crate::{Device, Kind, TchError, Tensor};

/// Number of features as input or output of a layer.
/// In Kaiming initialization, choosing `FanIn` preserves
/// the magnitude of the variance of the weights in the
/// forward pass, choosing `FanOut` preserves this
/// magnitude in the backward pass.
#[derive(Debug, Copy, Clone)]
pub enum FanInOut {
    FanIn,
    FanOut,
}

impl FanInOut {
    /// Compute the fan-in or fan-out value for a weight tensor of
    /// the specified dimensions.
    /// <https://github.com/pytorch/pytorch/blob/dbeacf11820e336e803bb719b7aaaf2125ae4d9c/torch/nn/init.py#L284>
    pub fn for_weight_dims(&self, dims: &[i64]) -> i64 {
        let receptive_field_size: i64 = dims.iter().skip(2).product();
        match &self {
            FanInOut::FanIn => {
                if dims.len() < 2 {
                    1
                } else {
                    dims[1] * receptive_field_size
                }
            }
            FanInOut::FanOut => {
                if dims.is_empty() {
                    1
                } else {
                    dims[0] * receptive_field_size
                }
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum NormalOrUniform {
    Normal,
    Uniform,
}

/// The non-linear function that follows this layer. ReLU is the
/// recommended value.
#[derive(Debug, Copy, Clone)]
pub enum NonLinearity {
    ReLU,
    /// Leaky ReLU parameterized by its negative slope.
    LeakyReLU(f64),
    Linear,
    Sigmoid,
    Tanh,
    SELU,
    ExplicitGain(f64),
}

impl NonLinearity {
    pub fn gain(&self) -> f64 {
        match *self {
            NonLinearity::ReLU => 2f64.sqrt(),
            NonLinearity::LeakyReLU(negative_slope) => {
                (2. / (1. + negative_slope * negative_slope)).sqrt()
            }
            NonLinearity::Tanh => 5. / 3.,
            NonLinearity::Linear | NonLinearity::Sigmoid => 1.,
            NonLinearity::SELU => 0.75,
            NonLinearity::ExplicitGain(g) => g,
        }
    }
}

/// Variable initializations.
#[derive(Debug, Copy, Clone)]
pub enum Init {
    /// Constant value.
    Const(f64),

    /// Random normal with some mean and standard deviation.
    Randn { mean: f64, stdev: f64 },

    /// Uniform initialization between some lower and upper bounds.
    Uniform { lo: f64, up: f64 },

    /// Kaiming uniform initialization.
    /// See "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification"
    /// He, K. et al. (2015). This uses a uniform distribution.
    Kaiming { dist: NormalOrUniform, fan: FanInOut, non_linearity: NonLinearity },

    /// Normal distribution truncated to `[lo, up]`, matching PyTorch's
    /// `nn.init.trunc_normal_(mean, std, a, b)` (values are re-drawn inside
    /// the bounds via the inverse CDF, not clipped after the fact).
    TruncatedNormal { mean: f64, stdev: f64, lo: f64, up: f64 },

    /// Orthogonal initialization
    Orthogonal { gain: f64 },
}

// PyTorch's `nn.Linear`/`nn.Conv*` reset_parameters use
// `kaiming_uniform_(weight, a=math.sqrt(5))`, i.e. a leaky-relu gain of
// sqrt(2 / (1 + 5)) = sqrt(1/3), which yields a bound of 1/sqrt(fan_in).
// 2.23606797749979 is sqrt(5) (sqrt is not const-evaluable).
pub const DEFAULT_KAIMING_UNIFORM: Init = Init::Kaiming {
    dist: NormalOrUniform::Uniform,
    fan: FanInOut::FanIn,
    non_linearity: NonLinearity::LeakyReLU(2.23606797749979),
};

pub const DEFAULT_KAIMING_NORMAL: Init = Init::Kaiming {
    dist: NormalOrUniform::Normal,
    fan: FanInOut::FanIn,
    non_linearity: NonLinearity::ReLU,
};

/// In-place truncated normal, following PyTorch's `_no_grad_trunc_normal_`:
/// draw uniformly in CDF space between the bounds, then map back through the
/// inverse error function. One RNG draw, exactly like PyTorch.
fn f_trunc_normal_(
    tensor: &mut Tensor,
    mean: f64,
    stdev: f64,
    lo: f64,
    up: f64,
) -> Result<(), TchError> {
    let norm_cdf = |x: f64| -> Result<f64, TchError> {
        // std lacks erf; a one-element tensor op keeps the value exact.
        let erf = Tensor::f_from_slice(&[x / std::f64::consts::SQRT_2])?.f_erf()?;
        Ok((1. + f64::try_from(erf)?) / 2.)
    };
    let l = norm_cdf((lo - mean) / stdev)?;
    let u = norm_cdf((up - mean) / stdev)?;
    let _ = tensor.f_uniform_(2. * l - 1., 2. * u - 1.)?;
    let _ = tensor.f_erfinv_()?;
    let _ = tensor.f_mul_scalar_(stdev * std::f64::consts::SQRT_2)?;
    let _ = tensor.f_add_scalar_(mean)?;
    let _ = tensor.f_clamp_(lo, up)?;
    Ok(())
}

/// Creates a new float tensor with the specified shape, device, and initialization.
pub fn f_init(i: Init, dims: &[i64], device: Device, kind: Kind) -> Result<Tensor, TchError> {
    match i {
        Init::Const(cst) => {
            // Optimize the case for which a single C++ call can be done.
            if cst == 0. {
                Tensor::f_zeros(dims, (kind, device))
            } else if (cst - 1.).abs() <= f64::EPSILON {
                Tensor::f_ones(dims, (kind, device))
            } else {
                Tensor::f_full(dims, cst, (kind, device))
            }
        }
        Init::Uniform { lo, up } => Tensor::f_empty(dims, (kind, device))?.f_uniform_(lo, up),
        Init::Randn { mean, stdev } => {
            if mean == 0. && (stdev - 1.).abs() <= f64::EPSILON {
                Tensor::f_randn(dims, (kind, device))
            } else {
                Tensor::f_empty(dims, (kind, device))?.f_normal_(mean, stdev)
            }
        }
        Init::Kaiming { dist, fan, non_linearity } => {
            let fan = fan.for_weight_dims(dims);
            let gain = non_linearity.gain();
            let std = gain / (fan as f64).sqrt();
            match dist {
                NormalOrUniform::Uniform => {
                    let bound = 3f64.sqrt() * std;
                    Tensor::f_empty(dims, (kind, device))?.f_uniform_(-bound, bound)
                }
                NormalOrUniform::Normal => {
                    Tensor::f_empty(dims, (kind, device))?.f_normal_(0., std)
                }
            }
        }
        Init::TruncatedNormal { mean, stdev, lo, up } => {
            let mut t = Tensor::f_empty(dims, (kind, device))?;
            f_trunc_normal_(&mut t, mean, stdev, lo, up)?;
            Ok(t)
        }
        Init::Orthogonal { gain } => {
            if dims.len() < 2 {
                return Err(TchError::Shape(
                    "Only tensors with 2 or more dimensions are supported".to_string(),
                ));
            }
            let rows = dims[0];
            let cols: i64 = dims.iter().skip(1).product();

            let mut flattened =
                Tensor::f_empty([rows, cols], (kind, device))?.f_normal_(0.0, 1.0)?;
            let flattened = if rows < cols { flattened.f_t_()? } else { flattened };

            let (mut q, r) = Tensor::f_linalg_qr(&flattened, "reduced")?;
            let d = r.f_diag(0)?;
            let ph = d.f_sign()?;
            q *= ph;

            let mut q = if rows < cols { q.f_t_()? } else { q };
            crate::no_grad(|| q *= gain);

            // The QR factorization happens on the [rows, cols] flattening;
            // restore the requested shape so e.g. conv weights keep their
            // 4d layout (PyTorch's orthogonal_ does tensor.view_as(q).copy_).
            q.f_contiguous()?.f_reshape(dims)
        }
    }
}

/// Creates a new float tensor with the specified shape, device, and initialization.
pub fn init(i: Init, dims: &[i64], device: Device) -> Tensor {
    f_init(i, dims, device, Kind::Float).unwrap()
}

impl Init {
    /// Re-initializes an existing tensor with the specified initialization
    pub fn set(self, tensor: &mut Tensor) {
        match self {
            Init::Const(cst) => {
                let _ = tensor.fill_(cst);
            }
            Init::Uniform { lo, up } => {
                let _ = tensor.uniform_(lo, up);
            }
            Init::Kaiming { dist, fan, non_linearity } => {
                let fan = fan.for_weight_dims(&tensor.size());
                let gain = non_linearity.gain();
                let std = gain / (fan as f64).sqrt();
                match dist {
                    NormalOrUniform::Uniform => {
                        let bound = 3f64.sqrt() * std;
                        let _ = tensor.uniform_(-bound, bound);
                    }
                    NormalOrUniform::Normal => {
                        let _ = tensor.normal_(0., std);
                    }
                }
            }
            Init::Randn { mean, stdev } => {
                let _ = tensor.normal_(mean, stdev);
            }
            Init::TruncatedNormal { mean, stdev, lo, up } => {
                f_trunc_normal_(tensor, mean, stdev, lo, up).unwrap();
            }
            Init::Orthogonal { gain } => {
                let q =
                    f_init(Init::Orthogonal { gain }, &tensor.size(), tensor.device(), Kind::Float)
                        .unwrap();
                crate::no_grad(|| tensor.view_as(&q).copy_(&q));
            }
        }
    }
}

impl Tensor {
    /// Re-initializes the tensor using the specified initialization.
    pub fn init(&mut self, i: Init) {
        i.set(self)
    }
}
