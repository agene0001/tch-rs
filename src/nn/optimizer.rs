//! Optimizers to be used for gradient-descent based training.
use super::var_store::{VarStore, Variables};
use crate::wrappers::optimizer::COptimizer;
use crate::{Device, Kind, TchError, Tensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// An optimizer to run gradient descent.
#[derive(Debug)]
pub struct Optimizer {
    opt: OptInner,
    variables: Arc<Mutex<Variables>>,
    variables_in_optimizer: usize,
}

/// The optimizer backend: either libtorch's C++ torch::optim (which updates
/// parameters one at a time) or the Rust-side multi-tensor implementation
/// driving the batched `_foreach` kernels.
#[derive(Debug)]
enum OptInner {
    C(COptimizer),
    // Boxed: ForeachOpt carries three HashMaps of per-group overrides and
    // would otherwise dwarf the C variant.
    Foreach(Box<ForeachOpt>),
}

/// Optimizer configurations. These configs can be used to build optimizer.
pub trait OptimizerConfig
where
    Self: std::marker::Sized,
{
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError>;

    /// Builds an optimizer with the specified learning rate handling variables stored in `vs`.
    fn build(self, vs: &VarStore, lr: f64) -> Result<Optimizer, TchError> {
        let mut opt = self.build_copt(lr)?;
        let v = vs.variables_.lock().unwrap();
        for var in &v.trainable_variables {
            opt.add_parameters(&var.tensor, var.group)?;
        }
        Ok(Optimizer {
            opt: OptInner::C(opt),
            variables: vs.variables_.clone(),
            variables_in_optimizer: v.trainable_variables.len(),
        })
    }
}

/// Parameters for the SGD optimizer.
#[derive(Debug, Copy, Clone)]
pub struct Sgd {
    pub momentum: f64,
    pub dampening: f64,
    pub wd: f64,
    pub nesterov: bool,
}

impl Default for Sgd {
    fn default() -> Self {
        Sgd { momentum: 0., dampening: 0., wd: 0., nesterov: false }
    }
}

/// Creates the configuration for a Stochastic Gradient Descent (SGD) optimizer.
pub fn sgd(momentum: f64, dampening: f64, wd: f64, nesterov: bool) -> Sgd {
    Sgd { momentum, dampening, wd, nesterov }
}

impl OptimizerConfig for Sgd {
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError> {
        COptimizer::sgd(lr, self.momentum, self.dampening, self.wd, self.nesterov)
    }
}

/// Parameters for the Adam optimizer.
#[derive(Debug, Copy, Clone)]
pub struct Adam {
    pub beta1: f64,
    pub beta2: f64,
    pub wd: f64,
    pub eps: f64,
    pub amsgrad: bool,
}

impl Default for Adam {
    fn default() -> Self {
        Adam { beta1: 0.9, beta2: 0.999, wd: 0., eps: 1e-8, amsgrad: false }
    }
}

/// Creates the configuration for the Adam optimizer.
pub fn adam(beta1: f64, beta2: f64, wd: f64) -> Adam {
    Adam { beta1, beta2, wd, eps: 1e-8, amsgrad: false }
}

impl Adam {
    pub fn beta1(mut self, b: f64) -> Self {
        self.beta1 = b;
        self
    }

    pub fn beta2(mut self, b: f64) -> Self {
        self.beta2 = b;
        self
    }

    pub fn wd(mut self, w: f64) -> Self {
        self.wd = w;
        self
    }

    pub fn eps(mut self, e: f64) -> Self {
        self.eps = e;
        self
    }

    pub fn amsgrad(mut self, a: bool) -> Self {
        self.amsgrad = a;
        self
    }
}

impl OptimizerConfig for Adam {
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError> {
        COptimizer::adam(lr, self.beta1, self.beta2, self.wd, self.eps, self.amsgrad)
    }
}

/// Parameters for the AdamW optimizer.
#[derive(Debug, Copy, Clone)]
pub struct AdamW {
    pub beta1: f64,
    pub beta2: f64,
    pub wd: f64,
    pub eps: f64,
    pub amsgrad: bool,
}

impl Default for AdamW {
    fn default() -> Self {
        AdamW { beta1: 0.9, beta2: 0.999, wd: 0.01, eps: 1e-8, amsgrad: false }
    }
}

/// Creates the configuration for the AdamW optimizer.
pub fn adamw(beta1: f64, beta2: f64, wd: f64) -> AdamW {
    AdamW { beta1, beta2, wd, eps: 1e-8, amsgrad: false }
}

impl AdamW {
    pub fn beta1(mut self, b: f64) -> Self {
        self.beta1 = b;
        self
    }

    pub fn beta2(mut self, b: f64) -> Self {
        self.beta2 = b;
        self
    }

    pub fn wd(mut self, w: f64) -> Self {
        self.wd = w;
        self
    }

    pub fn eps(mut self, e: f64) -> Self {
        self.eps = e;
        self
    }

    pub fn amsgrad(mut self, a: bool) -> Self {
        self.amsgrad = a;
        self
    }
}

impl OptimizerConfig for AdamW {
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError> {
        COptimizer::adamw(lr, self.beta1, self.beta2, self.wd, self.eps, self.amsgrad)
    }
}

/// Parameters for the RmsProp optimizer.
#[derive(Debug, Copy, Clone)]
pub struct RmsProp {
    pub alpha: f64,
    pub eps: f64,
    pub wd: f64,
    pub momentum: f64,
    pub centered: bool,
}

impl Default for RmsProp {
    fn default() -> Self {
        RmsProp { alpha: 0.99, eps: 1e-8, wd: 0., momentum: 0., centered: false }
    }
}

/// Creates the configuration for the RmsProp optimizer.
pub fn rms_prop(alpha: f64, eps: f64, wd: f64, momentum: f64, centered: bool) -> RmsProp {
    RmsProp { alpha, eps, wd, momentum, centered }
}

impl OptimizerConfig for RmsProp {
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError> {
        COptimizer::rms_prop(lr, self.alpha, self.eps, self.wd, self.momentum, self.centered)
    }
}

/// Per-variable state for the multi-tensor optimizer, allocated lazily on
/// the first step where the variable has a defined gradient.
#[derive(Debug)]
struct ForeachVarState {
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    max_exp_avg_sq: Option<Tensor>,
    step: i64,
    // Cached at state creation for bucketing; parameters are expected to
    // stay on their device/dtype for the optimizer's lifetime (as with
    // torch.optim, whose state also stays put after the first step).
    device: Device,
    kind: Kind,
}

/// Rust-side multi-tensor Adam/AdamW built on the batched `_foreach`
/// kernels: a fixed handful of kernel launches per step and a single FFI
/// crossing per parameter bucket, where libtorch's C++ torch::optim pays
/// several launches per parameter.
#[derive(Debug)]
struct ForeachOpt {
    lr: f64,
    beta1: f64,
    beta2: f64,
    wd: f64,
    eps: f64,
    amsgrad: bool,
    /// true applies AdamW's decoupled weight decay, false Adam's L2 term.
    decoupled_wd: bool,
    lr_group: HashMap<usize, f64>,
    beta1_group: HashMap<usize, f64>,
    wd_group: HashMap<usize, f64>,
    /// Parallel to `Variables::trainable_variables`.
    states: Vec<Option<ForeachVarState>>,
}

impl ForeachOpt {
    /// Runs one optimizer step: bucket the parameters with defined grads by
    /// (group, device, dtype, step count) and do each bucket in one FFI call.
    fn step(&mut self, variables: &Mutex<Variables>) -> Result<(), TchError> {
        // Like torch.optim's step, everything runs under no_grad: the
        // in-place parameter updates must not be recorded by autograd.
        crate::no_grad(|| {
            let v = variables.lock().unwrap();
            let n = v.trainable_variables.len();
            if self.states.len() < n {
                self.states.resize_with(n, || None);
            }
            // One FFI crossing fetches every gradient; undefined grads come
            // back as `None` and their params are skipped, like torch.optim.
            let params: Vec<&Tensor> = v.trainable_variables.iter().map(|v| &v.tensor).collect();
            let grads = Tensor::f_collect_grads(&params)?;
            // Params bucketed by (group, device, kind, step) so each bucket can
            // go through one fused `_foreach` call.
            type Buckets = HashMap<(usize, Device, Kind, i64), Vec<(usize, Tensor)>>;
            let mut buckets: Buckets = HashMap::new();
            for (i, grad) in grads.into_iter().enumerate() {
                let Some(grad) = grad else { continue };
                let var = &v.trainable_variables[i];
                if self.states[i].is_none() {
                    self.states[i] = Some(ForeachVarState {
                        exp_avg: var.tensor.zeros_like(),
                        exp_avg_sq: var.tensor.zeros_like(),
                        max_exp_avg_sq: self.amsgrad.then(|| var.tensor.zeros_like()),
                        step: 0,
                        device: var.tensor.device(),
                        kind: var.tensor.kind(),
                    });
                }
                let state = self.states[i].as_mut().unwrap();
                state.step += 1;
                buckets
                    .entry((var.group, state.device, state.kind, state.step))
                    .or_default()
                    .push((i, grad));
            }
            for ((group, _device, _kind, step), items) in buckets {
                let lr = self.lr_group.get(&group).copied().unwrap_or(self.lr);
                let beta1 = self.beta1_group.get(&group).copied().unwrap_or(self.beta1);
                let wd = self.wd_group.get(&group).copied().unwrap_or(self.wd);
                let (idxs, grads): (Vec<usize>, Vec<Tensor>) = items.into_iter().unzip();
                let params: Vec<&Tensor> =
                    idxs.iter().map(|&i| &v.trainable_variables[i].tensor).collect();
                let states: Vec<&ForeachVarState> =
                    idxs.iter().map(|&i| self.states[i].as_ref().unwrap()).collect();
                let exp_avgs: Vec<&Tensor> = states.iter().map(|s| &s.exp_avg).collect();
                let exp_avg_sqs: Vec<&Tensor> = states.iter().map(|s| &s.exp_avg_sq).collect();
                let max_exp_avg_sqs: Option<Vec<&Tensor>> = self.amsgrad.then(|| {
                    states.iter().map(|s| s.max_exp_avg_sq.as_ref().unwrap()).collect()
                });
                Tensor::f_foreach_adam_step(
                    &params,
                    &grads,
                    &exp_avgs,
                    &exp_avg_sqs,
                    max_exp_avg_sqs.as_deref(),
                    step,
                    lr,
                    beta1,
                    self.beta2,
                    wd,
                    self.eps,
                    self.decoupled_wd,
                )?;
            }
            Ok(())
        })
    }

    /// Resets all gradients to undefined in a single FFI crossing, matching
    /// PyTorch's `zero_grad(set_to_none=True)` default and the C++ optimizer
    /// backend: parameters that receive no gradient in a later backward are
    /// then skipped by `step` instead of being kept in motion by weight
    /// decay and stale momentum applied to a zero gradient.
    fn zero_grad(&self, variables: &Mutex<Variables>) -> Result<(), TchError> {
        let v = variables.lock().unwrap();
        let params: Vec<&Tensor> = v.trainable_variables.iter().map(|v| &v.tensor).collect();
        Tensor::f_zero_grads(&params, true)
    }
}

/// Parameters for the multi-tensor (foreach) AdamW optimizer.
///
/// Numerically equivalent to [`AdamW`] but implemented with the batched
/// `_foreach` kernels — the same default PyTorch's Python optimizers use —
/// instead of libtorch's per-parameter C++ loop, making the optimizer step
/// cost a fixed number of kernel launches rather than several per parameter.
#[derive(Debug, Copy, Clone)]
pub struct ForeachAdamW {
    pub beta1: f64,
    pub beta2: f64,
    pub wd: f64,
    pub eps: f64,
    pub amsgrad: bool,
}

impl Default for ForeachAdamW {
    fn default() -> Self {
        let AdamW { beta1, beta2, wd, eps, amsgrad } = AdamW::default();
        ForeachAdamW { beta1, beta2, wd, eps, amsgrad }
    }
}

/// Creates the configuration for the multi-tensor AdamW optimizer.
pub fn foreach_adamw(beta1: f64, beta2: f64, wd: f64) -> ForeachAdamW {
    ForeachAdamW { beta1, beta2, wd, ..Default::default() }
}

/// Parameters for the multi-tensor (foreach) Adam optimizer; see
/// [`ForeachAdamW`]. `wd` is Adam's L2 penalty, like [`Adam`].
#[derive(Debug, Copy, Clone)]
pub struct ForeachAdam {
    pub beta1: f64,
    pub beta2: f64,
    pub wd: f64,
    pub eps: f64,
    pub amsgrad: bool,
}

impl Default for ForeachAdam {
    fn default() -> Self {
        let Adam { beta1, beta2, wd, eps, amsgrad } = Adam::default();
        ForeachAdam { beta1, beta2, wd, eps, amsgrad }
    }
}

/// Creates the configuration for the multi-tensor Adam optimizer.
pub fn foreach_adam(beta1: f64, beta2: f64, wd: f64) -> ForeachAdam {
    ForeachAdam { beta1, beta2, wd, ..Default::default() }
}

fn build_foreach(
    vs: &VarStore,
    lr: f64,
    beta1: f64,
    beta2: f64,
    wd: f64,
    eps: f64,
    amsgrad: bool,
    decoupled_wd: bool,
) -> Result<Optimizer, TchError> {
    let v = vs.variables_.lock().unwrap();
    Ok(Optimizer {
        opt: OptInner::Foreach(Box::new(ForeachOpt {
            lr,
            beta1,
            beta2,
            wd,
            eps,
            amsgrad,
            decoupled_wd,
            lr_group: HashMap::new(),
            beta1_group: HashMap::new(),
            wd_group: HashMap::new(),
            states: Vec::new(),
        })),
        variables: vs.variables_.clone(),
        variables_in_optimizer: v.trainable_variables.len(),
    })
}

impl OptimizerConfig for ForeachAdamW {
    fn build_copt(&self, _lr: f64) -> Result<COptimizer, TchError> {
        Err(TchError::Convert(
            "ForeachAdamW has no C++ optimizer; use OptimizerConfig::build".to_string(),
        ))
    }

    fn build(self, vs: &VarStore, lr: f64) -> Result<Optimizer, TchError> {
        build_foreach(vs, lr, self.beta1, self.beta2, self.wd, self.eps, self.amsgrad, true)
    }
}

impl OptimizerConfig for ForeachAdam {
    fn build_copt(&self, _lr: f64) -> Result<COptimizer, TchError> {
        Err(TchError::Convert(
            "ForeachAdam has no C++ optimizer; use OptimizerConfig::build".to_string(),
        ))
    }

    fn build(self, vs: &VarStore, lr: f64) -> Result<Optimizer, TchError> {
        build_foreach(vs, lr, self.beta1, self.beta2, self.wd, self.eps, self.amsgrad, false)
    }
}

impl Optimizer {
    fn add_missing_variables(&mut self) {
        let v = self.variables.lock().unwrap();
        if v.trainable_variables.len() > self.variables_in_optimizer {
            if let OptInner::C(opt) = &mut self.opt {
                for var in &v.trainable_variables[self.variables_in_optimizer..] {
                    opt.add_parameters(&var.tensor, var.group).unwrap();
                }
            }
            // The foreach backend picks new variables up lazily in step().
            self.variables_in_optimizer = v.trainable_variables.len();
        }
    }

    /// Zeroes the gradient for the tensors tracked by this optimizer.
    pub fn zero_grad(&mut self) {
        self.add_missing_variables();
        self.zero_grad_inner()
    }

    /// Clips gradient value at some specified maximum value.
    pub fn clip_grad_value(&self, max: f64) {
        // Match PyTorch's clip_grad_value_, which runs under no_grad: an
        // in-place clamp on a grad that itself requires grad (create_graph)
        // must not be recorded into the autograd graph.
        crate::no_grad(|| {
            let v = self.variables.lock().unwrap();
            for var in v.trainable_variables.iter() {
                let mut grad = var.tensor.grad();
                if grad.defined() {
                    let _t = grad.clamp_(-max, max);
                }
            }
        })
    }

    /// Clips gradient L2 norm over all trainable parameters.
    ///
    /// The norm is computed over all gradients together, as if they were
    /// concatenated into a single vector.
    pub fn clip_grad_norm(&self, max: f64) {
        let _total_norm = self.clip_grad_norm_with(max, 2., false).unwrap();
    }

    /// Clips the gradients' total norm over all trainable parameters,
    /// following `torch.nn.utils.clip_grad_norm_`, and returns the total
    /// norm.
    ///
    /// The norm of the given `norm_type` (which may be `f64::INFINITY`) is
    /// computed over all gradients together, as if they were concatenated
    /// into a single vector. Everything runs in a single FFI call using the
    /// batched `_foreach` kernels with the clip coefficient kept on device,
    /// so no host synchronization occurs — unless `error_if_nonfinite` is
    /// set, which reads the norm back to error on NaN/inf totals like
    /// PyTorch.
    pub fn clip_grad_norm_with(
        &self,
        max: f64,
        norm_type: f64,
        error_if_nonfinite: bool,
    ) -> Result<Tensor, TchError> {
        // Match PyTorch's clip_grad_norm_, which runs under no_grad: the
        // in-place mul on a grad that itself requires grad (create_graph)
        // must not be recorded into the autograd graph.
        crate::no_grad(|| {
            let v = self.variables.lock().unwrap();
            let mut grads = Vec::with_capacity(v.trainable_variables.len());
            for var in v.trainable_variables.iter() {
                let grad = var.tensor.grad();
                if grad.defined() {
                    grads.push(grad);
                }
            }
            Tensor::f_clip_grad_norm(&grads, max, norm_type, error_if_nonfinite)
        })
    }

    /// Performs an optimization step, updating the tracked tensors based on their gradients.
    pub fn step(&mut self) {
        self.add_missing_variables();
        self.step_inner()
    }

    fn step_inner(&mut self) {
        match &mut self.opt {
            OptInner::C(opt) => opt.step().unwrap(),
            OptInner::Foreach(opt) => opt.step(&self.variables).unwrap(),
        }
    }

    fn zero_grad_inner(&mut self) {
        match &mut self.opt {
            OptInner::C(opt) => opt.zero_grad().unwrap(),
            OptInner::Foreach(opt) => opt.zero_grad(&self.variables).unwrap(),
        }
    }

    /// Applies a backward step pass, update the gradients, and performs an optimization step.
    pub fn backward_step(&mut self, loss: &Tensor) {
        self.add_missing_variables();
        self.zero_grad_inner();
        loss.backward();
        self.step_inner()
    }

    /// Applies a backward step pass, update the gradients, and performs an optimization step.
    ///
    /// The gradients are clipped based on `max` before being applied.
    pub fn backward_step_clip(&mut self, loss: &Tensor, max: f64) {
        self.add_missing_variables();
        self.zero_grad_inner();
        loss.backward();
        self.clip_grad_value(max);
        self.step_inner()
    }

    /// Applies a backward step pass, update the gradients, and performs an optimization step.
    ///
    /// The gradients L2 norm is clipped based on `max`.
    pub fn backward_step_clip_norm(&mut self, loss: &Tensor, max: f64) {
        self.add_missing_variables();
        self.zero_grad_inner();
        loss.backward();
        self.clip_grad_norm(max);
        self.step_inner()
    }

    /// Sets the optimizer learning rate.
    pub fn set_lr(&mut self, lr: f64) {
        match &mut self.opt {
            OptInner::C(opt) => opt.set_learning_rate(lr).unwrap(),
            OptInner::Foreach(opt) => {
                // Like the C++ path, setting the global rate applies to every
                // group: clear any per-group overrides.
                opt.lr = lr;
                opt.lr_group.clear();
            }
        }
    }

    /// Sets the optimizer momentum.
    ///
    /// For SGD and RMSprop this sets the actual momentum parameter. **For
    /// Adam and AdamW there is no momentum; this sets `beta1` instead**, the
    /// same as editing `param_groups[...]["betas"]` in PyTorch. Note that the
    /// bias-correction terms (`1 - beta1^step`) are recomputed with the new
    /// value for the already-accumulated step count, so driving an SGD-style
    /// momentum schedule against an Adam optimizer will silently change the
    /// scale of every subsequent update.
    pub fn set_momentum(&mut self, m: f64) {
        match &mut self.opt {
            OptInner::C(opt) => opt.set_momentum(m).unwrap(),
            OptInner::Foreach(opt) => {
                opt.beta1 = m;
                opt.beta1_group.clear();
            }
        }
    }

    /// Sets the optimizer learning rate for a parameter group.
    pub fn set_lr_group(&mut self, group: usize, lr: f64) {
        match &mut self.opt {
            OptInner::C(opt) => opt.set_learning_rate_group(group, lr).unwrap(),
            OptInner::Foreach(opt) => {
                opt.lr_group.insert(group, lr);
            }
        }
    }

    /// Sets the optimizer momentum for a parameter group.
    ///
    /// For Adam/AdamW this sets `beta1`, not a momentum parameter — see
    /// [`Optimizer::set_momentum`] for the caveats.
    pub fn set_momentum_group(&mut self, group: usize, m: f64) {
        match &mut self.opt {
            OptInner::C(opt) => opt.set_momentum_group(group, m).unwrap(),
            OptInner::Foreach(opt) => {
                opt.beta1_group.insert(group, m);
            }
        }
    }

    /// Returns all the trainable variables for this optimizer.
    pub fn trainable_variables(&self) -> Vec<Tensor> {
        let variables = self.variables.lock().unwrap();
        variables.trainable_variables.iter().map(|v| v.tensor.shallow_clone()).collect()
    }

    /// Sets the optimizer weight decay, clearing any per-group overrides.
    pub fn set_weight_decay(&mut self, weight_decay: f64) {
        match &mut self.opt {
            OptInner::C(opt) => opt.set_weight_decay(weight_decay).unwrap(),
            OptInner::Foreach(opt) => {
                opt.wd = weight_decay;
                opt.wd_group.clear();
            }
        }
    }

    /// Sets the optimizer weight decay for a variable group, leaving the
    /// other groups on their current value.
    pub fn set_weight_decay_group(&mut self, group: usize, weight_decay: f64) {
        match &mut self.opt {
            OptInner::C(opt) => opt.set_weight_decay_group(group, weight_decay).unwrap(),
            OptInner::Foreach(opt) => {
                opt.wd_group.insert(group, weight_decay);
            }
        }
    }
}
