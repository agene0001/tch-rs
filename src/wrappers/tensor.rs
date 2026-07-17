use super::stream::ReadSeekAdapter;
use super::utils::{path_to_cstring, ptr_to_string};
use super::{
    device::Device,
    kind,
    kind::Kind,
};
use crate::TchError;
use libc::{c_char, c_int, c_void};
use std::borrow::Borrow;
use std::io::{Read, Seek, Write};
use std::path::Path;
use torch_sys::io::ReadStream;
use torch_sys::*;

/// A tensor object.
#[must_use]
pub struct Tensor {
    pub(super) c_tensor: *mut C_tensor,
}

unsafe impl Send for Tensor {}

pub extern "C" fn add_callback(data: *mut c_void, name: *const c_char, c_tensor: *mut C_tensor) {
    let name = unsafe { std::ffi::CStr::from_ptr(name).to_str().unwrap() };
    let name = name.replace('|', ".");
    let v: &mut Vec<(String, Tensor)> = unsafe { &mut *(data as *mut Vec<(String, Tensor)>) };
    v.push((name, Tensor { c_tensor }))
}

impl Tensor {
    /// Creates a new tensor.
    pub fn new() -> Tensor {
        let c_tensor = unsafe_torch!(at_new_tensor());
        Tensor { c_tensor }
    }

    /// Creates a new tensor from the pointer to an existing C++ tensor.
    ///
    /// # Safety
    ///
    /// The caller must ensures that the pointer outlives the Rust
    /// object.
    pub unsafe fn from_ptr(c_tensor: *mut C_tensor) -> Self {
        Self { c_tensor }
    }

    /// Creates a new tensor from the pointer to an existing C++ tensor.
    ///
    /// # Safety
    ///
    /// A shallow copy of the pointer is made so there is no need for
    /// this pointer to remain valid for the whole lifetime of the Rust
    /// object.
    pub unsafe fn clone_from_ptr(c_tensor: *mut C_tensor) -> Self {
        // SAFETY: the caller guarantees `c_tensor` points to a valid tensor.
        let c_tensor = unsafe { at_shallow_clone(c_tensor) };
        crate::wrappers::utils::read_and_clean_error().unwrap();
        Self { c_tensor }
    }

    /// Returns a pointer to the underlying C++ tensor.
    ///
    /// The caller must ensures that the Rust tensor object outlives
    /// this pointer.
    pub fn as_ptr(&self) -> *const C_tensor {
        self.c_tensor
    }

    /// Returns a mutable pointer to the underlying C++ tensor.
    ///
    /// The caller must ensures that the Rust tensor object outlives
    /// this pointer.
    pub fn as_mut_ptr(&mut self) -> *mut C_tensor {
        self.c_tensor
    }

    /// Returns the number of dimension of the tensor.
    pub fn dim(&self) -> usize {
        unsafe_torch!(at_dim(self.c_tensor))
    }

    /// Returns the shape of the input tensor.
    pub fn size(&self) -> Vec<i64> {
        let dim = unsafe_torch!(at_dim(self.c_tensor));
        let mut sz = vec![0i64; dim];
        unsafe_torch!(at_shape(self.c_tensor, sz.as_mut_ptr()));
        sz
    }

    /// Reads the tensor shape into a stack-allocated array, erroring out on
    /// rank mismatch. This avoids the heap allocation that `size()` performs.
    fn size_n<const N: usize>(&self) -> Result<[i64; N], TchError> {
        const DIM_NAMES: [&str; 7] = [
            "zero dims", "one dim", "two dims", "three dims", "four dims", "five dims", "six dims",
        ];
        let dim = unsafe_torch!(at_dim(self.c_tensor));
        if dim != N {
            return Err(TchError::Shape(format!(
                "expected {}, got {:?}",
                DIM_NAMES.get(N).copied().unwrap_or("N dims"),
                self.size()
            )));
        }
        let mut sz = [0i64; N];
        unsafe_torch!(at_shape(self.c_tensor, sz.as_mut_ptr()));
        Ok(sz)
    }

    /// Reads the tensor strides into a stack-allocated array, erroring out on
    /// rank mismatch. This avoids the heap allocation that `stride()` performs.
    fn stride_n<const N: usize>(&self) -> Result<[i64; N], TchError> {
        const DIM_NAMES: [&str; 7] = [
            "zero dims", "one dim", "two dims", "three dims", "four dims", "five dims", "six dims",
        ];
        let dim = unsafe_torch!(at_dim(self.c_tensor));
        if dim != N {
            return Err(TchError::Shape(format!(
                "expected {}, got {:?}",
                DIM_NAMES.get(N).copied().unwrap_or("N dims"),
                self.size()
            )));
        }
        let mut sz = [0i64; N];
        unsafe_torch!(at_stride(self.c_tensor, sz.as_mut_ptr()));
        Ok(sz)
    }

    /// Returns the size of the tensor along a single dimension, avoiding the
    /// heap allocation that `size()` performs.
    pub(crate) fn size_at(&self, dim: usize) -> i64 {
        let ndim = unsafe_torch!(at_dim(self.c_tensor));
        assert!(dim < ndim, "size_at: dim {dim} out of range for a {ndim}-d tensor");
        if ndim <= 8 {
            let mut sz = [0i64; 8];
            unsafe_torch!(at_shape(self.c_tensor, sz.as_mut_ptr()));
            sz[dim]
        } else {
            self.size()[dim]
        }
    }

    /// Returns the tensor size for single dimension tensors.
    pub fn size1(&self) -> Result<i64, TchError> {
        let [s0] = self.size_n::<1>()?;
        Ok(s0)
    }

    /// Returns the tensor sizes for two dimension tensors.
    pub fn size2(&self) -> Result<(i64, i64), TchError> {
        let [s0, s1] = self.size_n::<2>()?;
        Ok((s0, s1))
    }

    /// Returns the tensor sizes for three dimension tensors.
    pub fn size3(&self) -> Result<(i64, i64, i64), TchError> {
        let [s0, s1, s2] = self.size_n::<3>()?;
        Ok((s0, s1, s2))
    }

    /// Returns the tensor sizes for four dimension tensors.
    pub fn size4(&self) -> Result<(i64, i64, i64, i64), TchError> {
        let [s0, s1, s2, s3] = self.size_n::<4>()?;
        Ok((s0, s1, s2, s3))
    }

    /// Returns the tensor sizes for five dimension tensors.
    pub fn size5(&self) -> Result<(i64, i64, i64, i64, i64), TchError> {
        let [s0, s1, s2, s3, s4] = self.size_n::<5>()?;
        Ok((s0, s1, s2, s3, s4))
    }

    /// Returns the tensor sizes for six dimension tensors.
    pub fn size6(&self) -> Result<(i64, i64, i64, i64, i64, i64), TchError> {
        let [s0, s1, s2, s3, s4, s5] = self.size_n::<6>()?;
        Ok((s0, s1, s2, s3, s4, s5))
    }

    /// Returns the stride of the input tensor.
    pub fn stride(&self) -> Vec<i64> {
        let dim = unsafe_torch!(at_dim(self.c_tensor));
        let mut sz = vec![0i64; dim];
        unsafe_torch!(at_stride(self.c_tensor, sz.as_mut_ptr()));
        sz
    }

    /// Returns the tensor strides for single dimension tensors.
    pub fn stride1(&self) -> Result<i64, TchError> {
        let [s0] = self.stride_n::<1>()?;
        Ok(s0)
    }

    /// Returns the tensor strides for two dimension tensors.
    pub fn stride2(&self) -> Result<(i64, i64), TchError> {
        let [s0, s1] = self.stride_n::<2>()?;
        Ok((s0, s1))
    }

    /// Returns the tensor strides for three dimension tensors.
    pub fn stride3(&self) -> Result<(i64, i64, i64), TchError> {
        let [s0, s1, s2] = self.stride_n::<3>()?;
        Ok((s0, s1, s2))
    }

    /// Returns the tensor strides for four dimension tensors.
    pub fn stride4(&self) -> Result<(i64, i64, i64, i64), TchError> {
        let [s0, s1, s2, s3] = self.stride_n::<4>()?;
        Ok((s0, s1, s2, s3))
    }

    /// Returns the tensor strides for five dimension tensors.
    pub fn stride5(&self) -> Result<(i64, i64, i64, i64, i64), TchError> {
        let [s0, s1, s2, s3, s4] = self.stride_n::<5>()?;
        Ok((s0, s1, s2, s3, s4))
    }

    /// Returns the tensor strides for six dimension tensors.
    pub fn stride6(&self) -> Result<(i64, i64, i64, i64, i64, i64), TchError> {
        let [s0, s1, s2, s3, s4, s5] = self.stride_n::<6>()?;
        Ok((s0, s1, s2, s3, s4, s5))
    }

    /// Returns the kind of elements stored in the input tensor. Returns
    /// an error on undefined tensors and unsupported data types.
    pub fn f_kind(&self) -> Result<Kind, TchError> {
        let kind = unsafe_torch!(at_scalar_type(self.c_tensor));
        Kind::from_c_int(kind)
    }

    /// Returns the kind of elements stored in the input tensor. Panics
    /// an error on undefined tensors and unsupported data types.
    pub fn kind(&self) -> Kind {
        self.f_kind().unwrap()
    }

    /// Returns the device on which the input tensor is located.
    pub fn device(&self) -> Device {
        let device = unsafe_torch!(at_device(self.c_tensor));
        Device::from_c_int(device)
    }

    /// Prints the input tensor.
    ///
    /// Caution: this uses the C++ printer which prints the whole tensor even if
    /// it is very large.
    pub fn print(&self) {
        unsafe_torch!(at_print(self.c_tensor))
    }

    /// Returns a double value on tensors holding a single element. An error is
    /// returned otherwise.
    pub fn f_double_value(&self, idx: &[i64]) -> Result<f64, TchError> {
        Ok(unsafe_torch_err!({
            at_double_value_at_indexes(self.c_tensor, idx.as_ptr(), idx.len() as i32)
        }))
    }

    /// Returns an int value on tensors holding a single element. An error is
    /// returned otherwise.
    pub fn f_int64_value(&self, idx: &[i64]) -> Result<i64, TchError> {
        Ok(unsafe_torch_err!({
            at_int64_value_at_indexes(self.c_tensor, idx.as_ptr(), idx.len() as i32)
        }))
    }

    /// Returns a double value on tensors holding a single element. Panics otherwise.
    pub fn double_value(&self, idx: &[i64]) -> f64 {
        self.f_double_value(idx).unwrap()
    }

    /// Returns an int value on tensors holding a single element. Panics otherwise.
    pub fn int64_value(&self, idx: &[i64]) -> i64 {
        self.f_int64_value(idx).unwrap()
    }

    /// Returns true if gradient are currently tracked for this tensor.
    pub fn requires_grad(&self) -> bool {
        unsafe_torch!(at_requires_grad(self.c_tensor)) != 0
    }

    /// Returns the address of the first element of this tensor.
    pub fn data_ptr(&self) -> *mut c_void {
        unsafe_torch!(at_data_ptr(self.c_tensor))
    }

    /// Returns true if the tensor is defined.
    pub fn defined(&self) -> bool {
        unsafe_torch!(at_defined(self.c_tensor) != 0)
    }

    /// Returns true if the tensor is compatible with MKL-DNN (oneDNN).
    pub fn is_mkldnn(&self) -> bool {
        unsafe_torch!(at_is_mkldnn(self.c_tensor) != 0)
    }

    /// Returns true if the tensor is sparse.
    pub fn is_sparse(&self) -> bool {
        unsafe_torch!(at_is_sparse(self.c_tensor) != 0)
    }

    // Returns true if the tensor if contiguous
    pub fn is_contiguous(&self) -> bool {
        unsafe_torch!(at_is_contiguous(self.c_tensor) != 0)
    }

    /// Zeroes the gradient tensor attached to this tensor if defined.
    pub fn zero_grad(&mut self) {
        let mut grad = self.grad();
        if grad.defined() {
            let _ = grad.detach_().zero_();
        }
    }

    /// Runs the backward pass, populating the gradient tensors for tensors
    /// which gradients are tracked.
    ///
    /// Gradients tracking can be turned on via `set_requires_grad`.
    pub fn f_backward(&self) -> Result<(), TchError> {
        unsafe_torch_err!(at_backward(self.c_tensor, 0, 0));
        Ok(())
    }

    /// Runs the backward pass, populating the gradient tensors for tensors
    /// which gradients are tracked.
    ///
    /// Gradients tracking can be turned on via `set_requires_grad`.
    /// Panics if the C++ api returns an exception.
    pub fn backward(&self) {
        self.f_backward().unwrap()
    }

    /// Clips the total norm of the given gradient tensors in-place, following
    /// `torch.nn.utils.clip_grad_norm_`, and returns the total norm.
    ///
    /// The whole computation happens in a single FFI call using the batched
    /// `_foreach` kernels, and the clip coefficient stays on device: no host
    /// synchronization unless `error_if_nonfinite` is set (which must read
    /// the norm back to raise). `norm_type` may be `f64::INFINITY`.
    pub fn f_clip_grad_norm<T: Borrow<Tensor>>(
        grads: &[T],
        max_norm: f64,
        norm_type: f64,
        error_if_nonfinite: bool,
    ) -> Result<Tensor, TchError> {
        let ptrs: Vec<_> = grads.iter().map(|t| t.borrow().c_tensor).collect();
        let c_tensor = unsafe_torch_err!(torch_sys::at_clip_grad_norm(
            ptrs.as_ptr(),
            ptrs.len() as c_int,
            max_norm,
            norm_type,
            c_int::from(error_if_nonfinite),
        ));
        Ok(Tensor { c_tensor })
    }

    /// One AdamW/Adam step over a homogeneous parameter bucket (same device,
    /// dtype, and step count) via the batched `_foreach` kernels; a single
    /// FFI crossing. `max_exp_avg_sqs` enables amsgrad; `step` is the
    /// post-increment step count used for the bias corrections.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn f_foreach_adam_step(
        params: &[&Tensor],
        grads: &[Tensor],
        exp_avgs: &[&Tensor],
        exp_avg_sqs: &[&Tensor],
        max_exp_avg_sqs: Option<&[&Tensor]>,
        step: i64,
        lr: f64,
        beta1: f64,
        beta2: f64,
        weight_decay: f64,
        eps: f64,
        decoupled_wd: bool,
    ) -> Result<(), TchError> {
        let p: Vec<_> = params.iter().map(|t| t.c_tensor).collect();
        let g: Vec<_> = grads.iter().map(|t| t.c_tensor).collect();
        let m: Vec<_> = exp_avgs.iter().map(|t| t.c_tensor).collect();
        let v: Vec<_> = exp_avg_sqs.iter().map(|t| t.c_tensor).collect();
        let vmax: Option<Vec<_>> =
            max_exp_avg_sqs.map(|ts| ts.iter().map(|t| t.c_tensor).collect());
        let err__ = unsafe {
            torch_sys::at_foreach_adam_step(
                p.as_ptr(),
                g.as_ptr(),
                m.as_ptr(),
                v.as_ptr(),
                vmax.as_ref().map_or(std::ptr::null(), |v| v.as_ptr()),
                p.len() as c_int,
                step,
                lr,
                beta1,
                beta2,
                weight_decay,
                eps,
                c_int::from(decoupled_wd),
            )
        };
        crate::wrappers::utils::ptr_err_to_result(err__)
    }

    /// Zeroes the given tensors (typically gradients) with a single batched
    /// `_foreach_zero_` call.
    pub(crate) fn f_foreach_zero(tensors: &[Tensor]) -> Result<(), TchError> {
        let ptrs: Vec<_> = tensors.iter().map(|t| t.c_tensor).collect();
        let err__ =
            unsafe { torch_sys::at_foreach_zero(ptrs.as_ptr(), ptrs.len() as c_int) };
        crate::wrappers::utils::ptr_err_to_result(err__)
    }

    pub fn f_run_backward<T1, T2>(
        tensors: &[T1],
        inputs: &[T2],
        keep_graph: bool,
        create_graph: bool,
    ) -> Result<Vec<Tensor>, TchError>
    where
        T1: Borrow<Tensor>,
        T2: Borrow<Tensor>,
    {
        let mut outputs = vec![std::ptr::null_mut(); inputs.len()];
        let tensors: Vec<_> = tensors.iter().map(|x| x.borrow().c_tensor).collect();
        let inputs: Vec<_> = inputs.iter().map(|x| x.borrow().c_tensor).collect();
        unsafe_torch_err!(at_run_backward(
            tensors.as_ptr(),
            tensors.len() as c_int,
            inputs.as_ptr(),
            inputs.len() as c_int,
            outputs.as_mut_ptr(),
            keep_graph as c_int,
            create_graph as c_int,
        ));
        Ok(outputs.into_iter().map(|c_tensor| Tensor { c_tensor }).collect())
    }

    pub fn run_backward<T1, T2>(
        tensors: &[T1],
        inputs: &[T2],
        keep_graph: bool,
        create_graph: bool,
    ) -> Vec<Tensor>
    where
        T1: Borrow<Tensor>,
        T2: Borrow<Tensor>,
    {
        Tensor::f_run_backward(tensors, inputs, keep_graph, create_graph).unwrap()
    }

    /// Copies `numel` elements from `self` to `dst`.
    pub fn f_copy_data_u8(&self, dst: &mut [u8], numel: usize) -> Result<(), TchError> {
        let elt_size_in_bytes = self.f_kind()?.elt_size_in_bytes();
        if dst.len() < numel * elt_size_in_bytes {
            return Err(TchError::Shape(format!("slice len < {numel}")));
        }
        unsafe_torch_err!(at_copy_data(
            self.c_tensor,
            dst.as_mut_ptr() as *const c_void,
            numel,
            elt_size_in_bytes,
        ));
        Ok(())
    }

    /// Unscale tensor while checking for infinities.
    ///
    /// `found_inf` is a singleton tensor that is used to record the
    /// presence of infinite values. `inv_scale` is a scalar containing
    /// the inverse scaling factor. This method is only available
    /// for CUDA tensors.
    pub fn f_internal_amp_non_finite_check_and_unscale(
        &mut self,
        found_inf: &mut Tensor,
        inv_scale: &Tensor,
    ) -> Result<(), TchError> {
        unsafe_torch_err!(at__amp_non_finite_check_and_unscale(
            self.c_tensor,
            found_inf.c_tensor,
            inv_scale.c_tensor
        ));

        Ok(())
    }

    /// Unscale tensor while checking for infinities.
    ///
    /// `found_inf` is a singleton tensor that is used to record the
    /// presence of infinite values. `inv_scale` is a scalar containing
    /// the inverse scaling factor. This method is only available
    /// for CUDA tensors.
    pub fn internal_amp_non_finite_check_and_unscale(
        &mut self,
        found_inf: &mut Tensor,
        inv_scale: &Tensor,
    ) {
        self.f_internal_amp_non_finite_check_and_unscale(found_inf, inv_scale).unwrap()
    }

    /// Copies `numel` elements from `self` to `dst`.
    pub fn copy_data_u8(&self, dst: &mut [u8], numel: usize) {
        self.f_copy_data_u8(dst, numel).unwrap()
    }

    /// Copies `numel` elements from `self` to `dst`.
    pub fn f_copy_data<T: kind::Element>(
        &self,
        dst: &mut [T],
        numel: usize,
    ) -> Result<(), TchError> {
        if T::KIND != self.f_kind()? {
            return Err(TchError::Kind(format!(
                "incoherent elt kind, {:?} != {:?}",
                self.f_kind(),
                T::KIND
            )));
        }
        if dst.len() < numel {
            return Err(TchError::Shape(format!("slice len < {numel}")));
        }
        unsafe_torch_err!(at_copy_data(
            self.c_tensor,
            dst.as_mut_ptr() as *const c_void,
            numel,
            T::KIND.elt_size_in_bytes(),
        ));
        Ok(())
    }

    /// Copies `numel` elements from `self` to `dst`.
    pub fn copy_data<T: kind::Element>(&self, dst: &mut [T], numel: usize) {
        self.f_copy_data(dst, numel).unwrap()
    }

    /// Returns the total number of elements stored in a tensor.
    pub fn numel(&self) -> usize {
        // Avoid the heap allocation that `size()` performs: tensor ranks are
        // tiny, so read the shape into a stack buffer for the common case.
        let dim = unsafe_torch!(at_dim(self.c_tensor));
        if dim <= 8 {
            let mut sz = [0i64; 8];
            unsafe_torch!(at_shape(self.c_tensor, sz.as_mut_ptr()));
            sz[..dim].iter().product::<i64>() as usize
        } else {
            self.size().iter().product::<i64>() as usize
        }
    }

    // This is similar to vec_... but faster as it directly blits the data.
    /// Converts a slice to a tensor.
    pub fn f_from_slice<T: kind::Element>(data: &[T]) -> Result<Tensor, TchError> {
        let data_len = data.len();
        let data = data.as_ptr() as *const c_void;
        let c_tensor = unsafe_torch_err!(at_tensor_of_data(
            data,
            [data_len as i64].as_ptr(),
            1,
            T::KIND.elt_size_in_bytes(),
            T::KIND.c_int(),
        ));
        Ok(Tensor { c_tensor })
    }

    /// Converts a slice to a tensor.
    pub fn from_slice<T: kind::Element>(data: &[T]) -> Tensor {
        Self::f_from_slice(data).unwrap()
    }

    /// Converts some byte data to a tensor with some specified kind and shape.
    pub fn f_from_data_size(data: &[u8], size: &[i64], kind: Kind) -> Result<Tensor, TchError> {
        let elt_size_in_bytes = kind.elt_size_in_bytes();
        // The C side memcpys numel * elt_size bytes out of `data`, so the
        // slice length has to be validated here: a short buffer (e.g. from a
        // truncated .npy/.npz file) would otherwise be an out-of-bounds read.
        let numel = size.iter().try_fold(1usize, |acc, &s| {
            usize::try_from(s).ok().and_then(|s| acc.checked_mul(s))
        });
        let expected = numel.and_then(|n| n.checked_mul(elt_size_in_bytes));
        match expected {
            Some(expected) if data.len() >= expected => {}
            _ => {
                return Err(TchError::Shape(format!(
                    "{} bytes of data do not fit a tensor of shape {size:?} and kind {kind:?}",
                    data.len()
                )))
            }
        }
        let data = data.as_ptr() as *const c_void;
        let c_tensor = unsafe_torch_err!(at_tensor_of_data(
            data,
            size.as_ptr(),
            size.len(),
            elt_size_in_bytes,
            kind.c_int(),
        ));
        Ok(Tensor { c_tensor })
    }

    /// Creates a tensor from data that is assumed to be initialized.
    /// Resize operations are not allowed on this tensor without copying the data first.
    /// An empty strides slice will result in using the default strides.
    /// # Safety
    /// The tensor does NOT copy or take ownership of `data`:
    /// - `data` must point to at least `numel * kind element size` bytes of
    ///   initialized memory (per `size`/`strides`), and must remain valid for
    ///   the whole lifetime of the returned tensor *and* of every view or
    ///   shallow clone derived from it.
    /// - In-place operations on the returned tensor write through `data`, so
    ///   the buffer must not be aliased by Rust references while the tensor
    ///   (or any derived view) is alive.
    pub unsafe fn f_from_blob(
        data: *const u8,
        size: &[i64],
        strides: &[i64],
        kind: Kind,
        device: Device,
    ) -> Result<Tensor, TchError> {
        let data = data as *const c_void;
        #[allow(unused_unsafe)]
        let c_tensor = unsafe_torch_err!(at_tensor_of_blob(
            data,
            size.as_ptr(),
            size.len(),
            strides.as_ptr(),
            strides.len(),
            kind.c_int(),
            device.c_int()
        ));
        Ok(Tensor { c_tensor })
    }

    /// Creates a tensor from data that is assumed to be initialized.
    /// Resize operations are not allowed on this tensor without copying the data first.
    /// An empty strides slice will result in using the default strides.
    /// # Safety
    /// See [`Tensor::f_from_blob`]: `data` is not copied, must outlive the
    /// returned tensor and all views of it, and must not be aliased while
    /// the tensor is alive.
    pub unsafe fn from_blob(
        data: *const u8,
        size: &[i64],
        strides: &[i64],
        kind: Kind,
        device: Device,
    ) -> Tensor {
        // SAFETY: forwarded to the caller, see f_from_blob's contract.
        unsafe { Self::f_from_blob(data, size, strides, kind, device) }.unwrap()
    }

    /// Converts some byte data to a tensor with some specified kind and shape.
    pub fn from_data_size(data: &[u8], size: &[i64], kind: Kind) -> Tensor {
        Self::f_from_data_size(data, size, kind).unwrap()
    }

    /// Returns a new tensor that share storage with the input tensor.
    pub fn shallow_clone(&self) -> Tensor {
        let c_tensor = unsafe_torch!(at_shallow_clone(self.c_tensor));
        Tensor { c_tensor }
    }

    /// Gets the sub-tensor at the given index.
    pub fn f_get(&self, index: i64) -> Result<Tensor, TchError> {
        let c_tensor = unsafe_torch_err!(at_get(self.c_tensor, index));
        Ok(Tensor { c_tensor })
    }

    /// Gets the sub-tensor at the given index.
    pub fn get(&self, index: i64) -> Tensor {
        self.f_get(index).unwrap()
    }

    /// Copies values from the argument tensor to the input tensor.
    pub fn f_copy_(&mut self, src: &Tensor) -> Result<(), TchError> {
        unsafe_torch_err!(at_copy_(self.c_tensor, src.c_tensor));
        Ok(())
    }

    /// Copies values from the argument tensor to the input tensor.
    pub fn copy_(&mut self, src: &Tensor) {
        self.f_copy_(src).unwrap()
    }

    /// Loads a tensor from a file.
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn load<T: AsRef<Path>>(path: T) -> Result<Tensor, TchError> {
        let path = path_to_cstring(path)?;
        let c_tensor = unsafe_torch_err!(at_load(path.as_ptr()));
        Ok(Tensor { c_tensor })
    }

    /// Loads a tensor from a stream.
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn load_from_stream<T: Read + Seek>(stream: T) -> Result<Tensor, TchError> {
        let adapter = ReadSeekAdapter::new(stream);
        let boxed_stream: Box<Box<dyn ReadStream>> = Box::new(Box::new(adapter));
        let c_tensor =
            unsafe_torch_err!(at_load_from_stream(Box::into_raw(boxed_stream) as *mut c_void,));
        Ok(Tensor { c_tensor })
    }

    /// Saves a tensor to a file.
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), TchError> {
        let path = path_to_cstring(path)?;
        unsafe_torch_err!(at_save(self.c_tensor, path.as_ptr()));
        Ok(())
    }

    /// Saves a tensor to a stream.
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn save_to_stream<W: Write>(&self, stream: W) -> Result<(), TchError> {
        let boxed_stream: Box<Box<dyn Write>> = Box::new(Box::new(stream));
        unsafe_torch_err!(at_save_to_stream(
            self.c_tensor,
            Box::into_raw(boxed_stream) as *mut c_void,
        ));
        Ok(())
    }

    /// Saves some named tensors to a file
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn save_multi<S: AsRef<str>, T: AsRef<Tensor>, P: AsRef<Path>>(
        named_tensors: &[(S, T)],
        path: P,
    ) -> Result<(), TchError> {
        let path = path_to_cstring(path)?;
        let c_tensors = named_tensors.iter().map(|nt| nt.1.as_ref().c_tensor).collect::<Vec<_>>();
        let names = named_tensors
            .iter()
            .map(|nt| nt.0.as_ref().replace('.', "|").into_bytes())
            .map(std::ffi::CString::new)
            .collect::<Result<Vec<_>, _>>()?;
        let name_ptrs = names.iter().map(|n| n.as_ptr()).collect::<Vec<_>>();
        unsafe_torch_err!(at_save_multi(
            c_tensors.as_ptr(),
            name_ptrs.as_ptr(),
            names.len() as i32,
            path.as_ptr(),
        ));
        Ok(())
    }

    /// Saves some named tensors to a stream
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn save_multi_to_stream<S: AsRef<str>, T: AsRef<Tensor>, W: Write>(
        named_tensors: &[(S, T)],
        stream: W,
    ) -> Result<(), TchError> {
        let boxed_stream: Box<Box<dyn Write>> = Box::new(Box::new(stream));
        let c_tensors = named_tensors.iter().map(|nt| nt.1.as_ref().c_tensor).collect::<Vec<_>>();
        let names = named_tensors
            .iter()
            .map(|nt| nt.0.as_ref().replace('.', "|").into_bytes())
            .map(std::ffi::CString::new)
            .collect::<Result<Vec<_>, _>>()?;
        let name_ptrs = names.iter().map(|n| n.as_ptr()).collect::<Vec<_>>();
        unsafe_torch_err!(at_save_multi_to_stream(
            c_tensors.as_ptr(),
            name_ptrs.as_ptr(),
            names.len() as i32,
            Box::into_raw(boxed_stream) as *mut c_void,
        ));
        Ok(())
    }

    /// Loads some named tensors from a file
    ///
    /// The file format is the same as the one used for modules in the PyTorch C++ API.
    /// It commonly uses the .ot extension.
    pub fn load_multi<T: AsRef<Path>>(path: T) -> Result<Vec<(String, Tensor)>, TchError> {
        let path = path_to_cstring(path)?;
        let mut v: Vec<(String, Tensor)> = vec![];
        unsafe_torch_err!(at_load_callback(
            path.as_ptr(),
            &mut v as *mut _ as *mut c_void,
            add_callback
        ));
        Ok(v)
    }

    /// Loads some named tensors from a file to a given device
    ///
    /// The file format is the same as the one used for modules in the PyTorch C++ API.
    /// It commonly uses the .ot extension.
    pub fn load_multi_with_device<T: AsRef<Path>>(
        path: T,
        device: Device,
    ) -> Result<Vec<(String, Tensor)>, TchError> {
        let path = path_to_cstring(path)?;
        let mut v: Vec<(String, Tensor)> = vec![];
        unsafe_torch_err!(at_load_callback_with_device(
            path.as_ptr(),
            &mut v as *mut _ as *mut c_void,
            add_callback,
            device.c_int(),
        ));
        Ok(v)
    }

    /// Loads some named tensors from a zip file
    ///
    /// The expected file format is a zip archive containing a data.pkl file describing
    /// the embedded tensors. These are commonly used with the .bin extension to export
    /// PyTorch models and weights using the Python api.
    pub fn loadz_multi<T: AsRef<Path>>(path: T) -> Result<Vec<(String, Tensor)>, TchError> {
        let path = path_to_cstring(path)?;
        let mut v: Vec<(String, Tensor)> = vec![];
        unsafe_torch_err!(at_loadz_callback(
            path.as_ptr(),
            &mut v as *mut _ as *mut c_void,
            add_callback
        ));
        Ok(v)
    }

    /// Loads some named tensors from a zip file to a given device
    ///
    /// The expected file format is a zip archive containing a data.pkl file describing
    /// the embedded tensors. These are commonly used with the .bin extension to export
    /// PyTorch models and weights using the Python api.
    pub fn loadz_multi_with_device<T: AsRef<Path>>(
        path: T,
        device: Device,
    ) -> Result<Vec<(String, Tensor)>, TchError> {
        let path = path_to_cstring(path)?;
        let mut v: Vec<(String, Tensor)> = vec![];
        unsafe_torch_err!(at_loadz_callback_with_device(
            path.as_ptr(),
            &mut v as *mut _ as *mut c_void,
            add_callback,
            device.c_int(),
        ));
        Ok(v)
    }

    /// Loads some named tensors from a stream
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn load_multi_from_stream<T: Read + Seek>(
        stream: T,
    ) -> Result<Vec<(String, Tensor)>, TchError> {
        let adapter = ReadSeekAdapter::new(stream);
        let boxed_stream: Box<Box<dyn ReadStream>> = Box::new(Box::new(adapter));
        let mut v: Vec<(String, Tensor)> = vec![];
        unsafe_torch_err!(at_load_from_stream_callback(
            Box::into_raw(boxed_stream) as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            add_callback,
            false,
            0,
        ));
        Ok(v)
    }

    /// Loads some named tensors from a stream to a given device
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn load_multi_from_stream_with_device<T: Read + Seek>(
        stream: T,
        device: Device,
    ) -> Result<Vec<(String, Tensor)>, TchError> {
        let adapter = ReadSeekAdapter::new(stream);
        let boxed_stream: Box<Box<dyn ReadStream>> = Box::new(Box::new(adapter));
        let mut v: Vec<(String, Tensor)> = vec![];
        unsafe_torch_err!(at_load_from_stream_callback(
            Box::into_raw(boxed_stream) as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            add_callback,
            true,
            device.c_int(),
        ));
        Ok(v)
    }

    /// Returns a string representation for the tensor.
    ///
    /// The representation will contain all the tensor element hence may be huge for
    /// large tensors.
    pub fn to_string(&self, lw: i64) -> Result<String, TchError> {
        let s =
            unsafe_torch_err!(ptr_to_string(torch_sys::at_to_string(self.c_tensor, lw as c_int)));
        match s {
            None => Err(TchError::Kind("nullptr representation".to_string())),
            Some(s) => Ok(s),
        }
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        // at_free is a plain `delete` and never sets the error TLS; checking it
        // would cost an extra FFI crossing per drop and could panic in Drop on
        // an error left pending by an unrelated call.
        unsafe { at_free(self.c_tensor) }
    }
}

fn autocast_clear_cache() {
    unsafe_torch!(at_autocast_clear_cache())
}

fn autocast_decrement_nesting() -> isize {
    unsafe_torch!(at_autocast_decrement_nesting() as isize)
}

fn autocast_increment_nesting() -> isize {
    unsafe_torch!(at_autocast_increment_nesting() as isize)
}

// Autocast device-type codes shared with the C shims.
fn autocast_device_type_c(device: Device) -> libc::c_int {
    match device {
        Device::Cpu => 0,
        Device::Cuda(_) => 1,
        Device::Mps => 2,
        Device::Vulkan => panic!("autocast is not supported on vulkan devices"),
    }
}

fn autocast_is_enabled_for(device_type: libc::c_int) -> bool {
    unsafe_torch!(at_autocast_is_enabled_for(device_type)) != 0
}

fn autocast_set_enabled_for(device_type: libc::c_int, b: bool) -> bool {
    unsafe_torch!(at_autocast_set_enabled_for(device_type, i32::from(b))) != 0
}

fn autocast_get_dtype(device_type: libc::c_int) -> crate::Kind {
    let kind = unsafe_torch!(at_autocast_get_dtype(device_type));
    crate::Kind::from_c_int(kind).expect("unexpected autocast dtype")
}

fn autocast_set_dtype(device_type: libc::c_int, kind: crate::Kind) {
    unsafe_torch!(at_autocast_set_dtype(device_type, kind.c_int()))
}

/// Runs a closure in mixed precision on the given device type, mirroring
/// `torch.autocast(device_type=..., dtype=..., enabled=...)`.
///
/// Only the device *type* matters (all CUDA devices share the autocast
/// state). `dtype` of `None` uses the device type's current autocast dtype —
/// by default fp16 on CUDA/MPS and bf16 on CPU. The previous autocast state
/// is restored when the closure finishes, including on panic.
pub fn autocast_device<T, F>(device: Device, dtype: Option<crate::Kind>, enabled: bool, f: F) -> T
where
    F: FnOnce() -> T,
{
    struct Guard {
        device_type: libc::c_int,
        prev_enabled: bool,
        prev_dtype: crate::Kind,
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            if autocast_decrement_nesting() == 0 {
                autocast_clear_cache();
            }
            autocast_set_enabled_for(self.device_type, self.prev_enabled);
            autocast_set_dtype(self.device_type, self.prev_dtype);
        }
    }

    let device_type = autocast_device_type_c(device);
    let prev_enabled = autocast_is_enabled_for(device_type);
    let prev_dtype = autocast_get_dtype(device_type);
    autocast_set_enabled_for(device_type, enabled);
    autocast_set_dtype(device_type, dtype.unwrap_or(prev_dtype));
    autocast_increment_nesting();
    let _guard = Guard { device_type, prev_enabled, prev_dtype };
    f()
}

/// Runs a closure in mixed precision, targeting the CUDA autocast state
/// (`torch.autocast("cuda")`). Use [`autocast_device`] for CPU or MPS
/// autocasting or to pick an explicit dtype.
pub fn autocast<T, F>(enabled: bool, f: F) -> T
where
    F: FnOnce() -> T,
{
    autocast_device(Device::Cuda(0), None, enabled, f)
}

fn grad_set_enabled(b: bool) -> bool {
    unsafe_torch!(at_grad_set_enabled(i32::from(b)) != 0)
}

/// Runs a closure without keeping track of gradients.
pub fn no_grad<T, F>(f: F) -> T
where
    F: FnOnce() -> T,
{
    // Restore through a guard so a panicking closure does not leave gradient
    // tracking disabled for the rest of the thread.
    let _guard = NoGradGuard { enabled: grad_set_enabled(false) };
    f()
}

/// Runs a closure explicitly keeping track of gradients, this could be
/// run within a no_grad closure for example.
pub fn with_grad<T, F>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let _guard = NoGradGuard { enabled: grad_set_enabled(true) };
    f()
}

/// A RAII guard that prevents gradient tracking until deallocated.
pub struct NoGradGuard {
    enabled: bool,
}

/// Disables gradient tracking, this will be enabled back when the
/// returned value gets deallocated.
/// Note that it is important to bind this to a name like `_guard`
/// and not to `_` as the latter would immediately drop the guard.
/// See <https://internals.rust-lang.org/t/pre-rfc-must-bind/12658/46>
/// for more details.
pub fn no_grad_guard() -> NoGradGuard {
    NoGradGuard { enabled: grad_set_enabled(false) }
}

impl std::convert::AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        let _enabled = grad_set_enabled(self.enabled);
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Reduction {
    /// Do not reduce.
    None,
    /// Mean of losses.
    Mean,
    /// Sum of losses.
    Sum,
    /// Escape hatch in case new options become available.
    Other(i64),
}

impl Reduction {
    // This has to stay in sync with
    // pytorch/aten/src/ATen/core/Reduction.h
    pub fn to_int(self) -> i64 {
        match self {
            Reduction::None => 0,
            Reduction::Mean => 1,
            Reduction::Sum => 2,
            Reduction::Other(i) => i,
        }
    }
}
