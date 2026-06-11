//! Implement various ops traits for tensors
use super::Tensor;
use crate::{Kind, Scalar};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

fn id<T>(v: T) -> T {
    v
}

fn neg(t: Tensor) -> Tensor {
    t.neg()
}

// `scalar / tensor` as a single division, with the scalar promoted the way
// PyTorch promotes wrapped numbers: it adopts the tensor's kind when that is
// floating-point or complex, and falls back to Float otherwise so that
// integer and bool tensors get true division (`2 / int_tensor` -> floats)
// instead of erroring out like `pow(-1)` does on integer tensors.
fn rdiv<S: Into<Scalar>>(lhs: S, rhs: &Tensor) -> Tensor {
    let kind = match rhs.kind() {
        kind @ (Kind::Half
        | Kind::BFloat16
        | Kind::Float
        | Kind::Double
        | Kind::ComplexHalf
        | Kind::ComplexFloat
        | Kind::ComplexDouble) => kind,
        _ => Kind::Float,
    };
    Tensor::full([0i64; 0], lhs, (kind, rhs.device())).g_div(rhs)
}

macro_rules! impl_op {
    ($trait:ident, $func:ident, $op:ident) => {
        impl $trait<Tensor> for Tensor {
            type Output = Tensor;

            fn $func(self, rhs: Tensor) -> Self::Output {
                self.$op(&rhs)
            }
        }

        impl $trait<&Tensor> for Tensor {
            type Output = Tensor;

            fn $func(self, rhs: &Tensor) -> Self::Output {
                self.$op(rhs)
            }
        }

        impl<'a> $trait<&Tensor> for &'a Tensor {
            type Output = Tensor;

            fn $func(self, rhs: &Tensor) -> Self::Output {
                self.$op(rhs)
            }
        }

        impl $trait<Tensor> for &Tensor {
            type Output = Tensor;

            fn $func(self, rhs: Tensor) -> Self::Output {
                self.$op(&rhs)
            }
        }
    };
}

impl<S> Add<S> for &Tensor
where
    S: Into<Scalar>,
{
    type Output = Tensor;

    fn add(self, rhs: S) -> Self::Output {
        self.g_add_scalar(rhs)
    }
}

impl<S> Add<S> for Tensor
where
    S: Into<Scalar>,
{
    type Output = Tensor;

    fn add(self, rhs: S) -> Self::Output {
        (&self).add(rhs)
    }
}

impl<S> Sub<S> for &Tensor
where
    S: Into<Scalar>,
{
    type Output = Tensor;

    fn sub(self, rhs: S) -> Self::Output {
        self.g_sub_scalar(rhs)
    }
}

impl<S> Sub<S> for Tensor
where
    S: Into<Scalar>,
{
    type Output = Tensor;

    fn sub(self, rhs: S) -> Self::Output {
        (&self).sub(rhs)
    }
}

impl<S> Mul<S> for &Tensor
where
    S: Into<Scalar>,
{
    type Output = Tensor;

    fn mul(self, rhs: S) -> Self::Output {
        self.g_mul_scalar(rhs)
    }
}

impl<S> Mul<S> for Tensor
where
    S: Into<Scalar>,
{
    type Output = Tensor;

    fn mul(self, rhs: S) -> Self::Output {
        (&self).mul(rhs)
    }
}

impl<S> Div<S> for &Tensor
where
    S: Into<Scalar>,
{
    type Output = Tensor;

    fn div(self, rhs: S) -> Self::Output {
        self.g_div_scalar(rhs)
    }
}

impl<S> Div<S> for Tensor
where
    S: Into<Scalar>,
{
    type Output = Tensor;

    fn div(self, rhs: S) -> Self::Output {
        (&self).div(rhs)
    }
}

// Scalar-on-the-left division gets its own impls rather than going through
// `impl_op_basic!`: there is no `rev` post-processing of `tensor op scalar`
// that yields `scalar / tensor` without either a second rounding step or a
// panic on integer tensors, so it is computed directly via `rdiv`.
macro_rules! impl_div_scalar_lhs {
    ($typ:ty, $conv:expr) => {
        impl Div<Tensor> for $typ {
            type Output = Tensor;

            fn div(self, rhs: Tensor) -> Self::Output {
                self.div(&rhs)
            }
        }

        impl Div<&Tensor> for $typ {
            type Output = Tensor;

            fn div(self, rhs: &Tensor) -> Self::Output {
                rdiv($conv(self), rhs)
            }
        }
    };
}

impl_div_scalar_lhs!(i32, |v| v as i64);
impl_div_scalar_lhs!(i64, |v| v);
impl_div_scalar_lhs!(f32, |v| v as f64);
impl_div_scalar_lhs!(f64, |v| v);

macro_rules! impl_op_basic {
    /* rev such that rev(op(b, a)) = op(a, b) */
    ($trait:ident, $func:ident, $op:ident, $rev:ident) => {
        impl $trait<Tensor> for i32 {
            type Output = Tensor;

            fn $func(self, rhs: Tensor) -> Self::Output {
                self.$func(&rhs)
            }
        }

        impl $trait<Tensor> for i64 {
            type Output = Tensor;

            fn $func(self, rhs: Tensor) -> Self::Output {
                self.$func(&rhs)
            }
        }

        impl $trait<Tensor> for f32 {
            type Output = Tensor;

            fn $func(self, rhs: Tensor) -> Self::Output {
                self.$func(&rhs)
            }
        }

        impl $trait<Tensor> for f64 {
            type Output = Tensor;

            fn $func(self, rhs: Tensor) -> Self::Output {
                self.$func(&rhs)
            }
        }

        impl $trait<&Tensor> for i32 {
            type Output = Tensor;

            fn $func(self, rhs: &Tensor) -> Self::Output {
                $rev(rhs.$op(self as i64))
            }
        }

        impl $trait<&Tensor> for i64 {
            type Output = Tensor;

            fn $func(self, rhs: &Tensor) -> Self::Output {
                $rev(rhs.$op(self))
            }
        }

        impl $trait<&Tensor> for f32 {
            type Output = Tensor;

            fn $func(self, rhs: &Tensor) -> Self::Output {
                $rev(rhs.$op(self as f64))
            }
        }

        impl $trait<&Tensor> for f64 {
            type Output = Tensor;

            fn $func(self, rhs: &Tensor) -> Self::Output {
                $rev(rhs.$op(self))
            }
        }
    };
}

macro_rules! impl_op_assign {
    ($trait:ident, $func:ident, $op:ident) => {
        impl $trait<Tensor> for Tensor {
            fn $func(&mut self, rhs: Tensor) {
                let _ = self.$op(&rhs);
            }
        }

        impl $trait<&Tensor> for Tensor {
            fn $func(&mut self, rhs: &Tensor) {
                let _ = self.$op(rhs);
            }
        }
    };
}

macro_rules! impl_op_assign_basic {
    ($trait:ident, $func:ident, $op:ident) => {
        impl $trait<i32> for Tensor {
            fn $func(&mut self, rhs: i32) {
                let _ = self.$op(rhs as i64);
            }
        }

        impl $trait<i64> for Tensor {
            fn $func(&mut self, rhs: i64) {
                let _ = self.$op(rhs);
            }
        }

        impl $trait<f32> for Tensor {
            fn $func(&mut self, rhs: f32) {
                let _ = self.$op(rhs as f64);
            }
        }

        impl $trait<f64> for Tensor {
            fn $func(&mut self, rhs: f64) {
                let _ = self.$op(rhs);
            }
        }
    };
}

impl_op!(Add, add, g_add);
impl_op_basic!(Add, add, g_add_scalar, id);
impl_op_assign!(AddAssign, add_assign, g_add_);
impl_op_assign_basic!(AddAssign, add_assign, g_add_scalar_);

impl_op!(Mul, mul, g_mul);
impl_op_basic!(Mul, mul, g_mul_scalar, id);
impl_op_assign!(MulAssign, mul_assign, g_mul_);
impl_op_assign_basic!(MulAssign, mul_assign, g_mul_scalar_);

impl_op!(Div, div, g_div);
impl_op_assign!(DivAssign, div_assign, g_div_);
impl_op_assign_basic!(DivAssign, div_assign, g_div_scalar_);

impl_op!(Sub, sub, g_sub);
impl_op_basic!(Sub, sub, g_sub_scalar, neg);
impl_op_assign!(SubAssign, sub_assign, g_sub_);
impl_op_assign_basic!(SubAssign, sub_assign, g_sub_scalar_);

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        self.f_neg().unwrap()
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        self.f_neg().unwrap()
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Tensor) -> bool {
        if self.size() != other.size() {
            return false;
        }
        match self.f_eq_tensor(other) {
            Err(_) => false,
            Ok(v) => match v.f_all() {
                Err(_) => false,
                Ok(v) => match i64::try_from(v) {
                    Err(_) => false,
                    Ok(v) => v > 0,
                },
            },
        }
    }
}
