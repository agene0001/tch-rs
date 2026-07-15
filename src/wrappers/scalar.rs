//! Scalar elements.
//!
//! `Scalar` used to wrap a heap-allocated C++ `torch::Scalar`, which cost two
//! FFI crossings plus an allocation/free for every scalar argument of every
//! op. It is now a plain Rust enum; the generated bindings pass the value by
//! value across the FFI boundary and the C shim builds the `at::Scalar`
//! inline. This also fixed a panic-in-Drop: the old Drop ran an error check
//! that picked up pending op errors before the fallible `f_*` API could
//! return them as `Err`.

use crate::TchError;

/// A single scalar value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    Int(i64),
    Float(f64),
}

impl Scalar {
    /// Creates an integer scalar.
    pub fn int(v: i64) -> Scalar {
        Scalar::Int(v)
    }

    /// Creates a float scalar.
    pub fn float(v: f64) -> Scalar {
        Scalar::Float(v)
    }

    /// Returns an integer value, truncating toward zero like
    /// `torch::Scalar::toLong`.
    pub fn to_int(self) -> Result<i64, TchError> {
        match self {
            Scalar::Int(i) => Ok(i),
            Scalar::Float(f) => Ok(f as i64),
        }
    }

    /// Returns a float value.
    pub fn to_float(self) -> Result<f64, TchError> {
        match self {
            Scalar::Int(i) => Ok(i as f64),
            Scalar::Float(f) => Ok(f),
        }
    }

    /// Returns a string representation of the scalar, matching the C++
    /// `operator<<` default of six significant digits for floats.
    pub fn to_string(&self) -> Result<String, TchError> {
        match self {
            Scalar::Int(i) => Ok(i.to_string()),
            Scalar::Float(f) => {
                let rounded: f64 = format!("{f:.5e}").parse().unwrap_or(*f);
                Ok(format!("{rounded}"))
            }
        }
    }

    // Lowered representation used by the generated bindings: the C shim
    // rebuilds the at::Scalar from (double, int64_t, is_int).
    pub(crate) fn d_value(&self) -> f64 {
        match self {
            Scalar::Int(i) => *i as f64,
            Scalar::Float(f) => *f,
        }
    }

    pub(crate) fn i_value(&self) -> i64 {
        match self {
            Scalar::Int(i) => *i,
            Scalar::Float(f) => *f as i64,
        }
    }

    pub(crate) fn is_int_scalar(&self) -> i8 {
        matches!(self, Scalar::Int(_)) as i8
    }
}

impl From<i64> for Scalar {
    fn from(v: i64) -> Scalar {
        Scalar::int(v)
    }
}

impl From<f64> for Scalar {
    fn from(v: f64) -> Scalar {
        Scalar::float(v)
    }
}

impl From<Scalar> for i64 {
    fn from(s: Scalar) -> i64 {
        Self::from(&s)
    }
}

impl From<Scalar> for f64 {
    fn from(s: Scalar) -> f64 {
        Self::from(&s)
    }
}

impl From<&Scalar> for i64 {
    fn from(s: &Scalar) -> i64 {
        s.to_int().unwrap()
    }
}

impl From<&Scalar> for f64 {
    fn from(s: &Scalar) -> f64 {
        s.to_float().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::Scalar;
    #[test]
    fn scalar() {
        let pi = Scalar::float(std::f64::consts::PI);
        assert_eq!(i64::from(&pi), 3);
        assert_eq!(f64::from(&pi), std::f64::consts::PI);
        let leet = Scalar::int(1337);
        assert_eq!(i64::from(&leet), 1337);
        assert_eq!(f64::from(&leet), 1337.);
        assert_eq!(&pi.to_string().unwrap(), "3.14159");
    }
}
