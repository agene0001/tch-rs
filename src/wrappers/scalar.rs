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
    /// `torch::Scalar::toLong`; like it, errors on NaN/infinite values and
    /// floats outside the i64 range instead of saturating.
    pub fn to_int(self) -> Result<i64, TchError> {
        match self {
            Scalar::Int(i) => Ok(i),
            // `i64::MAX as f64` rounds up to 2^63, so `<` (not `<=`) is the
            // correct in-range test; every float strictly below it truncates
            // to at most i64::MAX.
            Scalar::Float(f) if f.is_finite() && f >= i64::MIN as f64 && f < i64::MAX as f64 => {
                Ok(f as i64)
            }
            Scalar::Float(f) => {
                Err(TchError::Convert(format!("float scalar {f} out of range for an i64")))
            }
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
    /// `operator<<` default for doubles (printf `%g`: six significant
    /// digits, scientific notation when the exponent is below -4 or 6 and
    /// above, trailing zeros trimmed).
    pub fn to_string(&self) -> Result<String, TchError> {
        match self {
            Scalar::Int(i) => Ok(i.to_string()),
            Scalar::Float(f) => Ok(float_to_string_like_cpp(*f)),
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

fn trim_float_zeros(s: &str) -> &str {
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.') } else { s }
}

/// printf `%g` with precision 6, the format C++ `operator<<` uses for
/// doubles: scientific notation when the decimal exponent is < -4 or >= 6,
/// fixed otherwise, six significant digits, trailing zeros trimmed and the
/// exponent written with a sign and at least two digits.
fn float_to_string_like_cpp(f: f64) -> String {
    if f.is_nan() {
        return "nan".to_string();
    }
    if f.is_infinite() {
        return if f.is_sign_positive() { "inf" } else { "-inf" }.to_string();
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0".to_string() } else { "0".to_string() };
    }
    // `{:.5e}` rounds to six significant digits and renormalizes the
    // mantissa, so the exponent it reports is the post-rounding one `%g`
    // bases its notation choice on.
    let sci = format!("{f:.5e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} always contains an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponents are integers");
    if !(-4..6).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", trim_float_zeros(mantissa), sign, exp.abs())
    } else {
        trim_float_zeros(&format!("{:.*}", (5 - exp) as usize, f)).to_string()
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

    #[test]
    fn to_string_matches_cpp_ostream() {
        let s = |f: f64| Scalar::float(f).to_string().unwrap();
        assert_eq!(s(1e20), "1e+20");
        assert_eq!(s(1.2345678e-5), "1.23457e-05");
        assert_eq!(s(1e-5), "1e-05");
        assert_eq!(s(0.0001), "0.0001");
        assert_eq!(s(100.0), "100");
        assert_eq!(s(-0.5), "-0.5");
        assert_eq!(s(999999.5), "1e+06");
        assert_eq!(s(123456.0), "123456");
        assert_eq!(s(0.0), "0");
        assert_eq!(s(f64::NAN), "nan");
        assert_eq!(s(f64::NEG_INFINITY), "-inf");
    }

    #[test]
    fn to_int_rejects_out_of_range_floats() {
        assert!(Scalar::float(f64::NAN).to_int().is_err());
        assert!(Scalar::float(f64::INFINITY).to_int().is_err());
        assert!(Scalar::float(1e300).to_int().is_err());
        assert_eq!(Scalar::float(2f64.powi(62)).to_int().unwrap(), 1 << 62);
        assert_eq!(Scalar::float(-2.9).to_int().unwrap(), -2);
    }
}
