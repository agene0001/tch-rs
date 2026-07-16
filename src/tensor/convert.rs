//! Implement conversion traits for tensors
use super::Tensor;
use crate::{kind::Element, TchError};
use half::{bf16, f16};
use std::convert::{TryFrom, TryInto};

// Copies the tensor's data into `dst`, converting the kind only when it
// differs from `T` — the common already-matching case skips the `totype`
// FFI round-trip and its shallow tensor.
fn copy_data_as<T: Element + Copy>(
    tensor: &Tensor,
    dst: &mut [T],
    numel: usize,
) -> Result<(), TchError> {
    if tensor.f_kind()? == T::KIND {
        tensor.f_copy_data(dst, numel)
    } else {
        tensor.f_to_kind(T::KIND)?.f_copy_data(dst, numel)
    }
}

impl<T: Element + Copy> TryFrom<&Tensor> for Vec<T> {
    type Error = TchError;
    fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
        let size = tensor.size();
        if size.len() != 1 {
            Err(TchError::Convert(format!(
                "Attempting to convert a Tensor with {} dimensions to flat vector",
                size.len()
            )))?;
        }
        let numel = size[0] as usize;
        let mut vec = vec![T::ZERO; numel];
        copy_data_as(tensor, &mut vec, numel)?;
        Ok(vec)
    }
}

impl<T: Element + Copy> TryFrom<&Tensor> for Vec<Vec<T>> {
    type Error = TchError;
    fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
        let (s1, s2) = tensor.size2()?;
        let s1 = s1 as usize;
        let s2 = s2 as usize;
        let num_elem = s1 * s2;
        // TODO: Try to remove this intermediary copy.
        let mut all_elems = vec![T::ZERO; num_elem];
        copy_data_as(tensor, &mut all_elems, num_elem)?;
        // Build the rows from contiguous chunks rather than indexing element by
        // element, which lets the compiler lower each row to a memcpy.
        // `chunks_exact` panics on a zero chunk size, so handle empty rows.
        let out = if s2 == 0 {
            (0..s1).map(|_| Vec::new()).collect()
        } else {
            all_elems.chunks_exact(s2).map(<[T]>::to_vec).collect()
        };
        Ok(out)
    }
}

impl<T: Element + Copy> TryFrom<&Tensor> for Vec<Vec<Vec<T>>> {
    type Error = TchError;
    fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
        let (s1, s2, s3) = tensor.size3()?;
        let s1 = s1 as usize;
        let s2 = s2 as usize;
        let s3 = s3 as usize;
        let num_elem = s1 * s2 * s3;
        // TODO: Try to remove this intermediary copy.
        let mut all_elems = vec![T::ZERO; num_elem];
        copy_data_as(tensor, &mut all_elems, num_elem)?;
        // Slice into contiguous chunks rather than indexing element by element.
        // `chunks_exact` panics on a zero chunk size, so handle empty dims.
        let out = if s2 == 0 || s3 == 0 {
            (0..s1).map(|_| (0..s2).map(|_| Vec::new()).collect()).collect()
        } else {
            all_elems
                .chunks_exact(s2 * s3)
                .map(|plane| plane.chunks_exact(s3).map(<[T]>::to_vec).collect())
                .collect()
        };
        Ok(out)
    }
}

impl<T: Element + Copy> TryFrom<Tensor> for Vec<T> {
    type Error = TchError;
    fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
        Vec::<T>::try_from(&tensor)
    }
}

impl<T: Element + Copy> TryFrom<Tensor> for Vec<Vec<T>> {
    type Error = TchError;
    fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
        Vec::<Vec<T>>::try_from(&tensor)
    }
}

impl<T: Element + Copy> TryFrom<Tensor> for Vec<Vec<Vec<T>>> {
    type Error = TchError;
    fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
        Vec::<Vec<Vec<T>>>::try_from(&tensor)
    }
}

macro_rules! from_tensor {
    ($typ:ident) => {
        impl TryFrom<&Tensor> for $typ {
            type Error = TchError;

            fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
                let numel = tensor.numel();
                if numel != 1 {
                    return Err(TchError::Convert(format!(
                        "expected exactly one element, got {}",
                        numel
                    )));
                }
                // copy_data_as skips the `totype` round-trip when the kind
                // already matches, and at_copy_data itself handles non-CPU
                // and non-contiguous tensors — no need for an explicit
                // to_device hop and its extra shallow tensor.
                let mut vec = [$typ::ZERO; 1];
                copy_data_as(tensor, &mut vec, numel)?;
                Ok(vec[0])
            }
        }

        impl TryFrom<Tensor> for $typ {
            type Error = TchError;

            fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
                $typ::try_from(&tensor)
            }
        }
    };
}

from_tensor!(f64);
from_tensor!(f32);
from_tensor!(f16);
from_tensor!(i64);
from_tensor!(i32);
from_tensor!(i16);
from_tensor!(i8);
from_tensor!(u8);
from_tensor!(bool);
from_tensor!(bf16);

impl<T: Element + Copy> TryInto<ndarray::ArrayD<T>> for &Tensor {
    type Error = TchError;

    fn try_into(self) -> Result<ndarray::ArrayD<T>, Self::Error> {
        let num_elem = self.numel();
        let mut vec = vec![T::ZERO; num_elem];
        copy_data_as(self, &mut vec, num_elem)?;
        let shape: Vec<usize> = self.size().iter().map(|s| *s as usize).collect();
        Ok(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), vec)?)
    }
}

impl<T, D> TryFrom<&ndarray::ArrayBase<T, D>> for Tensor
where
    T: ndarray::Data,
    T::Elem: Element,
    D: ndarray::Dimension,
{
    type Error = TchError;

    fn try_from(value: &ndarray::ArrayBase<T, D>) -> Result<Self, Self::Error> {
        let slice = value
            .as_slice()
            .ok_or_else(|| TchError::Convert("cannot convert to slice".to_string()))?;
        let tn = Self::f_from_slice(slice)?;
        let shape: Vec<i64> = value.shape().iter().map(|s| *s as i64).collect();
        tn.f_reshape(shape)
    }
}

impl<T, D> TryFrom<ndarray::ArrayBase<T, D>> for Tensor
where
    T: ndarray::Data,
    T::Elem: Element,
    D: ndarray::Dimension,
{
    type Error = TchError;

    fn try_from(value: ndarray::ArrayBase<T, D>) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl<T: Element> TryFrom<&Vec<T>> for Tensor {
    type Error = TchError;

    fn try_from(value: &Vec<T>) -> Result<Self, Self::Error> {
        Self::f_from_slice(value.as_slice())
    }
}

impl<T: Element> TryFrom<Vec<T>> for Tensor {
    type Error = TchError;

    fn try_from(value: Vec<T>) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}
