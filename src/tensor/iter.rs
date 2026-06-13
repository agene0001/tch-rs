use super::Tensor;
use crate::{kind::Element, TchError};

/// An iterator over the elements of a one dimensional tensor.
///
/// The tensor values are read in bulk when the iterator is created: reading
/// element by element through the C api would perform one tensor to scalar
/// conversion per element, and on CUDA a device synchronization per element.
pub struct Iter<T> {
    index: usize,
    content: Vec<T>,
}

impl Tensor {
    pub fn iter<T: Element + Copy>(&self) -> Result<Iter<T>, TchError> {
        Ok(Iter { index: 0, content: Vec::<T>::try_from(self)? })
    }
}

impl<T: Copy> std::iter::Iterator for Iter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let v = *self.content.get(self.index)?;
        self.index += 1;
        Some(v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.content.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<T: Copy> std::iter::ExactSizeIterator for Iter<T> {}

impl std::iter::Sum for Tensor {
    fn sum<I: Iterator<Item = Tensor>>(mut iter: I) -> Tensor {
        match iter.next() {
            None => Tensor::from(0.),
            Some(t) => iter.fold(t, |acc, x| x + acc),
        }
    }
}

impl<'a> std::iter::Sum<&'a Tensor> for Tensor {
    fn sum<I: Iterator<Item = &'a Tensor>>(mut iter: I) -> Tensor {
        match iter.next() {
            None => Tensor::from(0.),
            Some(t) => iter.fold(t.shallow_clone(), |acc, x| x + acc),
        }
    }
}
