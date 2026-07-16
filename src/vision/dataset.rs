//! A simple dataset structure shared by various computer vision datasets.
use crate::data::Iter2;
use crate::{IndexOp, Kind, Tensor};

#[derive(Debug)]
pub struct Dataset {
    pub train_images: Tensor,
    pub train_labels: Tensor,
    pub test_images: Tensor,
    pub test_labels: Tensor,
    pub labels: i64,
}

impl Dataset {
    pub fn train_iter(&self, batch_size: i64) -> Iter2 {
        Iter2::new(&self.train_images, &self.train_labels, batch_size)
    }

    pub fn test_iter(&self, batch_size: i64) -> Iter2 {
        Iter2::new(&self.test_images, &self.test_labels, batch_size)
    }
}

/// Randomly applies horizontal flips
/// This expects a 4 dimension NCHW tensor and returns a tensor with
/// an identical shape.
pub fn random_flip(t: &Tensor) -> Tensor {
    let size = t.size();
    if size.len() != 4 {
        panic!("unexpected shape for tensor {t:?}")
    }
    // Flip the whole batch once and pick per-sample between the flipped and
    // original images with a broadcast mask: two kernels instead of a
    // narrow/flip/copy_ sequence per sample. Drawing the mask from the torch
    // RNG also makes the augmentation reproducible under crate::manual_seed.
    let mask = Tensor::rand([size[0], 1, 1, 1], (Kind::Float, t.device())).lt(0.5);
    t.flip([3]).where_self(&mask, t)
}

/// Pad the image using reflections and take some random crops.
/// This expects a 4 dimension NCHW tensor and returns a tensor with
/// an identical shape.
pub fn random_crop(t: &Tensor, pad: i64) -> Tensor {
    let size = t.size();
    if size.len() != 4 {
        panic!("unexpected shape for tensor {t:?}")
    }
    let (n, c, h, w) = (size[0], size[1], size[2], size[3]);
    let device = t.device();
    let padded = t.reflection_pad2d([pad, pad, pad, pad]); // [n, c, h+2*pad, w+2*pad]
    // The padded image is h/w + 2*pad high/wide so the valid crop offsets are
    // 0..=2*pad: randint's exclusive upper bound must be 2*pad + 1 or it would
    // never sample the bottom/right-most crop. Drawing the offsets from the
    // torch RNG also makes the augmentation reproducible under
    // crate::manual_seed.
    let offsets = Tensor::randint(2 * pad + 1, [2, n], (Kind::Int64, device)); // [2, n]
    // Output row r of sample i is padded row dy_i + r (and likewise for
    // columns), so the whole batch is two gathers over indices built from a
    // broadcast arange + per-sample offsets: a handful of kernels instead of
    // a narrow/copy_ pair per sample.
    let rows = offsets.i(0).view([n, 1]) + Tensor::arange(h, (Kind::Int64, device)); // [n, h]
    let cols = offsets.i(1).view([n, 1]) + Tensor::arange(w, (Kind::Int64, device)); // [n, w]
    // gather wants an index of the same rank as its input whose size may only
    // differ along the gathered dim; the expands are stride-0 views.
    let rows = rows.view([n, 1, h, 1]).expand([n, c, h, w + 2 * pad], false);
    let cols = cols.view([n, 1, 1, w]).expand([n, c, h, w], false);
    padded
        .gather(2, &rows, false) // [n, c, h, w+2*pad]
        .gather(3, &cols, false) // [n, c, h, w]
}

/// Applies cutout: randomly remove some square areas in the original images.
/// <https://arxiv.org/abs/1708.04552>
pub fn random_cutout(t: &Tensor, sz: i64) -> Tensor {
    let size = t.size();
    if size.len() != 4 || sz > size[2] || sz > size[3] {
        panic!("unexpected shape for tensor {t:?} {sz}")
    }
    let (n, h, w) = (size[0], size[2], size[3]);
    let device = t.device();
    // The square must fit inside the image so its top-left corner is uniform
    // over 0..=h-sz / 0..=w-sz; randint's upper bound is exclusive, matching
    // the previous per-sample rand::random_range(0..dim - sz + 1). Torch-RNG
    // offsets keep the augmentation reproducible under crate::manual_seed.
    let start_h = Tensor::randint(h - sz + 1, [n, 1, 1, 1], (Kind::Int64, device));
    let start_w = Tensor::randint(w - sz + 1, [n, 1, 1, 1], (Kind::Int64, device));
    // Pixel (r, c) of sample i is cut iff start_h_i <= r < start_h_i + sz and
    // the same for c: broadcasting the [n, 1, 1, 1] offsets against [1, 1, h, 1]
    // / [1, 1, 1, w] aranges gives per-sample row/column masks that combine
    // into one [n, 1, h, w] mask, so a single masked_fill replaces a fill_
    // per sample.
    let rows = Tensor::arange(h, (Kind::Int64, device)).view([1, 1, h, 1]);
    let cols = Tensor::arange(w, (Kind::Int64, device)).view([1, 1, 1, w]);
    let rows_in = rows.ge_tensor(&start_h).logical_and(&rows.lt_tensor(&(start_h + sz))); // [n, 1, h, 1]
    let cols_in = cols.ge_tensor(&start_w).logical_and(&cols.lt_tensor(&(start_w + sz))); // [n, 1, 1, w]
    // masked_fill broadcasts the [n, 1, h, w] mask over the channel dim and
    // returns a fresh tensor, leaving the input untouched like the previous
    // copy-then-fill_ version.
    t.masked_fill(&rows_in.logical_and(&cols_in), 0.0)
}

pub fn augmentation(t: &Tensor, flip: bool, crop: i64, cutout: i64) -> Tensor {
    let mut t = t.shallow_clone();
    if flip {
        t = random_flip(&t);
    }
    if crop > 0 {
        t = random_crop(&t, crop);
    }
    if cutout > 0 {
        t = random_cutout(&t, cutout);
    }
    t
}
