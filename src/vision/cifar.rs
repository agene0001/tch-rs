//! The CIFAR-10 dataset.
//!
//! The files can be downloaded from the following page:
//! <https://www.cs.toronto.edu/~kriz/cifar.html>
//! The binary version of the dataset is used.
use super::dataset::Dataset;
use crate::{Kind, Tensor};
use std::fs::File;
use std::io::{BufReader, Read, Result};

const W: i64 = 32;
const H: i64 = 32;
const C: i64 = 3;
const BYTES_PER_IMAGE: i64 = W * H * C + 1;
const SAMPLES_PER_FILE: i64 = 10000;

// Decodes a CIFAR binary batch where each record is one label byte followed
// by a C*H*W image. Slicing the whole batch at once replaces the previous
// per-sample copy loop (~60k tensor ops per file) with a handful of ops.
fn decode_samples(data: &[u8]) -> (Tensor, Tensor) {
    let content = Tensor::from_slice(data).view((SAMPLES_PER_FILE, BYTES_PER_IMAGE));
    let labels = content.select(1, 0).to_kind(Kind::Int64);
    let images = content
        .narrow(1, 1, BYTES_PER_IMAGE - 1)
        .reshape([SAMPLES_PER_FILE, C, H, W])
        .to_kind(Kind::Float)
        / 255.0;
    (images, labels)
}

fn read_file_(filename: &std::path::Path) -> Result<(Tensor, Tensor)> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    let mut data = vec![0u8; (SAMPLES_PER_FILE * BYTES_PER_IMAGE) as usize];
    buf_reader.read_exact(&mut data)?;
    Ok(decode_samples(&data))
}

fn read_file(filename: &std::path::Path) -> Result<(Tensor, Tensor)> {
    read_file_(filename)
        .map_err(|err| std::io::Error::new(err.kind(), format!("{filename:?} {err}")))
}

pub fn load_dir<T: AsRef<std::path::Path>>(dir: T) -> Result<Dataset> {
    let dir = dir.as_ref();
    let (test_images, test_labels) = read_file(&dir.join("test_batch.bin"))?;
    let train_images_and_labels = [
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin",
    ]
    .iter()
    .map(|x| read_file(&dir.join(x)))
    .collect::<Result<Vec<_>>>()?;
    let (train_images, train_labels): (Vec<_>, Vec<_>) =
        train_images_and_labels.into_iter().unzip();
    Ok(Dataset {
        train_images: Tensor::cat(&train_images, 0),
        train_labels: Tensor::cat(&train_labels, 0),
        test_images,
        test_labels,
        labels: 10,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IndexOp;

    #[test]
    fn decode_matches_binary_layout() {
        // Build a synthetic batch where every byte is a known function of its
        // record offset: label = n % 10, pixel byte = (n + pixel_index) % 256.
        let mut data = vec![0u8; (SAMPLES_PER_FILE * BYTES_PER_IMAGE) as usize];
        for n in 0..SAMPLES_PER_FILE {
            let base = (n * BYTES_PER_IMAGE) as usize;
            data[base] = (n % 10) as u8;
            for p in 0..(BYTES_PER_IMAGE - 1) as usize {
                data[base + 1 + p] = ((n as usize + p) % 256) as u8;
            }
        }
        let (images, labels) = decode_samples(&data);
        assert_eq!(images.size(), [SAMPLES_PER_FILE, C, H, W]);
        assert_eq!(labels.size(), [SAMPLES_PER_FILE]);
        assert_eq!(labels.kind(), Kind::Int64);
        assert_eq!(images.kind(), Kind::Float);

        for &n in &[0i64, 1, 4999, SAMPLES_PER_FILE - 1] {
            assert_eq!(i64::try_from(labels.get(n)).unwrap(), n % 10);
            // The binary format stores channel-major planes: pixel index for
            // (c, h, w) is c*H*W + h*W + w.
            for &(c, h, w) in &[(0i64, 0i64, 0i64), (1, 2, 3), (2, 31, 31)] {
                let p = (c * H * W + h * W + w) as usize;
                let expected = ((n as usize + p) % 256) as f64 / 255.0;
                let got = f64::try_from(images.i((n, c, h, w))).unwrap();
                assert!((got - expected).abs() < 1e-6, "sample {n} pixel {p}: {got} vs {expected}");
            }
        }
    }
}
