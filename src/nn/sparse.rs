//! Sparse Layers
use crate::{IndexOp, Tensor};
use std::borrow::Borrow;

/// Configuration option for an embedding layer.
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingConfig {
    pub sparse: bool,
    pub scale_grad_by_freq: bool,
    pub ws_init: super::Init,
    pub padding_idx: i64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        EmbeddingConfig {
            sparse: false,
            scale_grad_by_freq: false,
            ws_init: super::Init::Randn { mean: 0., stdev: 1. },
            padding_idx: -1,
        }
    }
}

/// An embedding layer.
///
/// An embedding layer acts as a simple lookup table that stores embeddings.
/// This is commonly used to store word embeddings.
#[derive(Debug)]
pub struct Embedding {
    pub ws: Tensor,
    config: EmbeddingConfig,
}

pub fn embedding<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    num_embeddings: i64,
    embedding_dim: i64,
    config: EmbeddingConfig,
) -> Embedding {
    let vs = vs.borrow();
    let mut config = config;
    // Python normalizes negative padding_idx values (-2 is the second-to-last
    // row, etc.); do the same so ported code behaves identically. -1 is
    // excluded: this crate reserves it as the "no padding" sentinel (PyTorch
    // uses None) — a Python padding_idx of -1 is num_embeddings - 1 here.
    if config.padding_idx < -1 {
        config.padding_idx += num_embeddings;
    }
    let ws = vs.var("weight", &[num_embeddings, embedding_dim], config.ws_init);
    // PyTorch zeroes the padding_idx row after init (and the embedding op
    // keeps its gradient at zero), so padding tokens embed to an exact zero
    // vector instead of a frozen random one. A padding_idx of -1 means no
    // padding handling, matching the ATen convention used by the op below.
    if config.padding_idx >= 0 && config.padding_idx < num_embeddings {
        crate::no_grad(|| {
            let _ = ws.i(config.padding_idx).fill_(0.);
        });
    }
    Embedding { ws, config }
}

impl super::module::Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::embedding(
            &self.ws,
            xs,
            self.config.padding_idx,
            self.config.scale_grad_by_freq,
            self.config.sparse,
        )
    }
}
