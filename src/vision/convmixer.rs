//! ConvMixer implementation.
//!
//! See "Patches Are All You Need?", submitted to ICLR'21
//! <https://openreview.net/forum?id=TVHS5Y4dNvM>
use crate::nn;

fn block(p: nn::Path, dim: i64, kernel_size: i64) -> impl nn::ModuleT + use<> {
    // With stride 1 and an odd kernel, PyTorch's padding="same" is exactly the
    // symmetric static padding k / 2 — no need for the dynamic zero_pad2d copy
    // that conv2d_same performs on every forward.
    let conv2d_cfg =
        nn::ConvConfig { groups: dim, padding: kernel_size / 2, ..Default::default() };
    let p_fn = &(&p / "0") / "fn";
    let conv1 = nn::conv2d(&p_fn / "0", dim, dim, kernel_size, conv2d_cfg);
    let bn1 = nn::batch_norm2d(&p_fn / "2", dim, Default::default());
    let conv2 = nn::conv2d(&p / "1", dim, dim, 1, Default::default());
    let bn2 = nn::batch_norm2d(&p / "3", dim, Default::default());
    nn::func_t(move |xs, train| {
        let ys = xs.apply(&conv1).gelu("none").apply_t(&bn1, train);
        (xs + ys).apply(&conv2).gelu("none").apply_t(&bn2, train)
    })
}

fn convmixer(
    p: &nn::Path,
    nclasses: i64,
    dim: i64,
    depth: i64,
    kernel_size: i64,
    patch_size: i64,
) -> nn::FuncT<'static> {
    let conv2d_cfg = nn::ConvConfig { stride: patch_size, ..Default::default() };
    let conv1 = nn::conv2d(p / "0", 3, dim, patch_size, conv2d_cfg);
    let bn1 = nn::batch_norm2d(p / "2", dim, Default::default());
    let blocks: Vec<_> = (0..depth).map(|index| block(p / (3 + index), dim, kernel_size)).collect();
    // Sequential index of the linear head: patch conv, gelu, bn, `depth`
    // blocks, pool, flatten.
    let fc = nn::linear(p / (3 + depth + 2), dim, nclasses, Default::default());
    nn::func_t(move |xs, train| {
        let mut xs = xs.apply(&conv1).gelu("none").apply_t(&bn1, train);
        for block in blocks.iter() {
            xs = xs.apply_t(block, train)
        }
        xs.adaptive_avg_pool2d([1, 1]).flat_view().apply(&fc)
    })
}

pub fn c1536_20(p: &nn::Path, nclasses: i64) -> nn::FuncT<'static> {
    convmixer(p, nclasses, 1536, 20, 9, 7)
}

pub fn c1024_20(p: &nn::Path, nclasses: i64) -> nn::FuncT<'static> {
    convmixer(p, nclasses, 1024, 20, 9, 14)
}
