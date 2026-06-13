//! Numerical parity checks of the vision models against torchvision.
//!
//! These tests are ignored by default since they need reference fixtures
//! produced by `tests/generate_vision_parity.py` (see the header of that
//! script). Both sides execute the same libtorch CPU kernels, so the outputs
//! must agree to float rounding; anything larger indicates an architecture
//! divergence (wrong stride/padding/eps/naming).
use std::convert::TryFrom;
use tch::nn::ModuleT;
use tch::{nn, vision, Device, Tensor};

fn fixture_dir() -> String {
    std::env::var("TCH_VISION_PARITY_DIR").unwrap_or_else(|_| "/tmp/tv_parity".to_string())
}

fn parity(name: &str, build: impl FnOnce(&nn::Path) -> Box<dyn ModuleT>) {
    let dir = fixture_dir();
    let weights = format!("{dir}/{name}.safetensors");
    assert!(
        std::path::Path::new(&weights).exists(),
        "missing fixtures for {name}: run tests/generate_vision_parity.py first"
    );
    let mut vs = nn::VarStore::new(Device::Cpu);
    let model = build(&vs.root());
    vs.load(&weights).unwrap_or_else(|e| panic!("{name}: loading weights failed: {e}"));
    let input = Tensor::read_npy(format!("{dir}/{name}_in.npy")).unwrap();
    let expected = Tensor::read_npy(format!("{dir}/{name}_out.npy")).unwrap();
    let output = tch::no_grad(|| model.forward_t(&input, false));
    assert_eq!(output.size(), expected.size(), "{name}: output shape mismatch");
    let diff = f64::try_from((&output - &expected).abs().max()).unwrap();
    let scale = f64::try_from(expected.abs().max()).unwrap();
    println!("{name}: max abs diff {diff:.3e} (output scale {scale:.3e})");
    assert!(diff <= 1e-4 * scale.max(1.0), "{name} diverges from torchvision: {diff:.3e}");
}

#[test]
#[ignore]
fn parity_alexnet() {
    parity("alexnet", |p| Box::new(vision::alexnet::alexnet(p, 1000)));
}

#[test]
#[ignore]
fn parity_vgg16() {
    parity("vgg16", |p| Box::new(vision::vgg::vgg16(p, 1000)));
}

#[test]
#[ignore]
fn parity_vgg16_bn() {
    parity("vgg16_bn", |p| Box::new(vision::vgg::vgg16_bn(p, 1000)));
}

#[test]
#[ignore]
fn parity_resnet18() {
    parity("resnet18", |p| Box::new(vision::resnet::resnet18(p, 1000)));
}

#[test]
#[ignore]
fn parity_resnet50() {
    parity("resnet50", |p| Box::new(vision::resnet::resnet50(p, 1000)));
}

#[test]
#[ignore]
fn parity_densenet121() {
    parity("densenet121", |p| Box::new(vision::densenet::densenet121(p, 1000)));
}

#[test]
#[ignore]
fn parity_mobilenet_v2() {
    parity("mobilenet_v2", |p| Box::new(vision::mobilenet::v2(p, 1000)));
}

#[test]
#[ignore]
fn parity_squeezenet1_0() {
    parity("squeezenet1_0", |p| Box::new(vision::squeezenet::v1_0(p, 1000)));
}

#[test]
#[ignore]
fn parity_squeezenet1_1() {
    parity("squeezenet1_1", |p| Box::new(vision::squeezenet::v1_1(p, 1000)));
}

#[test]
#[ignore]
fn parity_inception_v3() {
    parity("inception_v3", |p| Box::new(vision::inception::v3(p, 1000)));
}

#[test]
#[ignore]
fn parity_efficientnet_b0() {
    parity("efficientnet_b0", |p| Box::new(vision::efficientnet::b0(p, 1000)));
}

#[test]
#[ignore]
fn parity_efficientnet_b4() {
    parity("efficientnet_b4", |p| Box::new(vision::efficientnet::b4(p, 1000)));
}

#[test]
#[ignore]
fn parity_efficientnet_b5() {
    parity("efficientnet_b5", |p| Box::new(vision::efficientnet::b5(p, 1000)));
}
