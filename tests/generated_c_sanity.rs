//! Sanity checks on the generated C shim sources.
//!
//! `cargo check --features doc-only` never compiles the C side, so a syntax
//! error emitted by `gen/gen.ml` (e.g. the trailing comma once produced for
//! zero-argument ops: `char *atg_foo(int *out__, )`) can survive unnoticed on
//! machines without libtorch. These tests parse nothing — they just scan the
//! generated text for patterns that are ill-formed C/C++.

use std::path::Path;

fn read_generated(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("torch-sys/libtch").join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read generated file {}: {e}", path.display()))
}

#[test]
fn generated_c_has_no_trailing_commas_in_parameter_lists() {
    for name in ["torch_api_generated.h", "torch_api_generated.cpp"] {
        let src = read_generated(name);
        for (idx, line) in src.lines().enumerate() {
            assert!(
                !line.contains(", )") && !line.contains(",)"),
                "{name}:{}: trailing comma in parameter list (ill-formed C++): {line}\n\
                 gen.ml must emit the separator conditionally for zero-argument ops",
                idx + 1
            );
        }
    }
}

#[test]
fn generated_c_layers_declare_the_same_functions() {
    let extract = |src: &str| {
        let mut names: Vec<String> = src
            .lines()
            .filter_map(|l| {
                let l = l.strip_prefix("char *atg_")?;
                Some(l.split('(').next().unwrap_or_default().to_string())
            })
            .collect();
        names.sort();
        names
    };
    let header = extract(&read_generated("torch_api_generated.h"));
    let impls = extract(&read_generated("torch_api_generated.cpp"));
    assert!(!header.is_empty(), "no atg_ declarations found in header");
    assert_eq!(header, impls, "generated .h declarations and .cpp definitions diverge");
}
