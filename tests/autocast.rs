#[cfg(test)]
#[cfg(feature = "cuda-tests")]
mod tests {
    use tch::{autocast, Device, Kind, Tensor};

    #[test]
    fn autocast_narrows_type() {
        let device = Device::Cuda(0);

        let linear = Tensor::rand([10, 10], (Kind::Float, device));
        let input = Tensor::rand([10], (Kind::Float, device));

        autocast(true, || {
            let output1 = autocast(false, || linear.matmul(&input));
            assert_eq!(output1.kind(), Kind::Float);
            let output2 = linear.matmul(&output1);
            assert_eq!(output2.kind(), Kind::Half);
            let output3 = autocast(false, || linear.matmul(&output1));
            assert_eq!(output3.kind(), Kind::Float);
        });
    }
}

#[cfg(test)]
mod cpu_tests {
    use tch::{autocast_device, Device, Kind, Tensor};

    #[test]
    fn autocast_device_cpu_narrows_to_bf16() {
        let linear = Tensor::rand([10, 10], (Kind::Float, Device::Cpu));
        let input = Tensor::rand([10], (Kind::Float, Device::Cpu));

        autocast_device(Device::Cpu, None, true, || {
            let output = linear.matmul(&input);
            assert_eq!(output.kind(), Kind::BFloat16);
            // Nested disable restores full precision inside the region.
            let full = autocast_device(Device::Cpu, None, false, || linear.matmul(&input));
            assert_eq!(full.kind(), Kind::Float);
        });
        // State is restored after the region.
        let output = linear.matmul(&input);
        assert_eq!(output.kind(), Kind::Float);
    }

    #[test]
    fn autocast_device_explicit_dtype() {
        let linear = Tensor::rand([10, 10], (Kind::Float, Device::Cpu));
        let input = Tensor::rand([10], (Kind::Float, Device::Cpu));
        let output =
            autocast_device(Device::Cpu, Some(Kind::Half), true, || linear.matmul(&input));
        assert_eq!(output.kind(), Kind::Half);
    }
}
