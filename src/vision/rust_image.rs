/// Conversion helpers for the image crate.
use std::convert::TryFrom;

use image::{
    DynamicImage, EncodableLayout, GrayAlphaImage, GrayImage, Rgb32FImage, RgbImage, Rgba32FImage,
    RgbaImage,
};

use crate::vision::image::{chw_to_hwc, hwc_to_chw};
use crate::{Kind, TchError, Tensor};

impl<'i> TryFrom<&'i DynamicImage> for Tensor {
    type Error = TchError;

    fn try_from(image: &'i DynamicImage) -> Result<Self, Self::Error> {
        match image {
            DynamicImage::ImageLuma8(gray) => Tensor::try_from(gray),
            DynamicImage::ImageLumaA8(gray_a) => Tensor::try_from(gray_a),
            DynamicImage::ImageRgb8(rgb) => Tensor::try_from(rgb),
            DynamicImage::ImageRgba8(rgba) => Tensor::try_from(rgba),
            DynamicImage::ImageRgb32F(rgb) => Tensor::try_from(rgb),
            DynamicImage::ImageRgba32F(rgba) => Tensor::try_from(rgba),
            _ => Err(TchError::Convert("unsupported DynamicImage variant".to_string())),
        }
    }
}

// The image->tensor conversions materialize the CHW layout once up front so
// consumers don't pay a hidden contiguity copy on every downstream op.

impl<'i> TryFrom<&'i GrayImage> for Tensor {
    type Error = TchError;

    ///  `h * w` => `1 * h * w`
    fn try_from(gray: &'i GrayImage) -> Result<Self, Self::Error> {
        let size = &[gray.height() as i64, gray.width() as i64, 1];
        let tensor = Tensor::f_from_data_size(gray.as_bytes(), size, Kind::Uint8)?;
        Ok(hwc_to_chw(&tensor).contiguous())
    }
}

impl<'i> TryFrom<&'i GrayAlphaImage> for Tensor {
    type Error = TchError;

    ///  `h * w * 2` => `2 * h * w`
    fn try_from(gray: &'i GrayAlphaImage) -> Result<Self, Self::Error> {
        let size = &[gray.height() as i64, gray.width() as i64, 2];
        let tensor = Tensor::f_from_data_size(gray.as_bytes(), size, Kind::Uint8)?;
        Ok(hwc_to_chw(&tensor).contiguous())
    }
}

impl<'i> TryFrom<&'i RgbImage> for Tensor {
    type Error = TchError;

    /// `h * w * 3` => `3 * h * w`
    fn try_from(rgb: &'i RgbImage) -> Result<Self, Self::Error> {
        let size = &[rgb.height() as i64, rgb.width() as i64, 3];
        let tensor = Tensor::f_from_data_size(rgb.as_raw(), size, Kind::Uint8)?;
        Ok(hwc_to_chw(&tensor).contiguous())
    }
}

impl<'i> TryFrom<&'i RgbaImage> for Tensor {
    type Error = TchError;

    /// `h * w * 4` => `4 * h * w`
    fn try_from(rgb: &'i RgbaImage) -> Result<Self, Self::Error> {
        let kind = Kind::Uint8;
        let size = &[rgb.height() as i64, rgb.width() as i64, 4];
        let tensor = Tensor::f_from_data_size(rgb.as_raw(), size, kind)?;
        Ok(hwc_to_chw(&tensor).contiguous())
    }
}

impl<'i> TryFrom<&'i Tensor> for RgbImage {
    type Error = TchError;

    ///  `3 * h * w` (or `1 * 3 * h * w`) => `h * w * 3`
    fn try_from(value: &'i Tensor) -> Result<Self, Self::Error> {
        let tensor = assert_tensor_as_image(value, Kind::Uint8, 3)?;
        let width = tensor.size()[1] as u32;
        let height = tensor.size()[0] as u32;
        let length = (width * height * 3) as usize;
        let mut buffer = vec![0; length];
        tensor.f_copy_data(&mut buffer, length)?;
        RgbImage::from_raw(width, height, buffer)
            .ok_or_else(|| TchError::Convert("Failed to convert tensor to image".to_string()))
    }
}

impl<'i> TryFrom<&'i Rgb32FImage> for Tensor {
    type Error = TchError;

    /// `h * w * 3` => `3 * h * w`
    fn try_from(rgb: &'i Rgb32FImage) -> Result<Self, Self::Error> {
        let kind = Kind::Float;
        let size = &[rgb.height() as i64, rgb.width() as i64, 3];
        let tensor = Tensor::f_from_data_size(rgb.as_bytes(), size, kind)?;
        Ok(hwc_to_chw(&tensor).contiguous())
    }
}

impl<'i> TryFrom<&'i Rgba32FImage> for Tensor {
    type Error = TchError;

    /// `h * w * 4` => `4 * h * w`
    fn try_from(rgb: &'i Rgba32FImage) -> Result<Self, Self::Error> {
        let kind = Kind::Float;
        let size = &[rgb.height() as i64, rgb.width() as i64, 4];
        let tensor = Tensor::f_from_data_size(rgb.as_bytes(), size, kind)?;
        Ok(hwc_to_chw(&tensor).contiguous())
    }
}

impl<'i> TryFrom<&'i Tensor> for Rgb32FImage {
    type Error = TchError;

    ///  `3 * h * w` (or `1 * 3 * h * w`) => `h * w * 3`
    fn try_from(value: &'i Tensor) -> Result<Self, Self::Error> {
        let tensor = assert_tensor_as_image(value, Kind::Float, 3)?;
        let width = tensor.size()[1] as u32;
        let height = tensor.size()[0] as u32;
        let length = (width * height * 3) as usize;
        let mut buffer = vec![0.0; length];
        tensor.f_copy_data(&mut buffer, length)?;
        Rgb32FImage::from_raw(width, height, buffer)
            .ok_or_else(|| TchError::Convert("Failed to convert tensor to image".to_string()))
    }
}

#[inline]
fn assert_tensor_as_image(
    tensor: &Tensor,
    expected: Kind,
    channel: i64,
) -> Result<Tensor, TchError> {
    let kind = tensor.kind();
    let mut size = tensor.size();
    // Accept a singleton batch [1, C, H, W] as [C, H, W].
    let tensor = if size.len() == 4 && size[0] == 1 {
        size.remove(0);
        tensor.squeeze_dim(0)
    } else {
        tensor.shallow_clone()
    };
    if size.len() != 3 {
        let msg = format!("Tensor size is `{size:?}`, expected a rank 3 tensor");
        Err(TchError::Convert(msg))
    } else if size[0] != channel {
        let msg = format!("Tensor size is `{size:?}`, expected {channel} channels");
        Err(TchError::Convert(msg))
    } else if kind != expected {
        let msg = format!("Tensor kind is `{kind:?}`, expected `{expected:?}` tensor");
        Err(TchError::Convert(msg))
    } else {
        Ok(chw_to_hwc(&tensor))
    }
}
