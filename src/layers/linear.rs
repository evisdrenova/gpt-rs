use candle_core::{Device, Error, Tensor};
use rand::Rng;

use crate::rng;

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}
/// Applies an affine linear transformation to the incoming data: y = xA^T + b, where  x is the input tensor, A is a randomly intialized weight matrix and b is a bias term
impl Linear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Self, Error> {
        let mut rng = rng::rng();

        // Glorot uniform initialization to maintain variance across forward/backward prop
        let std_dev = (2.0 / (in_features + out_features) as f64).sqrt() as f32;
        let bound = std_dev * (3.0_f32).sqrt();

        let weight_size = in_features * out_features;

        let mut weight_data: Vec<f32> = Vec::with_capacity(weight_size);
        for _ in 0..weight_size {
            weight_data.push(rng.random_range(-bound..bound));
        }

        // PyTorch: weight shape is [out_features, in_features]
        let weight = Tensor::from_vec(weight_data, (out_features, in_features), device)?;

        let bias = if bias {
            Some(Tensor::zeros(
                out_features,
                candle_core::DType::F32,
                device,
            )?)
        } else {
            None
        };

        Ok(Linear {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let input_shape = input.dims();

        if input_shape.is_empty() {
            return Err(Error::Msg("Input tensor cannot be empty".into()));
        }

        let last_dim = input_shape[input_shape.len() - 1];
        if last_dim != self.in_features {
            return Err(Error::Msg(
                format!(
                    "Input last dimension {} doesn't match in_features {}",
                    last_dim, self.in_features
                )
                .into(),
            ));
        }

        // transpose the weight matric to [in_features, out_features]
        let weight_t = self.weight.t()?;

        // broadcast_matmul will auto-rerank matrices so they are compatible
        input.broadcast_matmul(&weight_t).and_then(|out| {
            // optionally add bias
            if let Some(b) = &self.bias {
                out.broadcast_add(b)
            } else {
                Ok(out)
            }
        })
    }
}
