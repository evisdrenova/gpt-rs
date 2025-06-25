use candle_core::{Error, Tensor};
use candle_nn::Module;
use std::f32::consts;

pub struct Activations {}

impl Activations {
    pub fn softmax(input: &Tensor, dim: Option<usize>) -> Result<Tensor, Error> {
        // Softmax formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)

        // Apply exponential function element-wise
        let exp_values = input.exp()?;

        let softmax_result = match dim {
            Some(dimension) => {
                // Row-wise (or dimension-wise) softmax
                // Sum along the specified dimension while keeping dimensions
                let exp_sum = exp_values.sum_keepdim(dimension)?;
                // Normalize: divide each element by the sum along that dimension
                exp_values.broadcast_div(&exp_sum)?
            }
            None => {
                // Sum all exponential values
                let exp_sum = exp_values.sum_all()?;
                // Normalize: divide each exp value by the total sum
                exp_values.broadcast_div(&exp_sum)?
            }
        };

        Ok(softmax_result)
    }
}

pub struct GeLU {}

impl GeLU {
    pub fn new() -> Result<Self, Error> {
        Ok(GeLU {})
    }
}

impl Module for GeLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let sqrt_2_over_pi_val = (2.0f32 / consts::PI).sqrt();
        let sqrt_2_over_pi = Tensor::new(sqrt_2_over_pi_val, x.device())?;

        let x_cubed = x.powf(3.0)?;

        let coefficient = Tensor::new(0.044715f32, x.device())?;
        let cubic_term = x_cubed.broadcast_mul(&coefficient)?;

        let inner = x.broadcast_add(&cubic_term)?;

        let tanh_input = inner.broadcast_mul(&sqrt_2_over_pi)?;

        let tanh_result = tanh_input.tanh()?;

        let ones = Tensor::ones(tanh_result.shape(), x.dtype(), x.device())?;
        let one_plus_tanh = tanh_result.broadcast_add(&ones)?;

        let half = Tensor::new(0.5f32, x.device())?;
        let half_x = x.broadcast_mul(&half)?;

        let gelu = half_x.broadcast_mul(&one_plus_tanh)?;

        Ok(gelu)
    }
}
