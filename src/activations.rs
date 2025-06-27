use candle_core::shape::Dim;
use candle_core::{Error, Tensor};
use candle_nn::Module;
use std::f32::consts;

pub struct Activations {}

impl Activations {
    pub fn softmax(input: &Tensor, dim: Option<usize>) -> Result<Tensor, Error> {
        // pick teh axis or default to last dim if none
        let axis_usize = dim.unwrap_or_else(|| input.shape().dims().len() - 1);
        let axis = axis_usize.to_index(input.shape(), "softmax")?;
        let max = input.max_keepdim(axis)?;
        let shifted = input.broadcast_sub(&max)?;
        let exps = shifted.exp()?;

        let sum = exps.sum_keepdim(axis)?;

        let probas = exps.broadcast_div(&sum)?;

        Ok(probas)
    }

    pub fn log_softmax(input: &Tensor, dim: Option<usize>) -> Result<Tensor, Error> {
        let axis_usize = dim.unwrap_or_else(|| input.shape().dims().len() - 1);
        let axis = axis_usize.to_index(input.shape(), "log_softmax")?;

        let max = input.max_keepdim(axis)?;
        let shifted = input.broadcast_sub(&max)?;

        let exps = shifted.exp()?;

        let sum = exps.sum_keepdim(axis)?;

        let lse = sum.log()? + &max;

        let output = input.broadcast_sub(&lse?)?;

        Ok(output)
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
