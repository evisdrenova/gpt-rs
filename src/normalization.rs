use candle_core::{DType, Device, Error, Tensor};
use candle_nn::Module;

pub struct LayerNorm {
    pub eps: f32,
    pub scale: Tensor,
    pub shift: Tensor,
}

impl LayerNorm {
    pub fn new(emb_dim: usize, eps: f32, device: &Device) -> Result<Self, Error> {
        let scale = Tensor::ones(emb_dim, DType::F32, device)?;
        let shift = Tensor::zeros(emb_dim, DType::F32, device)?;

        Ok(Self { eps, scale, shift })
    }

    pub fn new_default(emb_dim: usize, device: &Device) -> Result<Self, Error> {
        Self::new(emb_dim, 1e-5, device)
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let last_dim = x.dims().len() - 1;

        let mean = x.mean_keepdim(last_dim)?;
        let var = x.var_keepdim(last_dim)?;

        let x_centered = x.broadcast_sub(&mean)?;
        let var_eps = var.broadcast_add(&Tensor::new(self.eps, x.device())?)?;
        let std = var_eps.sqrt()?;
        let norm_x = x_centered.broadcast_div(&std)?;

        let scaled = norm_x.broadcast_mul(&self.scale)?;
        let result = scaled.broadcast_add(&self.shift)?;

        Ok(result)
    }
}
