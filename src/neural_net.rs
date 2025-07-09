use crate::{
    activations::GeLU,
    attention::MultiHeadAttention,
    gpt::GPTConfig,
    layers::{Dropout, Linear},
    normalization::LayerNorm,
};
use candle_core::{Device, Error, Tensor};
use candle_nn::Module;

pub struct FeedForward {
    layer1: Linear,
    gelu: GeLU,
    layer2: Linear,
}

impl FeedForward {
    pub fn new(cfg: GPTConfig) -> Result<Self, Error> {
        let (layer1_result, layer2_result) = rayon::join(
            || Linear::new(cfg.emb_dim, 4 * cfg.emb_dim, true, &Device::Cpu),
            || Linear::new(4 * cfg.emb_dim, cfg.emb_dim, true, &Device::Cpu),
        );

        let layer1 = layer1_result?;
        let layer2 = layer2_result?;

        let gelu = GeLU::new()?;

        Ok(FeedForward {
            layer1,
            gelu,
            layer2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let x = self.layer1.forward(x)?;
        let x = self.gelu.forward(&x)?;
        let x = self.layer2.forward(&x)?;
        Ok(x)
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();

        params.push(&self.layer1.weight);
        if let Some(bias) = &self.layer1.bias {
            params.push(bias);
        }

        params.push(&self.layer2.weight);
        if let Some(bias) = &self.layer2.bias {
            params.push(bias);
        }

        params
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();

        params.push(&mut self.layer1.weight);
        if let Some(bias) = &mut self.layer1.bias {
            params.push(bias);
        }

        params.push(&mut self.layer2.weight);
        if let Some(bias) = &mut self.layer2.bias {
            params.push(bias);
        }

        params
    }
}

pub struct TransformerBlock {
    attention: MultiHeadAttention,
    ff: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    drop_shortcut: Dropout,
}

impl TransformerBlock {
    pub fn new(cfg: &GPTConfig) -> Result<Self, Error> {
        let device = Device::Cpu;

        let attention = MultiHeadAttention::new(
            cfg.emb_dim,
            cfg.emb_dim,
            cfg.context_length,
            cfg.drop_rate,
            cfg.n_heads,
            Some(cfg.qkv_bias),
            device.clone(),
        )?;

        let ff = FeedForward::new(cfg.clone())?;
        let norm1 = LayerNorm::new_default(cfg.emb_dim, &device.clone())?;
        let norm2 = LayerNorm::new_default(cfg.emb_dim, &device.clone())?;
        let drop_shortcut = Dropout::new(cfg.drop_rate);

        Ok(Self {
            attention,
            ff,
            norm1,
            norm2,
            drop_shortcut,
        })
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();

        params.extend(self.attention.parameters());

        params.extend(self.ff.parameters());

        params.push(&self.norm1.scale);
        params.push(&self.norm1.shift);
        params.push(&self.norm2.scale);
        params.push(&self.norm2.shift);
        params
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();

        params.extend(self.attention.parameters_mut());

        params.extend(self.ff.parameters_mut());

        params.push(&mut self.norm1.scale);
        params.push(&mut self.norm1.shift);
        params.push(&mut self.norm2.scale);
        params.push(&mut self.norm2.shift);
        params
    }

    pub fn train(&mut self) {
        self.drop_shortcut.train();
        self.attention.train(); // Now we can control attention dropout too
    }

    pub fn eval(&mut self) {
        self.drop_shortcut.eval();
        self.attention.eval(); // Disable attention dropout during eval
    }

    pub fn set_training(&mut self, training: bool) {
        self.drop_shortcut.set_training(training);
        self.attention.set_training(training);
    }
}

impl Module for TransformerBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let shortcut = x.clone();
        let x = self.norm1.forward(x)?;
        let x = self.attention.forward(&x)?;
        let x = self.drop_shortcut.forward(&x)?;
        let x = x.broadcast_add(&shortcut)?;

        let shortcut = x.clone();
        let x = self.norm2.forward(&x)?;
        let x = self.ff.forward(&x)?;
        let x = self.drop_shortcut.forward(&x)?;
        let x = x.broadcast_add(&shortcut)?;

        Ok(x)
    }
}
