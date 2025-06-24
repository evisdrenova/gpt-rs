use candle_core::{DType, Device, Error, Tensor};
use candle_nn::{Module, Sequential, seq};
use rand::seq;

use crate::{
    attention::parse_batch_and_seq,
    embedding::Embedding,
    layers::{Dropout, Linear},
};

#[derive(Debug)]
pub struct GPTConfig {
    pub context_length: usize,
    pub vocab_size: usize,
    pub output_dim: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub drop_rate: f32,
    pub qkv_bias: bool,
}

pub struct GPT {
    pub tok_emb: Embedding,
    pub pos_emb: Embedding,
    pub drop_emb: Dropout,
    pub trf_blocks: Sequential,
    pub final_norm: LayerNorm,
    pub out_head: Linear,
}

impl GPT {
    pub fn new(cfg: GPTConfig) -> Result<Self, Error> {
        let device = Device::Cpu;

        // create embedding layers
        let tok_emb = Embedding::new(cfg.vocab_size, cfg.emb_dim, device.clone())?;
        let pos_emb = Embedding::new(cfg.context_length, cfg.emb_dim, device.clone())?;

        let drop_emb = Dropout::new(cfg.drop_rate);
        // creates a list of modules and then connects the inputs to ouputs of the models in a feed forward type of way

        let mut trf_blocks: Sequential = seq();
        for _ in 0..cfg.n_layers {
            let block = DummyTransformerBlock::new(&cfg)?;
            trf_blocks = trf_blocks.add(block);
        }

        let final_norm = LayerNorm::new(cfg.emb_dim, 0.00001, &device)?;

        let out_head = Linear::new(cfg.emb_dim, cfg.vocab_size, false, &device)?;

        Ok(GPT {
            tok_emb,
            pos_emb,
            out_head,
            drop_emb,
            trf_blocks,
            final_norm,
        })
    }

    pub fn forward(self, in_indx: Tensor) -> Result<Tensor, Error> {
        let dims = in_indx.shape().dims();
        let (batch_size, seq_len) = parse_batch_and_seq(dims)?;

        let tok_embeds = self.tok_emb.forward(&in_indx)?;
        // we need to output integeres here so that we can use them in the embedding layer
        let arrange = Tensor::arange(0u32, seq_len as u32, &Device::Cpu)?;

        let pos_embeds = self.pos_emb.forward(&arrange)?;
        let embeddings = tok_embeds.broadcast_add(&pos_embeds)?;
        let dropped_embeddings = self.drop_emb.forward(&embeddings)?;
        let transformer_output = self.trf_blocks.forward(&dropped_embeddings)?;
        let normalized_output = self.final_norm.forward(&transformer_output)?;
        let logits = self.out_head.forward(&normalized_output)?;
        Ok(logits)
    }
}

pub struct DummyTransformerBlock {}

impl DummyTransformerBlock {
    pub fn new(_cfg: &GPTConfig) -> Result<Self, Error> {
        Ok(Self {})
    }
}

impl Module for DummyTransformerBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        Ok(x.clone())
    }
}

pub struct LayerNorm {
    eps: f32,
    scale: Tensor,
    shift: Tensor,
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
