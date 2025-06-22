use candle_core::{Device, Error, Tensor};
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
    pub final_norm: DummyLayerNorm,
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

        let final_norm = DummyLayerNorm::new(cfg.emb_dim, 0.00001)?;

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
        let arrange = Tensor::arange(0f32, seq_len as f32, &Device::Cpu)?;
        let pos_embeds = self.pos_emb.forward(&arrange)?;

        let mut x = tok_embeds + pos_embeds;
        x = self.drop_emb.forward(&x.unwrap());
        x = self.trf_blocks.forward(&x.unwrap());
        x = self.final_norm.forward(&x.unwrap());
        let logits = self.out_head.forward(&x.unwrap())?;
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

pub struct DummyLayerNorm {}

impl DummyLayerNorm {
    pub fn new(dim: usize, eps: f32) -> Result<Self, Error> {
        Ok(Self {})
    }
}

impl Module for DummyLayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        Ok(x.clone())
    }
}
