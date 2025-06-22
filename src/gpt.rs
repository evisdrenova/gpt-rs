use candle_core::{Device, Error, Tensor};
use candle_nn::{LayerNorm, Sequential};

use crate::{
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
    pub trf_blocks: usize,
    pub final_norm: LayerNorm,
    pub out_head: Linear,
}

impl GPT {
    pub fn new(cfg: GPTConfig) -> Result<Self,Error> {
        let device = Device::Cpu;

        // create embedding layers
        let tok_emb = Embedding::new(cfg.vocab_size, cfg.emb_dim, device)?;
        let pos_emb = Embedding::new(cfg.context_length, cfg.emb_dim, device)?;

        let drop_emb = Dropout::new(cfg.drop_rate);


        // creates a list of modules and then connects the inputs to ouputs of the models in a feed forward type of way
        let trf_blocks = Sequential



        Ok(GPT { tok_emb })
    }
}


pub struct DummyTransformerModel {

}

impl DummyTransformerModel {
    pub fn new(cfg: GPTConfig) -> Result<Self,Error>{Ok(DummyTransformerModel {  })}

    pub fn forward(self, input:Tensor) -> Result<Tensor,Error>{Ok(input)}
}

pub struct DummyLayerNorm {

}

impl DummyLayerNorm {
    pub fn new( norm_shape:usize, eps:f32) -> Result<Self,Error>{Ok(DummyLayerNorm {  })}

    pub fn forward(self, input:Tensor) -> Result<Tensor,Error>{Ok(input)}
}