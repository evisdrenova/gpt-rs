use candle_core::{Device, Error, Tensor};
use candle_nn::Module;
use rayon::prelude::*;
use std::time::Instant;

use crate::{
    embedding::Embedding,
    layers::{Dropout, Linear},
    neural_net::TransformerBlock,
    normalization::LayerNorm,
    utils::parse_batch_and_seq,
};

#[derive(Debug, Clone)]
pub struct GPTConfig {
    pub context_length: usize,
    pub vocab_size: usize,
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
    pub trf_blocks: Vec<TransformerBlock>,
    pub final_norm: LayerNorm,
    pub out_head: Linear,
}

impl GPT {
    pub fn new(cfg: GPTConfig) -> Result<Self, Error> {
        let device = Device::Cpu;

        let tok_emb = Embedding::new(cfg.vocab_size, cfg.emb_dim, device.clone())?;
        let pos_emb = Embedding::new(cfg.context_length, cfg.emb_dim, device.clone())?;
        let drop_emb = Dropout::new(cfg.drop_rate);

        let trf_blocks: Result<Vec<TransformerBlock>, Error> = (0..cfg.n_layers)
            .into_par_iter()
            .map(|i| {
                let block = TransformerBlock::new(&cfg)?;
                Ok(block)
            })
            .collect();

        let trf_blocks = trf_blocks?;
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

    pub fn forward(&self, in_idx: &Tensor) -> Result<Tensor, Error> {
        let forward_start = Instant::now();

        // Time token embeddings
        let emb_start = Instant::now();
        let batch_size = in_idx.dim(0)?;
        let seq_len = in_idx.dim(1)?;
        let tok_embeds = self.tok_emb.forward(in_idx)?;
        let emb_time = emb_start.elapsed();

        // Time position embeddings
        let pos_start = Instant::now();
        let pos_indices = Tensor::arange(0, seq_len as i64, &Device::Cpu)?;
        let pos_embeds = self.pos_emb.forward(&pos_indices)?;
        let pos_time = pos_start.elapsed();

        // Time embedding addition
        let add_start = Instant::now();
        let x = tok_embeds.broadcast_add(&pos_embeds)?;
        let x = self.drop_emb.forward(&x)?;
        let add_time = add_start.elapsed();

        // Time transformer blocks (likely the biggest bottleneck)
        let blocks_start = Instant::now();
        let mut x = x;
        for (i, block) in self.trf_blocks.iter().enumerate() {
            let block_start = Instant::now();
            x = block.forward(&x)?;
            println!("    📦 Block {} forward: {:?}", i, block_start.elapsed());
        }
        let blocks_time = blocks_start.elapsed();

        // Time final operations
        let final_start = Instant::now();
        let x = self.final_norm.forward(&x)?;
        let logits = self.out_head.forward(&x)?;
        let final_time = final_start.elapsed();

        let total_forward_time = forward_start.elapsed();

        println!(
            "  🧠 Forward pass: {:?} (emb: {:?}, pos: {:?}, add: {:?}, blocks: {:?}, final: {:?})",
            total_forward_time, emb_time, pos_time, add_time, blocks_time, final_time
        );

        Ok(logits)
    }

    pub fn train(&mut self) {
        self.drop_emb.train();

        for block in &mut self.trf_blocks {
            block.train();
        }
    }

    pub fn eval(&mut self) {
        self.drop_emb.eval();

        for block in &mut self.trf_blocks {
            block.eval();
        }
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();

        params.extend(self.tok_emb.parameters());
        params.extend(self.pos_emb.parameters());

        for block in &self.trf_blocks {
            params.extend(block.parameters());
        }

        params.push(&self.final_norm.scale);
        params.push(&self.final_norm.shift);

        params.push(&self.out_head.weight);
        if let Some(bias) = &self.out_head.bias {
            params.push(bias);
        }

        params
    }

    pub fn parameter_count(&self) -> usize {
        self.parameters().iter().map(|p| p.elem_count()).sum()
    }
}
