use candle_core::{Device, Error, Tensor};
use candle_nn::Module;
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
    // pub fn new(cfg: GPTConfig) -> Result<Self, Error> {
    //     let device = Device::Cpu;

    //     let tok_emb = Embedding::new(cfg.vocab_size, cfg.emb_dim, device.clone())?;
    //     let pos_emb = Embedding::new(cfg.context_length, cfg.emb_dim, device.clone())?;

    //     let drop_emb = Dropout::new(cfg.drop_rate);

    //     let mut trf_blocks: Vec<TransformerBlock> = Vec::new();
    //     for _ in 0..cfg.n_layers {
    //         let block: TransformerBlock = TransformerBlock::new(&cfg)?;
    //         trf_blocks.push(block);
    //     }

    //     let final_norm = LayerNorm::new(cfg.emb_dim, 0.00001, &device)?;

    //     let out_head = Linear::new(cfg.emb_dim, cfg.vocab_size, false, &device)?;

    //     Ok(GPT {
    //         tok_emb,
    //         pos_emb,
    //         out_head,
    //         drop_emb,
    //         trf_blocks,
    //         final_norm,
    //     })
    // }

    pub fn new(cfg: GPTConfig) -> Result<Self, Error> {
        let init_start = Instant::now();
        println!("🔧 Starting GPT model initialization...");

        let device = Device::Cpu;

        // Create shared embedding weights once (replaces both token embedding and output head)
        let step_start = Instant::now();
        let shared_emb_weights =
            Tensor::randn(0.0f32, 1.0f32, &[cfg.vocab_size, cfg.emb_dim], &device)?;
        println!(
            "⏱️  Shared embedding weights ({} x {}): {:?}",
            cfg.vocab_size,
            cfg.emb_dim,
            step_start.elapsed()
        );

        // Time token embedding creation (using shared weights - should be ~instant)
        let step_start = Instant::now();
        let tok_emb = Embedding::from_weights(shared_emb_weights.clone())?;
        println!(
            "⏱️  Token embedding (from shared): {:?}",
            step_start.elapsed()
        );

        // Time position embedding creation
        let step_start = Instant::now();
        let pos_emb = Embedding::new(cfg.context_length, cfg.emb_dim, device.clone())?;
        println!(
            "⏱️  Position embedding ({} x {}): {:?}",
            cfg.context_length,
            cfg.emb_dim,
            step_start.elapsed()
        );

        // Time dropout creation
        let step_start = Instant::now();
        let drop_emb = Dropout::new(cfg.drop_rate);
        println!("⏱️  Dropout creation: {:?}", step_start.elapsed());

        // Time transformer blocks creation
        let step_start = Instant::now();
        let mut trf_blocks: Vec<TransformerBlock> = Vec::new();
        println!("🔄 Creating {} transformer blocks...", cfg.n_layers);

        for i in 0..cfg.n_layers {
            let block_start = Instant::now();
            let block: TransformerBlock = TransformerBlock::new(&cfg)?;
            println!("  📦 Block {}: {:?}", i, block_start.elapsed());
            trf_blocks.push(block);
        }
        println!("⏱️  All transformer blocks: {:?}", step_start.elapsed());

        // Time layer norm creation
        let step_start = Instant::now();
        let final_norm = LayerNorm::new(cfg.emb_dim, 0.00001, &device)?;
        println!("⏱️  Final layer norm: {:?}", step_start.elapsed());

        // Time output head creation (using transposed shared weights - should be ~instant)
        let step_start = Instant::now();
        let transposed_weights = shared_emb_weights.t()?; // Transpose: [vocab_size, emb_dim] -> [emb_dim, vocab_size]
        let out_head = Linear::from_weights(transposed_weights, None)?;
        println!(
            "⏱️  Output head (from shared, transposed): {:?}",
            step_start.elapsed()
        );

        println!("✅ Total GPT initialization: {:?}", init_start.elapsed());

        Ok(GPT {
            tok_emb,
            pos_emb,
            out_head,
            drop_emb,
            trf_blocks,
            final_norm,
        })
    }

    pub fn forward(&self, in_indx: &Tensor) -> Result<Tensor, Error> {
        let dims = in_indx.shape().dims();
        let (_, seq_len) = parse_batch_and_seq(dims)?;

        let tok_embeds = self.tok_emb.forward(&in_indx)?;
        let arrange = Tensor::arange(0u32, seq_len as u32, &Device::Cpu)?;

        let pos_embeds = self.pos_emb.forward(&arrange)?;
        let embeddings = tok_embeds.broadcast_add(&pos_embeds)?;

        let mut x = self.drop_emb.forward(&embeddings)?;

        for block in &self.trf_blocks {
            x = block.forward(&x)?;
        }
        let normalized_output = self.final_norm.forward(&x)?;
        let logits = self.out_head.forward(&normalized_output)?;
        Ok(logits)
    }

    pub fn train(&mut self) {
        self.drop_emb.train();

        for block in &mut self.trf_blocks {
            block.train();
        }
    }

    // Set model to evaluation mode (disables dropout)
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
