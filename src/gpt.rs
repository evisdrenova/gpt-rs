use candle_core::{Device, Error, Tensor};
use candle_nn::Module;

use crate::{
    activations::{Activations, GeLU},
    attention::{MultiHeadAttention, parse_batch_and_seq},
    embedding::Embedding,
    layers::{Dropout, Linear},
    normalization::LayerNorm,
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

        let mut trf_blocks: Vec<TransformerBlock> = Vec::new();
        for _ in 0..cfg.n_layers {
            let block: TransformerBlock = TransformerBlock::new(&cfg)?;
            trf_blocks.push(block);
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

    pub fn forward(&self, in_indx: &Tensor) -> Result<Tensor, Error> {
        let dims = in_indx.shape().dims();
        let (_, seq_len) = parse_batch_and_seq(dims)?;

        let tok_embeds = self.tok_emb.forward(&in_indx)?;
        // we need to output integers here so that we can use them in the embedding layer
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

pub struct FeedForward {
    layer1: Linear,
    gelu: GeLU,
    layer2: Linear,
}

impl FeedForward {
    pub fn new(cfg: GPTConfig) -> Result<Self, Error> {
        let layer1 = Linear::new(cfg.emb_dim, 4 * cfg.emb_dim, true, &Device::Cpu)?;
        let gelu = GeLU::new()?;
        let layer2 = Linear::new(4 * cfg.emb_dim, cfg.emb_dim, true, &Device::Cpu)?;

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

pub fn generate_text_simple(
    model: &GPT,
    mut idx: Tensor,
    max_new_tokens: usize,
    context_size: usize,
) -> Result<Tensor, Error> {
    // Generate tokens one by one
    for _ in 0..max_new_tokens {
        // Crop current context if it exceeds the supported context size
        // if LLM supports only 5 tokens, and the context size is 10,
        // then only the last 5 tokens are used as context
        let seq_len = idx.dim(1)?;
        let idx_cond = if seq_len > context_size {
            let start_idx = seq_len - context_size;
            idx.narrow(1, start_idx, context_size)?
        } else {
            idx.clone()
        };

        let logits = model.forward(&idx_cond)?;

        let last_token_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?;
        let last_token_logits = last_token_logits.squeeze(1)?;

        let probas = Activations::softmax(&last_token_logits, Some(1))?;

        let idx_next = probas.argmax_keepdim(1)?;
        let idx_next = idx_next.unsqueeze(1)?;

        idx = Tensor::cat(&[idx, idx_next], 1)?;
    }

    Ok(idx)
}
