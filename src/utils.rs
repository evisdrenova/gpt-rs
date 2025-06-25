use candle_core::{Error, Tensor};

use crate::{activations::Activations, gpt::GPT};

pub fn parse_batch_and_seq(dims: &[usize]) -> Result<(usize, usize), Error> {
    match dims.len() {
        1 => Ok((1, dims[0])),
        2 => Ok((dims[0], dims[1])),
        3 => Ok((dims[0], dims[1])),
        _ => Err(Error::Msg(format!(
            "Expected 1D, 2D or 3D input, got {}D",
            dims.len()
        ))),
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
