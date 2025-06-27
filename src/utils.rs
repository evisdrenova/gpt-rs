use std::{cmp::min_by, f32};

use candle_core::{Device, Error, Tensor};
use tiktoken_rs::CoreBPE;

use crate::{
    activations::Activations,
    file_operations::{DataLoader, DataLoaderIterator},
    gpt::GPT,
    losses::cross_entropy_loss,
};

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

pub fn generate_text_loop(
    model: &GPT,
    mut idx: Tensor,
    max_new_tokens: usize,
    context_size: usize,
) -> Result<Tensor, Error> {
    // Generate tokens one by one
    for _ in 0..max_new_tokens {
        // Crop current context if it exceeds the supported context size
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

        let probas = Activations::softmax(&last_token_logits, Some(last_token_logits.rank() - 1))?;

        let idx_next = probas.argmax(probas.rank() - 1)?;

        let idx_next = idx_next.unsqueeze(1)?;

        idx = Tensor::cat(&[idx, idx_next], 1)?;
    }
    Ok(idx)
}

pub fn text_to_token_ids(text: &str, tokenizer: &CoreBPE) -> Result<Tensor, Error> {
    let encoded = tokenizer.encode_with_special_tokens(text);
    let encoded_tensor = Tensor::from_vec(encoded.clone(), (1, encoded.len()), &Device::Cpu)?;
    Ok(encoded_tensor)
}

pub fn token_ids_to_text(token_ids: &Tensor, tokenizer: &CoreBPE) -> Result<String, Error> {
    let flat = token_ids.squeeze(0)?;
    let vec: Vec<u32> = flat.to_vec1()?;
    let output = tokenizer
        .decode(vec)
        .map_err(|e| Error::Msg(format!("Tokenizer decode error: {}", e)))?;
    Ok(output)
}

pub fn calc_loss_batch(
    input: &Tensor,
    target: &Tensor,
    model: &GPT,
    device: &Device,
) -> Result<Tensor, Error> {
    let input = input.to_device(&device)?;
    let target = target.to_device(&device)?;

    let logits = model.forward(&input)?;

    let logits_flat = logits.flatten(0, 1)?;
    let target_flat = target.flatten_all()?;

    let loss = cross_entropy_loss(&logits_flat, &target_flat)?;
    Ok(loss)
}

pub fn calc_loss_loader(
    data_loader: DataLoader,
    model: &GPT,
    device: &Device,
    num_batches: usize,
) -> Result<f32, Error> {
    let ds_len = data_loader.len();
    if ds_len == 0 {
        return Ok(f32::NAN);
    }

    let num_batches = if num_batches == 0 {
        ds_len
    } else {
        num_batches.min(ds_len)
    };

    let mut total_loss: f32 = 0.0;

    for (batch_idx, batch_res) in data_loader.iter().enumerate() {
        if batch_idx >= num_batches {
            break;
        }

        let (input, target) =
            batch_res.map_err(|e| Error::Msg(format!("DataLoader error: {}", e)))?;
        let loss_tensor = calc_loss_batch(&input, &target, model, device)?;
        let loss_value: f32 = loss_tensor.to_scalar::<f32>()?;

        println!("Batch {} loss = {}", batch_idx, loss_value);

        total_loss += loss_value;
    }
    Ok(total_loss / (num_batches as f32))
}
