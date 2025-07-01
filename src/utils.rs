use std::{cmp::min_by, f32};

use crate::{
    activations::Activations,
    file_operations::{DataLoader, DataLoaderIterator},
    gpt::GPT,
    losses::cross_entropy_loss,
};
use candle_core::{Device, Error, Tensor};
use std::time::Instant;
use tiktoken_rs::CoreBPE;

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

// pub fn generate_text_loop(
//     model: &GPT,
//     mut idx: Tensor,
//     max_new_tokens: usize,
//     context_size: usize,
// ) -> Result<Tensor, Error> {
//     // Generate tokens one by one
//     for _ in 0..max_new_tokens {
//         // Crop current context if it exceeds the supported context size
//         let seq_len = idx.dim(1)?;
//         let idx_cond = if seq_len > context_size {
//             let start_idx = seq_len - context_size;
//             idx.narrow(1, start_idx, context_size)?
//         } else {
//             idx.clone()
//         };

//         let logits = model.forward(&idx_cond)?;

//         let last_token_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?;

//         let last_token_logits = last_token_logits.squeeze(1)?;

//         let probas = Activations::softmax(&last_token_logits, Some(last_token_logits.rank() - 1))?;

//         let idx_next = probas.argmax(probas.rank() - 1)?;

//         let idx_next = idx_next.unsqueeze(1)?;

//         idx = Tensor::cat(&[idx, idx_next], 1)?;
//     }
//     Ok(idx)
// }

pub fn generate_text_loop(
    model: &GPT,
    mut idx: Tensor,
    max_new_tokens: usize,
    context_size: usize,
) -> Result<Tensor, Error> {
    let generation_start = Instant::now();
    println!(
        "🔄 Starting text generation for {} tokens...",
        max_new_tokens
    );

    // Generate tokens one by one
    for i in 0..max_new_tokens {
        let token_start = Instant::now();

        // Time context cropping
        let crop_start = Instant::now();
        let seq_len = idx.dim(1)?;
        let idx_cond = if seq_len > context_size {
            let start_idx = seq_len - context_size;
            idx.narrow(1, start_idx, context_size)?
        } else {
            idx.clone()
        };
        let crop_time = crop_start.elapsed();

        // Time model forward pass (likely the bottleneck)
        let forward_start = Instant::now();
        let logits = model.forward(&idx_cond)?;
        let forward_time = forward_start.elapsed();

        // Time logits processing
        let logits_start = Instant::now();
        let last_token_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?;
        let last_token_logits = last_token_logits.squeeze(1)?;
        let logits_time = logits_start.elapsed();

        // Time softmax
        let softmax_start = Instant::now();
        let probas = Activations::softmax(&last_token_logits, Some(last_token_logits.rank() - 1))?;
        let softmax_time = softmax_start.elapsed();

        // Time sampling
        let sampling_start = Instant::now();
        let idx_next = probas.argmax(probas.rank() - 1)?;
        let idx_next = idx_next.unsqueeze(1)?;
        let sampling_time = sampling_start.elapsed();

        // Time concatenation
        let concat_start = Instant::now();
        idx = Tensor::cat(&[idx, idx_next], 1)?;
        let concat_time = concat_start.elapsed();

        let total_token_time = token_start.elapsed();

        println!(
            "  🔢 Token {}: {:?} (forward: {:?}, crop: {:?}, logits: {:?}, softmax: {:?}, sample: {:?}, concat: {:?})",
            i + 1,
            total_token_time,
            forward_time,
            crop_time,
            logits_time,
            softmax_time,
            sampling_time,
            concat_time
        );
    }

    let total_time = generation_start.elapsed();
    let avg_per_token = total_time.as_secs_f64() / max_new_tokens as f64;

    println!("⏱️  Total generation time: {:?}", total_time);
    println!("⏱️  Average per token: {:.3}s", avg_per_token);

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

pub struct ModelTrainResponse {
    pub train_losses: Vec<f32>,
    pub validation_losses: Vec<f32>,
    pub track_tokens_seen: Vec<usize>,
}

pub fn train_model_simple(
    model: &mut GPT,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: &mut Optimizer,
    device: &Device,
    num_epochs: usize,
    eval_freq: usize,
    eval_iter: usize,
    start_context: &str,
    tokenizer: &CoreBPE,
) -> Result<ModelTrainResponse, Error> {
    let mut train_losses: Vec<f32> = Vec::new();
    let mut validation_losses: Vec<f32> = Vec::new();
    let mut track_tokens_seen: Vec<usize> = Vec::new();
    let mut tokens_seen: usize = 0;
    let mut global_step: usize = 0;

    for e in 0..num_epochs {
        model.train();

        for (batch_idx, batch_result) in train_loader.iter().enumerate() {
            let (inputs, targets) = batch_result?;

            optimizer.zero_grad();
            let loss = calc_loss_batch(&inputs, &targets, model, device)?;
            loss.backward();
            optimizer.step();

            tokens_seen += inputs.numel();
            global_step += 1;

            if global_step % eval_freq == 0 {
                let (train_loss, validation_loss) =
                    evaluate_model(model, &train_loader, &validation_loader, device, eval_iter)?;

                train_losses.push(train_loss);
                validation_losses.push(validation_loss);
                track_tokens_seen.push(tokens_seen);
                println!(
                    "Ep {} (Step {:06}): Train loss {:.3}, Val loss {:.3}",
                    e + 1,
                    global_step,
                    train_loss,
                    validation_loss
                );

                generate_and_print_sample(model, tokenizer, device, start_context)?;
            }
        }
    }

    Ok(ModelTrainResponse {
        train_losses,
        validation_losses,
        track_tokens_seen,
    })
}

/// prints training and validation loss after each model update
pub fn evaluate_model(
    model: &mut GPT,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: &Device,
    eval_iter: usize,
) -> Result<(f32, f32), Error> {
    model.eval();
    let train_loss = calc_loss_loader(train_loader, model, device, eval_iter)?;

    let validation_loss = calc_loss_loader(validation_loader, model, device, eval_iter)?;

    model.train();
    Ok((train_loss, validation_loss))
}

pub fn generate_and_print_sample(
    model: &mut GPT,
    tokenizer: &CoreBPE,
    device: &Device,
    start_context: &str,
) -> Result<String, Error> {
    model.eval();
    let context_size = model.pos_emb.weights.shape().dims()[0];

    let encoded = text_to_token_ids(start_context, tokenizer)?;

    let token_ids = generate_text_loop(model, encoded, 50, context_size)?;

    let decoded_text = token_ids_to_text(&token_ids, tokenizer)?;

    let output = decoded_text.replace("Än", " ");
    println!("{}", output);

    model.train();
    Ok(output)
}
