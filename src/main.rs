use candle_core::{Device, Tensor};

use crate::{
    file_operations::split_train_validation,
    gpt::{GPT, GPTConfig},
    utils::{generate_text_loop, token_ids_to_text},
};
use file_operations::{create_dataloader_v1, load_file};

mod activations;
mod attention;
mod embedding;
mod file_operations;
mod gpt;
mod layers;
mod losses;
mod module_list;
mod neural_net;
mod normalization;
mod rng;
mod simple_tokenizer;
mod utils;

pub const GPT_CONFIG_124M: GPTConfig = GPTConfig {
    context_length: 256,
    vocab_size: 50257,
    emb_dim: 768,
    n_heads: 12,
    n_layers: 12,
    drop_rate: 0.1,
    qkv_bias: false,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let max_length: i64 = 4;
    let batch_size = 8;
    let stride = 4;

    let file_name = "the-verdict.txt";
    let raw_text = load_file(file_name);

    let tokenizer = tiktoken_rs::r50k_base()?;
    let total_characters = raw_text.len();
    let encoded_tokens = tokenizer.encode_with_special_tokens(&raw_text);
    let total_tokens = encoded_tokens.len();

    println!("=== Text Analysis ===");
    println!("Characters: {}", total_characters);
    println!("Tokens: {}", total_tokens);

    let (train, validation) = split_train_validation(&raw_text, 0.9)?;

    let train_loader = create_dataloader_v1(
        train,
        2,
        GPT_CONFIG_124M.context_length,
        GPT_CONFIG_124M.context_length,
        true,
        true,
    )?;

    let val_loader = create_dataloader_v1(
        validation,
        2,
        GPT_CONFIG_124M.context_length,
        GPT_CONFIG_124M.context_length,
        false,
        false,
    )?;

    println!("train length{}", train.len());
    println!("validatiohn length{}", validation.len());

    println!("Train loader:");
    for (batch_idx, batch_result) in train_loader.iter().enumerate() {
        let (inputs, targets) = batch_result?;
        println!(
            "torch.Size({:?}) torch.Size({:?})",
            inputs.shape(),
            targets.shape()
        );
    }

    println!("\nValidation loader:");
    for (batch_idx, batch_result) in val_loader.iter().enumerate() {
        let (inputs, targets) = batch_result?;
        println!(
            "torch.Size({:?}) torch.Size({:?})",
            inputs.shape(),
            targets.shape()
        );
    }

    // let tokenizer = tiktoken_rs::r50k_base()?;

    // let start_text = "Every effort moves you ";

    // let encoded = tokenizer.encode_with_special_tokens(start_text);

    // println!("encoded: {:?}", encoded);

    // let encoded_tensor = Tensor::from_vec(encoded.clone(), (1, encoded.len()), &Device::Cpu)?;

    // println!("encoded_tensor.shape: {:?}", encoded_tensor.shape());

    // let mut model = GPT::new(GPT_CONFIG_124M.clone())?;

    // // For training
    // model.train(); // Enables dropouts
    // For inference/evaluation
    // model.eval(); // Disables dropout (like your Python code)

    // // Now generate text without dropout
    // let out: Tensor = generate_text_loop(
    //     &model,
    //     encoded_tensor.clone(),
    //     10,
    //     GPT_CONFIG_124M.context_length,
    // )?;

    // println!("=== Generation Results ===");
    // println!("Output tensor: {:?}", out);
    // println!("Output length: {}", out.dim(1)?);
    // println!("Output shape: {:?}", out.shape());

    // println!("output text:{:?}", token_ids_to_text(&out, &tokenizer)?);

    // // Show original vs generated tokens
    // let output_tokens: Vec<u32> = out.squeeze(0)?.to_vec1()?;
    // let original_tokens: Vec<u32> = encoded_tensor.squeeze(0)?.to_vec1()?;

    // println!("Original tokens: {:?}", original_tokens);
    // println!(
    //     "Generated tokens: {:?}",
    //     &output_tokens[original_tokens.len()..]
    // );
    // println!("All tokens: {:?}", output_tokens);

    // // Decode text
    // let decoded_text = tokenizer.decode(output_tokens)?;
    // println!("Generated text: {}", decoded_text);

    Ok(())
}
