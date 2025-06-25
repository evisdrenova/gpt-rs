use candle_core::{Device, Tensor};

use crate::{
    gpt::{GPT, GPTConfig},
    utils::generate_text_loop,
};

mod activations;
mod attention;
mod embedding;
mod file_operations;
mod gpt;
mod layers;
mod module_list;
mod neural_net;
mod normalization;
mod rng;
mod simple_tokenizer;
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = GPTConfig {
        context_length: 1024,
        vocab_size: 50257,
        emb_dim: 768,
        n_heads: 12,
        n_layers: 12,
        drop_rate: 0.1,
        qkv_bias: false,
    };

    let tokenizer = tiktoken_rs::r50k_base()?;

    let start_text = "Hello, I am";

    let encoded = tokenizer.encode_with_special_tokens(start_text);

    println!("encoded: {:?}", encoded);

    let encoded_tensor = Tensor::from_vec(encoded.clone(), (1, encoded.len()), &Device::Cpu)?;

    println!("encoded_tensor.shape: {:?}", encoded_tensor.shape());

    let mut model = GPT::new(config.clone())?;

    // // For training
    // model.train(); // Enables dropout

    // For inference/evaluation
    model.eval(); // Disables dropout (like your Python code)

    // Now generate text without dropout
    let out = generate_text_loop(&model, encoded_tensor.clone(), 6, config.context_length)?;

    println!("=== Generation Results ===");
    println!("Output tensor: {:?}", out);
    println!("Output length: {}", out.dim(1)?);
    println!("Output shape: {:?}", out.shape());

    // Show original vs generated tokens
    let output_tokens: Vec<u32> = out.squeeze(0)?.to_vec1()?;
    let original_tokens: Vec<u32> = encoded_tensor.squeeze(0)?.to_vec1()?;

    println!("Original tokens: {:?}", original_tokens);
    println!(
        "Generated tokens: {:?}",
        &output_tokens[original_tokens.len()..]
    );
    println!("All tokens: {:?}", output_tokens);

    // Decode text
    let decoded_text = tokenizer.decode(output_tokens)?;
    println!("Generated text: {}", decoded_text);

    Ok(())
}
