use candle_core::{Device, Tensor};
use file_operations::{create_dataloader_v1, load_file};
use gpt_rs::gpt::{GPT, GPTConfig};

use crate::attention::MultiHeadAttention;

mod attention;
mod embedding;
mod file_operations;
mod layers;
mod module_list;
mod rng;
mod simple_tokenizer;

mod gpt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_name = "the-verdict.txt";

    let raw_text = load_file(file_name);

    let max_length: i64 = 4;
    let batch_size = 8;
    let stride = 4;
    let shuffle = false;
    let drop_last = true;
    let context_length = 1024;
    let vocab_size: i64 = 50257;
    let output_dim: i64 = 3;
    let emb_dim = 768;
    let n_heads = 12;
    let n_layers = 12;
    let drop_rate: f32 = 0.1;
    let qkv_bias = false;

    let dataloader = create_dataloader_v1(
        &raw_text,
        batch_size,
        max_length as usize,
        stride,
        shuffle,
        drop_last,
    )?;

    let data = vec![
        0.43, 0.15, 0.89, // Your (x^1)
        0.55, 0.87, 0.66, // journey (x^2)
        0.57, 0.85, 0.64, // starts (x^3)
        0.22, 0.58, 0.33, // with (x^4)
        0.77, 0.25, 0.10, // one (x^5)
        0.05, 0.80, 0.55, // step (x^6)
    ];

    let inputs: Tensor = Tensor::from_vec(data, (6, 3), &Device::Cpu)?;

    let inputs = inputs.to_dtype(candle_core::DType::F32)?;

    let device = Device::Cpu;
    let d_in = 3;
    let d_out = 2;

    let batch = Tensor::stack(&[&inputs, &inputs], 0)?;

    let context_length = batch.shape().dims()[1];

    let nn_layer = MultiHeadAttention::new(d_in, d_out, context_length, 0.0, 2, None, device)?;

    let ca = MultiHeadAttention::forward(&nn_layer, &batch)?;

    println!("ca:{:?}", ca);

    let mha = MultiHeadAttention::new(d_in, d_out, context_length, 0.0, 2, None, Device::Cpu)?;

    let cv = mha.forward(&batch)?;

    println!("sv:{:?}", cv);

    let values: Vec<Vec<Vec<f32>>> = cv.to_vec3()?; // For 3D tensor

    println!("Tensor values: {:?}", values);

    let tokenizer = tiktoken_rs::r50k_base()?;

    let txt1 = "Every effort moves you";

    let txt2 = "Every day holds a";

    let txt1_tokens = tokenizer.encode_with_special_tokens(txt1);

    let txt2_tokens = tokenizer.encode_with_special_tokens(txt2);

    let txt1_tensor = Tensor::from_vec(txt1_tokens.clone(), (txt1_tokens.len(),), &Device::Cpu)?;

    let txt2_tensor = Tensor::from_vec(txt2_tokens.clone(), (txt2_tokens.len(),), &Device::Cpu)?;

    let batch = Tensor::stack(&[&txt1_tensor, &txt2_tensor], 0)?;

    println!("batch:{:?}", batch);
    let batch_vec: Vec<Vec<u32>> = batch.to_vec2::<u32>()?;
    println!("batch as vec: {:?}", batch_vec);

    let config = GPTConfig {
        context_length: context_length,
        vocab_size: 50257,
        output_dim: 3,
        emb_dim: 768,
        n_heads: 12,
        n_layers: 12,
        drop_rate: 0.1,
        qkv_bias: false,
    };

    let model = GPT::new(config)?;

    println!("1");
    let logits = model.forward(batch)?;
    println!("Logits shape: {:?}", logits.shape());

    // Slice to get only first 5 values in the last dimension
    let logits_slice = logits.narrow(2, 0, 5)?; // (dim=2, start=0, length=5)
    let logits_vec: Vec<Vec<Vec<f32>>> = logits_slice.to_vec3::<f32>()?;
    println!("First 5 logits: {:?}", logits_vec);

    Ok(())
}
