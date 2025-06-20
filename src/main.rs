use candle_core::{DType, Device, Error, Tensor};
use file_operations::{create_dataloader_v1, load_file};

use crate::attention::{CausalAttention, MultiHeadAttentionWrapper};

mod attention;
mod embedding;
mod file_operations;
mod layers;
mod module_list;
mod rng;
mod simple_tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_name = "the-verdict.txt";

    let raw_text = load_file(file_name);

    let max_length: i64 = 4;
    let batch_size = 8;
    let stride = 4;
    let shuffle = false;
    let drop_last = true;
    let context_length = 4;
    let vocab_size: i64 = 50267;
    let output_dim: i64 = 3;

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

    let nn_layer = CausalAttention::new(d_in, d_out, &device, context_length, 0.0, None)?;

    let ca = CausalAttention::forward(&nn_layer, &batch)?;

    println!("ca:{:?}", ca);

    let mha = MultiHeadAttentionWrapper::new(d_in, d_out, device, context_length, 0.0, 2, None)?;

    let cv = mha.forward(&batch)?;

    println!("sv:{:?}", cv);

    Ok(())
}
