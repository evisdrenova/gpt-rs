use candle_core::{DType, Device, Error, Tensor};
use candle_nn::Module;
use file_operations::{create_dataloader_v1, load_file};

use crate::{
    attention::MultiHeadAttention,
    gpt::{FeedForward, GPT, GPTConfig, LayerNorm, TransformerBlock},
    layers::Linear,
};

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

    let max_length = 4;
    let batch_size = 8;
    let stride = 4;
    let shuffle = false;
    let drop_last = true;
    let context_length = 1024;
    let vocab_size = 50257;
    let output_dim = 3;
    let emb_dim = 768;
    let n_heads = 12;
    let n_layers = 12;
    let drop_rate = 0.1;
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
        context_length: 1024,
        vocab_size: 50257,
        output_dim: 3,
        emb_dim: 768,
        n_heads: 12,
        n_layers: 12,
        drop_rate: 0.1,
        qkv_bias: false,
    };

    let batch_example = Tensor::randn(0.0f32, 1.0f32, (2, 5), &Device::Cpu)?;

    println!("1");

    // Create layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    // Note: Candle doesn't have a Sequential container, so we'll do it manually
    let linear_layer = Linear::new(5, 6, false, &Device::Cpu)?;

    println!("2");

    // Forward pass: Linear -> ReLU
    let linear_out = linear_layer.forward(&batch_example)?;
    println!("2.5");
    let out = linear_out.relu()?;

    println!("Sequential output:");
    println!("{:?}", out.to_vec2::<f32>()?);

    println!("3");

    // Create LayerNorm with emb_dim=5
    let ln = LayerNorm::new_default(5, &Device::Cpu)?;

    // Apply LayerNorm: out_ln = ln(batch_example)
    let out_ln = ln.forward(&batch_example)?;

    println!("3");

    // Calculate mean along last dimension with keepdim=True
    let mean = out_ln.mean_keepdim(1)?; // dim=1 is the last dimension for shape [2, 5]

    // Calculate variance along last dimension with keepdim=True, unbiased=False
    let var = out_ln.var_keepdim(1)?;

    println!("\nLayerNorm output:");
    println!("{:?}", out_ln.to_vec2::<f32>()?);

    println!("\nMean:");
    println!("{:?}", mean.to_vec2::<f32>()?);

    println!("\nVariance:");
    println!("{:?}", var.to_vec2::<f32>()?);

    // Verify that mean is close to 0 and variance is close to 1 after LayerNorm
    let mean_values = mean.to_vec2::<f32>()?;
    let var_values = var.to_vec2::<f32>()?;

    println!("\nVerification (should be close to 0 and 1 respectively):");
    println!("Mean values: {:?}", mean_values);
    println!("Var values: {:?}", var_values);

    let ff = FeedForward::new(config.clone())?;

    let x = Tensor::rand(0.0f32, 1.0f32, (2, 3, 768), &Device::Cpu)?;

    let output = ff.forward(&x)?;

    // Print shape
    println!("Output shape: {:?}", output.shape());

    let x = Tensor::rand(0.0f32, 1.0f32, (2, 4, 768), &Device::Cpu)?;

    let block = TransformerBlock::new(&config.clone())?;

    let output = block.forward(&x)?;
    println!("Input shape:  {:?}", x.shape());
    println!("Output shape: {:?}", output.shape());

    let model = GPT::new(config)?;

    let total_params = model.parameter_count();
    println!("Total number of parameters: {:?}", total_params);

    //make this way faster its really slow right now
    let logits = model.forward(batch)?;
    println!("Logits shape: {:?}", logits.shape());

    // Slice to get only first 5 values in the last dimension
    let logits_slice = logits.narrow(2, 0, 5)?; // (dim=2, start=0, length=5)
    let logits_vec: Vec<Vec<Vec<f32>>> = logits_slice.to_vec3::<f32>()?;
    println!("First 5 logits: {:?}", logits_vec);

    Ok(())
}
