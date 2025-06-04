use candle_core::Device;
use embedding::Embedding;
use file_operations::{create_dataloader_v1, load_file};

mod embedding;
mod file_operations;
mod simple_tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_name = "the-verdict.txt";

    let raw_text = load_file(file_name);

    let max_length = 4;
    let batch_size = 8;
    let stride = 4;
    let shuffle = false;
    let drop_last = true;

    let dataloader = create_dataloader_v1(
        &raw_text, batch_size, max_length, stride, shuffle, drop_last,
    )?;

    for (batch_idx, batch_result) in dataloader.iter().take(2).enumerate() {
        let (inputs, targets) = batch_result?;

        let input_vec: Vec<Vec<u32>> = inputs.to_vec2()?;
        let target_vec: Vec<Vec<u32>> = targets.to_vec2()?;

        println!("Batch {}:", batch_idx);
        println!("  Input tokens: {:?}", input_vec);
        println!("  Target tokens: {:?}", target_vec);
        println!("inputs shape: {:?}", inputs.shape());
    }

    let vocab_size: i64 = 6;
    let output_dim: i64 = 3;
    let device = Device::Cpu;

    let ids: Vec<i64> = vec![3];
    let input_ids = candle_core::Tensor::new(ids, &device)?;

    let embed = Embedding::new(vocab_size, output_dim, device)?;

    println!("Embedding weights:");
    let weights_vec: Vec<Vec<f32>> = embed.weights.to_vec2()?;
    for (i, row) in weights_vec.iter().enumerate() {
        println!("  [{}, :] = {:?}", i, row);
    }
    println!("Weights shape: {:?}", embed.weights.dims());

    let embeddings = embed.forward(&input_ids)?;
    println!("input_ids shape: {:?}", input_ids.dims());
    println!("embeddings shape: {:?}", embeddings.dims());

    Ok(())
}
