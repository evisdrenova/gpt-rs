use candle_core::Device;
use embedding::Embedding;
use file_operations::{create_dataloader_v1, load_file};

mod embedding;
mod file_operations;
mod simple_tokenizer;

fn main() {
    let file_name = "the-verdict.txt";

    let raw_text = load_file(file_name);

    let max_length = 4;
    let batch_size = 8;
    let stride = 4;
    let shuffle = false;
    let drop_last = true;

    let dataloader = create_dataloader_v1(
        &raw_text, batch_size, max_length, stride, shuffle, drop_last,
    )
    .unwrap();

    for (batch_idx, batch_result) in dataloader.iter().take(2).enumerate() {
        let (inputs, targets) = batch_result.unwrap();

        let input_vec: Vec<Vec<u32>> = inputs.to_vec2().unwrap();
        let target_vec: Vec<Vec<u32>> = targets.to_vec2().unwrap();

        println!("Batch {}:", batch_idx);
        println!("  Input tokens: {:?}", input_vec);
        println!("  Target tokens: {:?}", target_vec);
        println!("inputs shape: {:?}", inputs.shape());
    }

    let vocab_size: i64 = 50_257;
    let output_dim: i64 = 256;
    let device: Device = Device::Cpu;

    let embed = Embedding::new(vocab_size, output_dim, device);
    let batch_size = 2;
    let seq_len = 5;

    // update to be the token_ids from the batch
    let embeddings = embed.forward(&token_ids);
    println!("embeddings shape: {:?}", embeddings.dims());
}
