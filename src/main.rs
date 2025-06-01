use file_operations::{create_dataloader_v1, load_file};

mod file_operations;
mod simple_tokenizer;

fn main() {
    let file_name = "the-verdict.txt";

    let raw_text = load_file(file_name);

    let dataloader = create_dataloader_v1(&raw_text, 1, 4, 1, false, true).unwrap();

    for (batch_idx, batch_result) in dataloader.iter().take(2).enumerate() {
        let (inputs, targets) = batch_result.unwrap();

        let input_vec: Vec<Vec<u32>> = inputs.to_vec2().unwrap();
        let target_vec: Vec<Vec<u32>> = targets.to_vec2().unwrap();

        println!("Batch {}:", batch_idx);
        println!("  Input tokens: {:?}", input_vec);
        println!("  Target tokens: {:?}", target_vec);
        println!();
    }
}
