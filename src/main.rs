use file_operations::{create_dataloader_v1, load_file};

mod file_operations;
mod simple_tokenizer;

fn main() {
    let file_name = "the-verdict.txt";

    let raw_text = load_file(file_name);

    let dataloader = create_dataloader_v1(&raw_text, 1, 4, 1, false, true).unwrap();

    if let Some(batch_result) = dataloader.iter().next() {
        let (inputs, targets) = batch_result.unwrap();

        let input_vec: Vec<Vec<u32>> = inputs.to_vec2().unwrap();
        let target_vec: Vec<Vec<u32>> = targets.to_vec2().unwrap();

        println!("Input tokens: {:?}", input_vec);
        println!("Target tokens: {:?}", target_vec);
    }
}
