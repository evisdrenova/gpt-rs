use candle_core::{Device, Error, Tensor};
use embedding::Embedding;
use file_operations::{create_dataloader_v1, load_file};
use tiktoken_rs::r50k_base;

mod embedding;
mod file_operations;
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

    // only does one batch
    if let Some(batch_result) = dataloader.iter().next() {
        let (inputs, targets) = batch_result?;

        // let bpe = r50k_base().unwrap();

        // let new_text = "Your joueny starts with one step";

        // let tokens = bpe.encode_with_special_tokens(new_text);

        // println!("encode: {:?}", tokens);

        let input_vec: Vec<Vec<u32>> = inputs.to_vec2()?;
        let target_vec: Vec<Vec<u32>> = targets.to_vec2()?;

        println!("  Input tokens: {:?}", input_vec);
        println!("  Target tokens: {:?}", target_vec);
        println!("inputs shape: {:?}", inputs.shape());

        // create token embedding layer
        let token_embedding_layer = Embedding::new(vocab_size, output_dim, Device::Cpu)?;
        let token_embeddings = token_embedding_layer.forward(&inputs)?;

        // creates positional token embedding layer
        let pos_embedding_layer = Embedding::new(context_length, output_dim, Device::Cpu)?;

        let placeholder_vector = Tensor::arange(0i64, context_length as i64, &Device::Cpu);

        let pos_emebeddings = pos_embedding_layer.forward(&placeholder_vector?)?;

        println!("embeddings shape: {:?}", token_embeddings.dims());
        println!("positional embedding shape: {:?}", pos_emebeddings.dims());

        let result_with_broadcast = token_embeddings.broadcast_add(&pos_emebeddings)?;
        println!(
            "Broadcasted result shape: {:?}",
            result_with_broadcast.dims()
        );
    }

    // runs n batches as specificed in the .take()
    // alterantively can do .iter().enumerate() to do all batches
    // for (batch_idx, batch_result) in dataloader.iter().take(2).enumerate() {
    //     let (inputs, targets) = batch_result?;

    //     let input_vec: Vec<Vec<u32>> = inputs.to_vec2()?;
    //     let target_vec: Vec<Vec<u32>> = targets.to_vec2()?;

    //     println!("Batch {}:", batch_idx);
    //     println!("  Input tokens: {:?}", input_vec);
    //     println!("  Target tokens: {:?}", target_vec);
    //     println!("inputs shape: {:?}", inputs.shape());

    //     let embed = Embedding::new(vocab_size, output_dim, Device::Cpu)?;
    //     let embeddings = embed.forward(&inputs)?;
    //     println!("embeddings shape: {:?}", embeddings.dims());
    // }

    // let ids: Vec<i64> = vec![3];
    // let input_ids = candle_core::Tensor::new(ids, &Device::Cpu)?;

    // let embed = Embedding::new(vocab_size, output_dim, Device::Cpu)?;

    // // Print the embedding weights (like embedding_layer.weight in PyTorch)
    // // println!("Embedding weights:");
    // // let weights_vec: Vec<Vec<f32>> = embed.weights.to_vec2()?;
    // // for (i, row) in weights_vec.iter().enumerate() {
    // //     println!("  [{}, :] = {:?}", i, row);
    // // }
    // // println!("Weights shape: {:?}", embed.weights.dims());

    // // Create 1D tensor just like PyTorch: torch.tensor([2, 3, 5, 1])
    // let ids: Vec<i64> = vec![2, 3, 5, 1];
    // let input_ids = Tensor::new(ids, &Device::Cpu)?;

    // // This will now work with 1D input just like PyTorch
    // let embeddings = embed.forward(&input_ids)?;
    // println!("input_ids shape: {:?}", input_ids.dims());
    // println!("embeddings shape: {:?}", embeddings.dims());

    let data = vec![
        0.43, 0.15, 0.89, // Your (x^1)
        0.55, 0.87, 0.66, // journey (x^2)
        0.57, 0.85, 0.64, // starts (x^3)
        0.22, 0.58, 0.33, // with (x^4)
        0.77, 0.25, 0.10, // one (x^5)
        0.05, 0.80, 0.55, // step (x^6)
    ];

    // Create tensor with shape [6, 3] (6 rows, 3 columns)
    let inputs: Tensor = Tensor::from_vec(data, (6, 3), &Device::Cpu)?;

    // Print the tensor
    println!("Inputs tensor:");
    println!("{}", inputs);

    // get the first index in the inputs tensor which is the tokenized word "journey"
    let query = inputs.get(1);

    let query_tensor = query?;

    let output = compute_intermediate_attention_scores(&inputs, &query_tensor)?;

    println!("print the attention scores: {:?}", output);

    Ok(())
}

fn compute_intermediate_attention_scores(inputs: &Tensor, query: &Tensor) -> Result<Tensor, Error> {
    let atten_scores = inputs.matmul(&query.unsqueeze(1)?)?;
    let attn_scores_flat = atten_scores.flatten_all()?;

    Ok(attn_scores_flat)
}
