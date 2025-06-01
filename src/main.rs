use file_operations::load_file;

mod file_operations;
mod simple_tokenizer;

use tiktoken_rs::r50k_base;

fn main() {
    let file_name = "the-verdict.txt";

    let raw_text = load_file(file_name);

    // this is gpt-2 tokenizer
    let bpe = r50k_base().unwrap();

    let enc_text = bpe.encode_with_special_tokens(&raw_text);

    println!("Tokens: {:?}", enc_text.len());

    let enc_sample = &enc_text[50..];

    let context_size = 4;

    let x = &enc_sample[..context_size];
    let y = &enc_sample[1..context_size + 1];

    println!("x:    {:?}", x);
    println!("y:        {:?}", y);

    for i in 1..context_size + 1 {
        let context = &enc_sample[..i];
        let desired = enc_sample[i];
        println!(
            "{:?} ----> {:?}",
            bpe.decode(context.to_vec()).unwrap(),
            bpe.decode(vec![desired]).unwrap()
        )
    }
}
