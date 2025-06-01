use file_operations::load_file;
use tokenizer::{SimpleTokenizer, create_vocab};

mod file_operations;
mod tokenizer;

use tiktoken_rs::r50k_base;

fn main() {
    let file_name = "the-verdict.txt";

    let text = load_file(file_name);

    let vocab = create_vocab(&text);

    println!("{:?}", vocab.len());
    let text1 = "Hello, do you like tea?";
    let text2 = "In the sunlit terraces of the palace.";
    let text = text1.to_string() + " <|endoftext|> " + text2;
    println!("{:?}", text);

    let tokenizer: SimpleTokenizer<'_> = SimpleTokenizer::new(vocab);

    let ids = tokenizer.encode(&text);
    println!("Encoded: {:?}", ids);

    let decoded = tokenizer.decode(&ids);
    println!("Decoded: {}", decoded);

    // this is gpt-2 tokenizer
    let bpe = r50k_base().unwrap();

    let new_text = "Akwirw ier";

    let tokens = bpe.encode_with_special_tokens(new_text);

    println!("Tokens: {:?}", tokens);

    let detokenize = bpe.decode(tokens);
    println!("detokenize : {:?}", detokenize);
}
