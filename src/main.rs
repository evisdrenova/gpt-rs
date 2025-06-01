use tokenizer::{SimpleTokenizer, load_file, simple_tokenizer};

mod tokenizer;

fn main() {
    let file_name = "the-verdict.txt";

    let text = load_file(file_name);

    let vocab = simple_tokenizer(&text);

    println!("{:?}", vocab.len());

    let tokenizer: SimpleTokenizer<'_> = SimpleTokenizer::new(vocab);

    let ids = tokenizer.encode("Hello, do you like tea?");
    println!("Encoded: {:?}", ids);

    let decoded = tokenizer.decode(&ids);
    println!("Decoded: {}", decoded);
}
