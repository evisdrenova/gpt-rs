use file_loader::{load_file, simple_tokenizer};

mod file_loader;

fn main() {
    let file_name = "the-verdict.txt";

    let text = load_file(file_name);

    let _parsed_text = simple_tokenizer(&text);
}
