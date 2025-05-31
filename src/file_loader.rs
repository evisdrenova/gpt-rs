use regex::Regex;
use std::fs;

pub fn load_file(file: &str) -> String {
    let raw_text = fs::read_to_string(file).expect("Failed to read file");

    println!("Total number of characters: {:?}", raw_text.len());

    println!("{}", &raw_text[..99.min(raw_text.len())]);

    raw_text
}

pub fn simple_tokenizer(raw_text: &str) {
    let re = Regex::new(r#"([,.:;?_!"()']|--|\s)"#).unwrap();

    let mut tokens = Vec::new();
    let mut last_end = 0;

    for mat in re.find_iter(raw_text) {
        if mat.start() > last_end {
            let token = raw_text[last_end..mat.start()].trim();
            if !token.is_empty() {
                tokens.push(token);
            }
        }

        let delimiter = mat.as_str();
        if !delimiter.trim().is_empty() {
            tokens.push(delimiter);
        }

        last_end = mat.end();
    }

    if last_end < raw_text.len() {
        let token = raw_text[last_end..].trim();
        if !token.is_empty() {
            tokens.push(token);
        }
    }
}
