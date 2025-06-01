use std::fs;

pub fn load_file(file: &str) -> String {
    let raw_text = fs::read_to_string(file).expect("Failed to read file");

    raw_text
}
