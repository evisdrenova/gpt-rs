use indexmap::IndexMap;
use regex::Regex;
use std::collections::HashSet;

pub struct SimpleTokenizer<'a> {
    str_to_int: IndexMap<&'a str, usize>,
    int_to_str: IndexMap<usize, &'a str>,
}

impl<'a> SimpleTokenizer<'a> {
    pub fn new(vocab: IndexMap<&'a str, usize>) -> Self {
        let int_to_str: IndexMap<usize, &'a str> = vocab
            .iter()
            .map(|(&token, &index)| (index, token))
            .collect();

        SimpleTokenizer {
            str_to_int: vocab,
            int_to_str,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let re = Regex::new(r#"([,.:;?_!"()']|--|\s)"#).unwrap();

        let mut tokens = Vec::new();
        let mut last_end = 0;

        for mat in re.find_iter(text) {
            // Add token before delimiter
            if mat.start() > last_end {
                let token = text[last_end..mat.start()].trim();
                if !token.is_empty() {
                    tokens.push(token);
                }
            }
            // add delimiter
            let delimiter = mat.as_str();
            if !delimiter.trim().is_empty() {
                tokens.push(delimiter);
            }

            last_end = mat.end();
        }

        if last_end < text.len() {
            let token = text[last_end..].trim();
            if !token.is_empty() {
                tokens.push(token);
            }
        }

        // Convert tokens to IDs and panic if a word is not the in vocab
        tokens
            .iter()
            .map(|&s| {
                self.str_to_int.get(s).copied().unwrap_or_else(|| {
                    self.str_to_int
                        .get("<|unk|>")
                        .copied()
                        .expect("<|unk|> token not found in vocabulary")
                })
            })
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let text: Vec<&str> = ids
            .iter()
            .filter_map(|&id| self.int_to_str.get(&id).copied())
            .collect();

        let mut result = text.join(" ");

        let re = Regex::new(r#"\s+([,.?!\"()'])"#).unwrap();

        // Fix spacing around punctuation
        result = re.replace_all(&result, "$1").to_string();

        result
    }
}

pub fn create_vocab(raw_text: &str) -> IndexMap<&str, usize> {
    let re = Regex::new(r#"([,.:;?_!"()']|--|\s)"#).unwrap();

    let mut unique_tokens: HashSet<&str> = HashSet::new();
    let mut last_end = 0;

    for mat in re.find_iter(raw_text) {
        if mat.start() > last_end {
            let token = raw_text[last_end..mat.start()].trim();
            if !token.is_empty() {
                unique_tokens.insert(token);
            }
        }

        let delimiter = mat.as_str();
        if !delimiter.trim().is_empty() {
            unique_tokens.insert(delimiter);
        }

        last_end = mat.end();
    }

    if last_end < raw_text.len() {
        let token = raw_text[last_end..].trim();
        if !token.is_empty() {
            unique_tokens.insert(token);
        }
    }

    let mut sorted_tokens: Vec<_> = unique_tokens.into_iter().collect();
    sorted_tokens.sort_unstable();

    sorted_tokens.push("<|endoftext|>");
    sorted_tokens.push("<|unk|>");

    let vocab_tokens = sorted_tokens
        .into_iter()
        .enumerate()
        .map(|(index, token)| (token, index))
        .collect();

    vocab_tokens
}
