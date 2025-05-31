# Introduction

This blog contains thoughts and notes as I implement GPT-2 (originally implemented in Python) in Rust. Throughout the blog I try to explain concepts in simple english and assume the reader has no AI/ML experience. Hopefully someone who is just starting out can pick this up and follow along.

# Part 1 - Data processing and sampling pipeline

Large Language Models (LLMs) and other Deep Neural Networks (DNNs) cannot process raw text directly. We must first encode or embed the text into continuous vector values.

**Continuous ... values** is a fancy word for values that are measured such as temperature. You can always get more precise by measuring more of the decimals.

A **vector** is just a 1-dimensional array of n continuous values i.e. [0.142, 3.453, 2.2390].

We create these embeddings using an embedding model which converts objects like a word, image or even video into continuous vector values.

The number of continuous values in a vector comes from the embedding model and the object we're embedding.

We need to write some code that translates a sentence like "Hey, my name is Evis" to a vector representation such as [0.2323532, 123.23242, 0.2342, 2.239976].

## Loading the file

For this exercise, we're using [The Verdict](https://en.wikisource.org/wiki/The_Verdict) to create our embeddings.

First we want to load the book in memory so we can process it. In rust, we can do that using: `fs::read_to_string(file_name)`. This is easy enough.

Next, we want to start to parse a sentence to get it ready for tokenization.

Let's create a simple regex tokenizer in rust. Here's what we have:

```rust
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

    tokens
}
```

Let's go through this line by line.

1. We create a regex using the Regex crate in rust to match on punctuation marks, double dashes and whitespace.
2. Then we create an empty vector to store the final tokens and a variable to track where the last match ended.
3. Iterate through every match in our regex pattern and check if there's text between the last match and the current match.
4. If it's not empty after trimming whitespace, we add it to our tokens vector.
5. Then we handle the delimiter itself similar to text by checking if it's not empty after trimming whitespace and adding it to the tokens vector.
6. Then update position to the end of the current matcha and start over from step 4.
7. Return the tokens vector
