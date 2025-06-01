# Introduction

This blog contains thoughts and notes as I implement GPT-2 (originally implemented in Python) in Rust. Throughout the blog I try to explain concepts in simple english and assume the reader has no AI/ML experience. Hopefully someone who is just starting out can pick this up and follow along.

# Part 1 - Data processing and sampling pipeline

Large Language Models (LLMs) and other Deep Neural Networks (DNNs) cannot process raw text directly. We must first encode or embed the text into continuous vector values.

**Continuous ... values** is a fancy word for values that are measured such as temperature. You can always get more precise by measuring more of the decimals.

A **vector** is just a 1-dimensional array of n continuous values i.e. [0.142, 3.453, 2.2390].

We create these embeddings using an embedding model which converts objects like a word, image or even video into continuous vector values.

The number of continuous values in a vector comes from the embedding model and the object we're embedding.

We need to write some code that translates a sentence like "Hey, my name is Evis" to a vector representation such as [0.2323532, 123.23242, 0.2342, 2.239976].

## Loading and parsing text

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

## Creating token IDs

Once we've tokenized the data, we want to create tokenIds from the unique set of tokenized words and characters. To do this, we just sort the tokenized data alphabetically and then assign integers to each token starting from a 0 index.

So let's add another variable called `unique_tokens` which is a `HashSet`. Since we're already iterating over the tokens, we can just insert the tokens to the `HashSet`.

## Creating a vocabulary

Next we want to create a vocabulary which is just a a vector of tuples with (tokenId, token) in each tuple. This effectively creates a map that we will use later to translate new texts into our existing vocabulary (adding where we haven't seen the word or character before).

To do this, we'll create a new function called `create_vocab` which takes in tokens and then returns a vector of tuples such as `Vec<(&str, i32)>`. Here's the code:

```rust
fn create_vocab(tokens: Vec<&str>) -> IndexMap<&str, usize> {
    tokens
        .into_iter()
        .enumerate()
        .map(|(index, token)| (token, index))
        .collect()
}
```

This a nice pure function that takes in a vector of (sorted) tokens and then iterates over each token and creates a tuple from the index and the token and then collects it into an IndexMap and returns it.

`IndexMap` is a way of created a map that maintains the order of the elements that we insert. Since we're alphabetically sorting the map, we can't use a good old regular `HashMap`.

## Creating a "class"

While rust doesn't really support classes in the same way that python does, we can still create a nice `SimpleTokenizer` struct and add `encode` and `decode` methods on it like we would in python.

First, let's create a struct to hold two fields. The first is a `str_to_int` field which is our encoder. This converts strings to integers (tokenIds). The second is our decoder. This converts integers (tokenIds) to strings. We need this to convert the output of the LLM back to strings so we can show the user.

```rust
pub struct SimpleTokenizer<'a> {
    str_to_int: IndexMap<&'a str, usize>,
    int_to_str: IndexMap<usize, &'a str>,
}
```

Since we're using string pointers, we also need to define lifetimes, the `'a` for references so the Rust compiler knows how long they need to live for. In this case, the string references need to live as long as the struct is around.

Next, let's create a function that instantiates a new SimpleTokenizer.

```rust
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
```

This function seeds our `SimpleTokenizer` with the vocab that we created earlier and then returns the struct.

Now we can create our encode and decode functions. Luckily, we've pretty much already built our encode function from before so let's just create a method on our `SimpleTokenizer` struct:

```rust

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
                self.str_to_int
                    .get(s)
                    .copied()
                    .unwrap_or_else(|| panic!("Token '{}' not found in vocabulary", s))
            })
            .collect()
    }
```

And then the decoder is just the opposite process of going from an id to a string.

```rust
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
```

Sweet! Things are coming along. Right now, when we encounter a word that isn't in our vocab, the code panics. So let's start dealing with that. First, let's add in `<|unk|>` and `<|endoftext|>` to handle end of text files when we concat a bunch of pre-training data. And for words we don't have in our vocab, we'll return the `<|unk|>`.

Let's update our `encode` function to to replace unknown words with `<|unk|>`:

```rust
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
```

## Byte Pair Encoding

Byte Pair Encoding (BPE) is the underlying tokenization scheme used by GPT2 and GPT-3. It's a little different than our tokenization scheme but honestly not all that different. It mainly handles unknown words in a much nicer way. If it encounters an unknown word, it breaks it down into sub-words or even letters until we can match it against our vocabulary.

The actual vocabulary that GPT2/3 use is about 50,257 tokens long. For example, the unknown word: "Akwirw ier" will get encoded as: [33901, 86, 343, 86, 220, 959]. And then when we decode it, we get "Akwirw ier" back again. This is nice because then it doesn't have to replace unknown words with `<|unk|>`.

The [BPE](https://github.com/openai/tiktoken) that OpenAI used is actually written in Rust but interfaces through Python. Luckily someone had already written a [Rust port](https://github.com/zurawiki/tiktoken-rs?tab=readme-ov-file), so to save ourselves some time, I just picked that one up.

This is how we can use it:

```rust

    use tiktoken_rs::r50k_base;

    // this is gpt-2 model
    let bpe = r50k_base().unwrap();

    let new_text = "Akwirw ier";

    let tokens = bpe.encode_with_special_tokens(new_text);

    println!("encode: {:?}", tokens);

    let detokenize = bpe.decode(tokens);
    println!("decode : {:?}", detokenize);

```
