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

Byte Pair Encoding (BPE) is the underlying tokenization scheme used by GPT2 and GPT-3. It's not all that different than our simple tokenization scheme, just more powerful. The actual vocabulary that GPT2/3 use is about 50,257 tokens long.

If it encounters an unknown word, it breaks it down into sub-words or even letters until we can match it against our vocabulary. For example, the unknown word: "Akwirw ier" will get encoded as: [33901, 86, 343, 86, 220, 959]. And then when we decode it, we get "Akwirw ier" back again. This is nice because then it doesn't have to replace unknown words with `<|unk|>`.

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

Moving forward we'll use the BPE tokenizer and will ignore our simple tokenizer. Even though we won't be using it, it was fun to build!

## Data Sampling with a Sliding Window

LLMs predict the next token in a sentence. In order to train our LLM, we need to generate input-target pairs. We can do this by assigning the input to an X variable and then the output to a Y variable which is just one word past the X.

So if the sentence is: "Let's train an LLM together".

let x = "train" (will be tokenized)
let y = "an" (one word past the x variable)

This is how we train the LLM. We give it the input, then we ask it to predict the next token, then we can check if it predicted the right one based on the y variable.

Here's how we set it up:

```rust

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
```

This prints out:

```
x:    [290, 4920, 2241, 287]
y:         [4920, 2241, 287, 257]
```

If we convert the tokenIDs to words, we get:

```rust
for i in 1..context_size + 1 {
        let context = &enc_sample[..i];
        let desired = enc_sample[i];
        println!(
            "{:?} ----> {:?}",
            bpe.decode(context.to_vec()).unwrap(),
            bpe.decode(vec![desired]).unwrap()
        )
    }
```

```
" and" ----> " established"
" and established" ----> " himself"
" and established himself" ----> " in"
" and established himself in" ----> " a"
```

You can see the sliding window here in the words much better. Now let's do this for the entire data set. We'll create an input tensor and a target tensor.

### Scalars, Vectors and Tensors

A quick reminder on scalars, vectors and tensors.

Scalar -> a single value i.e. an integer or a float like 2, 0.45, 10.32 etc.
Vector -> an array (or vector in rust land) of values i.e. [1,2,3], [0.34, 3.45,23.1]
Tensor -> a 2-dimensional vector of values [[1,2,3],[4,5,6]], [[0.4,3.5],[9.5,239.1]]

### Loading Data and Creating Tensors

I was going back and forth on what I wanted to use here. One option is tch-rs which is as close to pytorch as get in rust. It has rust bindings for the C++ API of pytorch. The other option is something like candle which is a hugging face library that implements low-level tensor functions and operations. Another option is burn which is a pure rust implementation but it's still pretty young. I decided to go with candle since tch-rs requires py-torch, but we'll see I may have to do tch-rs anyways depending on what comes later.

`cargo add candle-core`

First things, we need to load our data from our file and be able to create these input-target pairs as tensors efficiently. Let's implement a `GPTDataset` struct and methods that allow us to do this.

Our struct:

```rust
use candle_core::{Device, Tensor}; // Device comes a little later

struct GPTDataset {
    input_ids: Vec<Tensor>,
    target_ids: Vec<Tensor>,
}
```

And now our implementation of the `GPTDataset` struct:

```rust
impl GPTDataset {
    pub fn new(
        self,
        txt: &str,
        tokenizer: &CoreBPE,
        max_length: usize,
        stride: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut input_ids = Vec::new();
        let mut target_ids = Vec::new();

        let token_ids: Vec<u32> = tokenizer.encode_with_special_tokens(txt);

        let mut i = 0;

        while i + max_length < token_ids.len() {
            // Extract input chunk
            let input_chunk = &token_ids[i..i + max_length];
            // Extract target chunk (shifted by 1)
            let target_chunk = &token_ids[i + 1..i + max_length + 1];

            // Convert to tensors
            let device = Device::Cpu;
            let input_tensor = Tensor::new(input_chunk, &device)?;
            let target_tensor = Tensor::new(target_chunk, &device)?;

            input_ids.push(input_tensor);
            target_ids.push(target_tensor);

            i += stride;
        }

        Ok(GPTDataset {
            input_ids,
            target_ids,
        })
    }

    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    // gets a single row in the data set
    pub fn get(&self, idx: usize) -> Option<(&Tensor, &Tensor)> {
        if idx < self.len() {
            Some((&self.input_ids[idx], &self.target_ids[idx]))
        } else {
            None
        }
    }
}
```

The `new` function does a lot of the heavy lifting. The sliding window bit is in the `input_chunk` and `target_chunk` variables. In the `input_chunk` we get the id from `i` to `i+max_length`. Then in the `target_chunk` we get the target by incrementing it to the chunk with the `+1` in `i+1..i + max_length +1`. Then we create the tensors using the Candle library and setting the device to be CPU . Then some getter methods.

Let's keep going and add in our `DataLaoder` struct and methods.

```rust
pub struct DataLoader {
    dataset: GPTDataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}
```

We set some inputs into the `DataLoader` struct that will tune our data loader. Next our implementation:

```rust
impl DataLoader {

    // instantiate a new dataloader instance
    pub fn new(dataset: GPTDataset, batch_size: usize, shuffle: bool, drop_last: bool) -> Self {
        DataLoader {
            dataset,
            batch_size,
            shuffle,
            drop_last,
        }
    }

    // create an ityerator on the dataloader

    pub fn iter(&self) -> DataLoaderIterator {
        let mut indices: Vec<usize> = (0..self.dataset.len()).collect();

        if self.shuffle {
            let mut rng = rng();
            indices.shuffle(&mut rng);
        }

        DataLoaderIterator {
            dataset: &self.dataset,
            indices,
            batch_size: self.batch_size,
            current: 0,
            drop_last: self.drop_last,
        }
    }
}
```

The `DataLoader` struct defines the params that we want to use when we load the data. Note that we're not doing any async or parallel loading. Trying to keep it relatively simply. The `iter()` is a wrapper for a `DataLoaderIterator` implementation which we'll see next. And then lastly our `DataIterator` struct and implementation:

```rust
pub struct DataLoaderIterator<'a> {
    dataset: &'a GPTDataset,
    indices: Vec<usize>,
    batch_size: usize,
    current: usize,
    drop_last: bool,
}

impl<'a> Iterator for DataLoaderIterator<'a> {
    type Item = Result<(Tensor, Tensor), Box<dyn std::error::Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        // out of bounds, return
        if self.current >= self.indices.len() {
            return None;
        }

        let batch_end = (self.current + self.batch_size).min(self.indices.len());

        // Check if we should drop the last batch
        if self.drop_last && batch_end - self.current < self.batch_size {
            return None;
        }

        // Collect batch tensors
        let mut input_batch = Vec::new();
        let mut target_batch = Vec::new();

        for i in self.current..batch_end {
            let idx = self.indices[i];
            if let Some((input, target)) = self.dataset.get(idx) {
                input_batch.push(input.clone());
                target_batch.push(target.clone());
            }
        }

        self.current = batch_end;

        // Stack tensors into batch
        match (
            Tensor::stack(&input_batch, 0),
            Tensor::stack(&target_batch, 0),
        ) {
            (Ok(inputs), Ok(targets)) => Some(Ok((inputs, targets))),
            (Err(e), _) | (_, Err(e)) => Some(Err(Box::new(e))),
        }
    }
}

pub fn create_dataloader_v1(
    txt: &str,
    batch_size: usize,
    max_length: usize,
    stride: usize,
    shuffle: bool,
    drop_last: bool,
) -> Result<DataLoader, Box<dyn std::error::Error>> {
    let tokenizer = r50k_base()?;

    let dataset = GPTDataset::new(txt, &tokenizer, max_length, stride)?;

    Ok(DataLoader::new(dataset, batch_size, shuffle, drop_last))
}
```

This `DataLoaderIterator` struct and implementation defines how the iterator works and moves across the batches and stacks the tensors. We also see those pesky lifetimes again since we're using a reference to the `GPTDataset`.

Let's give it a shot and see what it outputs. We can try using this in our main func:

```rust
  let file_name = "the-verdict.txt";

    let raw_text = load_file(file_name);

    let dataloader = create_dataloader_v1(&raw_text, 1, 4, 1, false, true).unwrap();

    for (batch_idx, batch_result) in dataloader.iter().take(2).enumerate() {
        let (inputs, targets) = batch_result.unwrap();

        let input_vec: Vec<Vec<u32>> = inputs.to_vec2().unwrap();
        let target_vec: Vec<Vec<u32>> = targets.to_vec2().unwrap();

        println!("Batch {}:", batch_idx);
        println!("  Input tokens: {:?}", input_vec);
        println!("  Target tokens: {:?}", target_vec);
        println!();
    }

```

This outputs:

```
Batch 0:
  Input tokens: [[40, 367, 2885, 1464]]
  Target tokens: [[367, 2885, 1464, 1807]]

Batch 1:
  Input tokens: [[367, 2885, 1464, 1807]]
  Target tokens: [[2885, 1464, 1807, 3619]]
```

Nice! One way to double check this is that the target tokens are 1 token ahead of the input tokens. This is set by the `stride` hyperparameter. It dictates the number of positions the inputs shift across batches, creating a sliding window approach. By setting the stride equal to the input window size (or the max_length) we prevent any overlaps in the input-target text.

## Creating Token Embeddings

We create token embeddings from our tokenIDs. These embeddings are ultimately what gets fed into the LLM. It's important that we also encode the position of the tokenID in the embedding to help the LLM learn how the tokends relate to the each other. This is analogous to ordering the rows in the tensor in order of the words as they appear in the sentence.

There are two types of positional encoding. The first is absolute which means that we encode the tokens position based on the order they appear in the sentence. The second is relative positional encoding which means we encode the tokens based on their distance in embedding space from each other.

Let's run the same code as before (I also just factored out the params for the dataloader into their own variables to make changing them easier) but let's also print out the shape of the tensor:

```rust
   let file_name = "the-verdict.txt";

    let raw_text = load_file(file_name);

    let max_length = 4;
    let batch_size = 8;
    let stride = 4;
    let shuffle = false;
    let drop_last = true;

    let dataloader = create_dataloader_v1(
        &raw_text, batch_size, max_length, stride, shuffle, drop_last,
    )
    .unwrap();

    for (batch_idx, batch_result) in dataloader.iter().take(2).enumerate() {
        let (inputs, targets) = batch_result.unwrap();

        let input_vec: Vec<Vec<u32>> = inputs.to_vec2().unwrap();
        let target_vec: Vec<Vec<u32>> = targets.to_vec2().unwrap();

        println!("Batch {}:", batch_idx);
        println!("  Input tokens: {:?}", input_vec);
        println!("  Target tokens: {:?}", target_vec);
        println!("inputs shape: {:?}", inputs.shape());
    }
```

We get:

```
Batch 0:
  Input tokens: [[40, 367, 2885, 1464], [1807, 3619, 402, 271], [10899, 2138, 257, 7026], [15632, 438, 2016, 257], [922, 5891, 1576, 438], [568, 340, 373, 645], [1049, 5975, 284, 502], [284, 3285, 326, 11]]
  Target tokens: [[367, 2885, 1464, 1807], [3619, 402, 271, 10899], [2138, 257, 7026, 15632], [438, 2016, 257, 922], [5891, 1576, 438, 568], [340, 373, 645, 1049], [5975, 284, 502, 284], [3285, 326, 11, 287]]
inputs shape: [8, 4]
Batch 1:
  Input tokens: [[287, 262, 6001, 286], [465, 13476, 11, 339], [550, 5710, 465, 12036], [11, 6405, 257, 5527], [27075, 11, 290, 4920], [2241, 287, 257, 4489], [64, 319, 262, 34686], [41976, 13, 357, 10915]]
  Target tokens: [[262, 6001, 286, 465], [13476, 11, 339, 550], [5710, 465, 12036, 11], [6405, 257, 5527, 27075], [11, 290, 4920, 2241], [287, 257, 4489, 64], [319, 262, 34686, 41976], [13, 357, 10915, 314]]
inputs shape: [8, 4]
```

The tokenIDs are the same as before (good!) and the input shape is the new thing we just printed. It's telling us that there 8 samples in each batch and each sample has 4 tokens each. It's a vector with 8 vectors as elements and each vector element has 4 elements in it.

### Creating an Embedding layer

Now to start creating some weights. Here is where our `candle_core` decision over `tch-rs` is starting to haunt us a little bit (or make it more fun!). `candle_core` is fairly low level and it doesn't have a nice way to create an embedding layer. So we're going to create ourselves.

First, let's think about what we need. We need a way to matrix multiply tensors in order to get a weight matrix that we can then use to loop up weights later on.

So first, let's create a struct.

```rust
use candle_core::{Device, Result, Tensor};

pub struct Embedding {
    pub weights: Tensor,
}
```

A field called 'weights' is going to hold our weights which will be `Tensor` types from the `candle_core` library. Now onto the implementation:

```rust
impl Embedding {
    // creates a weight matrix with mean 0, std of 1 and shape[vocab_size*output_dim
    pub fn new(vocab_size: i64, output_dim: i64, device: Device) -> Result<Embedding> {
        let weights = Tensor::randn(
            0.0f32,
            1.0f32,
            &[vocab_size as usize, output_dim as usize],
            &device,
        )?;
        Ok(Embedding { weights })
    }
}
```

First, we'll create a way to instantiate a new embedding layer with a `new` method. This will take in our vocab size (which should usually be 50,267 like GPT-2/3), an `output_dim` argument which will dictate how many dimensions we want to output and lastly a `device` param which will tell candle where to run the computation i.e. CPU vs GPU.

Then we'll randomly initialize the weights using the built in `Tensor::randn` method which takes in a mean, standard deviation, shape and then a device. Since we're just randomly initializing the weights, we can set the mean to 0 and standard deviation to 1. The shape is going to be our [vocab size * output_dim] and device is .. device.

Then we'll wrap the struct in an Ok and return a result with the Embedding struct and weights. Next, let's add in a forward pass.

```rust
impl Embedding {
    // creates a weight matrix with mean 0, std of 1 and shape[vocab_size*output_dim
    pub fn new(vocab_size: i64, output_dim: i64, device: Device) -> Result<Embedding> {
        let weights = Tensor::randn(
            0.0f32,
            1.0f32,
            &[vocab_size as usize, output_dim as usize],
            &device,
        )?;
        Ok(Embedding { weights })
    }

    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let shape = token_ids.dims();

        // handle both 1d and 2d vectors
        let (batch_size, seq_len, is_1d) = match shape.len() {
            1 => (1, shape[0], true),
            2 => (shape[0], shape[1], false),
            _ => {
                return Err(candle_core::Error::Msg(
                    "Input must be 1D or 2D tensor".into(),
                ));
            }
        };

        // flatten the token_ids into a 1-dim vector
        let flat_ids = if is_1d {
            token_ids.clone()
        } else {
            token_ids.flatten_all()?
        };

        // index_select on dim=0
        let gathered = self.weights.index_select(&flat_ids, 0)?;

        // reshape to [batch_size, seq_len, output_dim]
        let output_dim = self.weights.dims()[1];
        let reshaped = gathered.reshape(&[batch_size, seq_len, output_dim])?;

        if is_1d {
            reshaped.squeeze(0) // if 1d the reshape so it's 0 dim
        } else {
            Ok(reshaped)
        }
    }
}
```

Let's go over the forward pass i.e. `forward` method. The forward pass is a look up operation on the weight matrix based on the `token_ids` that the user passes in.

It takes in a `&self` argument and then a `Tensor` pointer to `token_ids` and returns a `Tensor` wrapped in a `Result`. First, we get the shape from the `token_ids` (which we will need later) and then we run a match to set the right batch_size and seq_len, and also set if its a 1 dimensional tensor.

Then we flatten the `token_id` tensor if it's not a 1 dimensional tensor. Flattening the tensor allows us to then run an index select or look up on the weights by the `token_ids` that the user passed in.

Once we've done the look up operation, we reshape the output tensor based on the shape and then return it.

### Back to our positional encodings

Now that we've got our embedding layers all sorted, let's put in our positional encodings. We mentioned earlier that there are two types of encoding: absolute and relative. We're going to go with absolute this time.

There are two steps here: 1. create another embedding layer with the same dimensionality as our token embeddings to encode the position of the `token_ids` and then add the resulting tensor to the token embedding layer tensor.

Let's update our main driver function:

```rust
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
    let output_dim: i64 = 256;
    let device = Device::Cpu;

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
    Ok(())
}
```

Now we're creating a second embedding layer and then add the two layers together. Since they have different shapes with the `token_Embeddings` as [8,4,256] and the `pos_embeddings` as [4,256], we can do `broadcast_add()` which will handle resolving the shapes for us and add them together nicely by doing element-wise addition starting with the right-most element.

We can verify it worked correctly by checking that the resulting shape is [8,4,256].

# Coding Attention Mechanisms

Self attention is a mechanism that allows each position in the input sequence to consider the relevancy of, or "attend to" all other positions in the same sequence when computing the representation of a sequence.

The goal of self-attention is to compute a context vector for each input element that combines information from all other input elements. The importance or contribution of each input element for computing the context vector is determined by the attention weights.

Essentially, for each token, you're computing weights from every other input element to determine how relevant other tokens are to this token.

In order to do this,let's work through an example.

Assume we've embedding the sentence "Your journey starts with one step" into a 3 dimensions and the resulting tensor is:

```rust
    let data = vec![
        0.43, 0.15, 0.89, // Your (x^1)
        0.55, 0.87, 0.66, // journey (x^2)
        0.57, 0.85, 0.64, // starts (x^3)
        0.22, 0.58, 0.33, // with (x^4)
        0.77, 0.25, 0.10, // one (x^5)
        0.05, 0.80, 0.55, // step (x^6)
    ];
```

Now let's create a tensor from that with the shape [6,3] meaning 6 rows and 3 columns in each row (3 values in each vector):

`let inputs: Tensor = Tensor::from_vec(data, (6, 3), &Device::Cpu)?;`

Which will print to:

```rust
Inputs tensor:
[[0.4300, 0.1500, 0.8900],
 [0.5500, 0.8700, 0.6600],
 [0.5700, 0.8500, 0.6400],
 [0.2200, 0.5800, 0.3300],
 [0.7700, 0.2500, 0.1000],
 [0.0500, 0.8000, 0.5500]]
```

Nice. Next let's compute the intermediate attention scores between a query token (in this case the word journey at index 1) and each input token(every token in the input sentence). This will give us the attention or relevancy of the different tokens in the sentence with respect to the token at index 1.

The code is quite simple but powerful:

```rust
    // get our query token by using the get method to get the token at index 1
    let query = inputs.get(1);
    // unwrap the query to get the tensor
    let query_tensor = query?;

    //calc the dot product of the query tensor with each element of the input tensor
    let atten_scores = inputs.matmul(&query.unsqueeze(1)?)?;
    // flatten the result into a one dimensional vector
    let attn_scores_flat = atten_scores.flatten_all()?;


    println!("print the attention scores: {:?}", attn_scores_flat);

    Ok(())
```

We're taking the dot product (matmul or element-wise matrix multiplication and sum) of the query tensor with the input tensor (really it's the vector within the tensor). When we run that we get:

```
print the attention scores: Tensor[0.9544, 1.495, 1.4754, 0.8433999999999999, 0.7070000000000001, 1.0865; f64]
```

This is our attention score for the query at input[1].

The dot product is a measure of how similar two vectors are. A higher dot product means there is higher similarity and attention score between two elements.

### Normalizing attention scores

Next we want to normalize our attention scores so that they sum to 1. This helps us get an idea of the weight of each attention score.

To do this we just want to divide each element in our `attn_scores_flat` by the sum of the elements in that tensor so then we can sum up the entire tensor to 1. Like this:

```rust
    let query = inputs.get(1);

    let query_tensor = query?;


    let atten_scores = inputs.matmul(&query.unsqueeze(1)?)?;

    let attn_scores_flat = atten_scores.flatten_all()?;

    let atten_sum = attn_scores_flat.sum_all()?;

    let atten_weights_2_temp = atten_scores.broadcast_div(&atten_sum)?;

    let atten_unwrapped = atten_weights_2_temp;

    println!("Attention weights: {:?}", atten_unwrapped);
    println!("sum: {:?}", atten_unwrapped.sum_all());
```

And when we print that we get:

```
Attention weights: Tensor[0.145450112013655, 0.22783729826112137, 0.22485026746117623, 0.1285337641160065, 0.10774646814087814, 0.16558209000716279; f64]
sum: Ok(Tensor[1; f64])
```

And our check at the end shows that it sums up to 1, nice!

In reality, we would use something like a `softmax` to also handle negative numbers nicely. We can write a quick implementation of it:

```rust
fn softmax(input: &Tensor) -> Result<Tensor, Error> {
    // Softmax formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)

    // take the exponnent of each value
    let exp_values = input.exp()?;

    // sum all exp values
    let exp_sum = exp_values.sum_all()?;
    println!("  Sum of exp values: {}", exp_sum);

    // use broadcast to handle shape differences and divide each exp value by the sum to normalize
    let softmax_result = exp_values.broadcast_div(&exp_sum)?;

    Ok(softmax_result)
}
```

This is just my own implementation that I did quickly and definitely has not been extensively tested for overflows and underflows.

Okay, let's move onto the last part here which is creating the context vector. We do this by multiplying the embedded input tokens (x_i) by the corresponding attention weights and then summing the resulting vectors. So the context vector is a weighted sum of all input vectors obtained by multiplying each input vector by it's attention weight.

Let's create a function to do this:

```rust
fn compute_context_vector(inputs: &Tensor, atten_weights: &Tensor) -> Result<Tensor, Error> {
    //  sum over i of (atten_weights[i] * inputs[i])

    // Reshape attention weights to [1, 6] for matrix multiplication
    let weights_reshaped = atten_weights.unsqueeze(0)?; // [6] -> [1, 6]

    // Matrix multiplication: [1, 6] × [6, 3] = [1, 3]
    let context_matrix = weights_reshaped.matmul(inputs)?;

    // Flatten to get [3] vector
    let context_vector = context_matrix.flatten_all()?;
    Ok(context_vector)
}

let attn_weight_2 = softmax(&output)?;

let context_vector = compute_context_vector(&inputs, &attn_weight_2);

```

In this function, we're unsqueezing the normalized attention weights to get then into the vector shape we want with 6 dimensions. Then we run matmul on the reshaped normalized weights by taking the dot product with the input vector. Then we flatten the resulting tensor into a 1 dimension vector and return it.

When we print it we get:

```
Ok(Tensor[0.4418657478512921, 0.651481978030222, 0.5683088877257294; f64])
```

Sweet! Things are looking up. Now we know how to calculate attention weights and context vectors for an individual token. Now we can expand this to do it for every token in our input.

### Computing attention weights for all inputs

Before we computed the attention weight for just input[2], now we can do it for all of the inputs.

We'll want to iterate over each vector in the tensor and then over each element in the vector and calculate the context vector for each one.

Essentially computing the attention score for every token in relation to every other token.

Here is our function:

```rust

    let inputs = inputs.to_dtype(candle_core::DType::F32)?;
    let attn_scores = NeuralNet::compute_attention_scores_matrix(&inputs)?;

    pub fn compute_attention_scores_matrix(inputs: &Tensor) -> Result<Tensor, Error> {
        let seq_len = inputs.shape().dims()[0];
        let mut scores_vec: Vec<f32> = Vec::new();

        for i in 0..seq_len {
            let x_i = inputs.get(i)?;
            for j in 0..seq_len {

                let x_j = inputs.get(j)?;

                let dot_product = (&x_i * &x_j)?.sum_all()?.to_scalar::<f32>()?;

                scores_vec.push(dot_product);
            }
        }

        let attn_scores = Tensor::from_vec(scores_vec, (seq_len, seq_len), inputs.device())?;
        Ok(attn_scores)
    }
```

First we convert the inputs from a float64 to float32 in order to do our matrix multiplication. THen we call our `compute_attention_scores_matrix` function.

Then we initialize an empty vector to hold our attention weights (dot products). Then we iterate over every vector in the tensor and over every element in the vector and compute the dot product of that element with every other element.

Since the shape of our tensor is [6,6] we'll have 36 total weights.

We could have used our previous `compute_attention_scores` function here instead of calculating the raw dot product but I thought it was nice to shot it.

Once we have the dot product, we push that into our vector and keep moving forward.

Lastly, we convert our vector back to a Tensor. We could be more efficient in this function but honestly I think this is much clearer.

Once we do all of that and print our attention matrix, we get:

```
Attention scores matrix:
Row 0: [0.9994999, 0.95440006, 0.9422, 0.4753, 0.4576, 0.63100004]
Row 1: [0.95440006, 1.4950001, 1.4754001, 0.8434, 0.707, 1.0865]
Row 2: [0.9422, 1.4754001, 1.457, 0.8296, 0.7154, 1.0605]
Row 3: [0.4753, 0.8434, 0.8296, 0.49369997, 0.34739998, 0.6565]
Row 4: [0.4576, 0.707, 0.7154, 0.34739998, 0.66539997, 0.2935]
Row 5: [0.63100004, 1.0865, 1.0605, 0.6565, 0.2935, 0.94500005]
```
