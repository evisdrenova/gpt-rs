use candle_core::{Device, Tensor};
use rand::rng;
use rand::seq::SliceRandom;
use std::fs;
use tiktoken_rs::{CoreBPE, r50k_base};

pub fn load_file(file: &str) -> String {
    let raw_text = fs::read_to_string(file).expect("Failed to read file");

    raw_text
}

pub struct GPTDataset {
    input_ids: Vec<Tensor>,
    target_ids: Vec<Tensor>,
}

impl GPTDataset {
    pub fn new(
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

pub struct DataLoader {
    dataset: GPTDataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl DataLoader {
    pub fn new(dataset: GPTDataset, batch_size: usize, shuffle: bool, drop_last: bool) -> Self {
        DataLoader {
            dataset,
            batch_size,
            shuffle,
            drop_last,
        }
    }

    // iterator on the data loader
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
    let tokenizer: CoreBPE = r50k_base()?;

    let dataset: GPTDataset = GPTDataset::new(txt, &tokenizer, max_length, stride)?;

    Ok(DataLoader::new(dataset, batch_size, shuffle, drop_last))
}
