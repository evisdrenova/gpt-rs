use candle_core::{Device, Result, Tensor};

pub struct Embedding {
    pub weights: Tensor,
}

// an emebdding layer is essentialy a single weight-matrix of shape(vocab_size * output_dim) plus a lookup operation at inference time
// we can allocate a tensor of shape[vocab_size*output_dim] to hold our weights
// then for a given batch of tokenIds, we can index-select along the 0th dimension of the embedding-weight matrix
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
        //   If `token_ids` has shape [batch_size, seq_len], then
        //   `weights.index_select(0, token_ids.flatten())` will yield a
        //   [batch_size * seq_len, output_dim] matrix. We then reshape to
        //   [batch_size, seq_len, output_dim].

        let (batch_size, seq_len) = {
            let shape = token_ids.dims();
            (shape[0], shape[1])
        };

        // flatten the token_ids into a 1-dim vector
        let flat_ids = token_ids.flatten_all()?;

        // index_select on dim=0
        let gathered = self.weights.index_select(&flat_ids, 0)?;

        // reshape to [batch_size, seq_len, output)dim]
        let output_dim = self.weights.dims()[1];
        let out = gathered.reshape(&[batch_size, seq_len, output_dim])?;

        Ok(out)
    }
}
