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
        let shape = token_ids.dims();

        // handle both 1d and 2d vectors
        let (batch_size, seq_len, is_1d) = match shape.len() {
            1 => (1, shape[0], true),         // 1D input: treat as batch_size=1
            2 => (shape[0], shape[1], false), // 2D input: normal case
            _ => {
                return Err(candle_core::Error::Msg(
                    "Input must be 1D or 2D tensor".into(),
                ));
            }
        };

        // Flatten the token_ids into a 1-dim vector
        let flat_ids = token_ids.flatten_all()?;

        // Index_select on dim=0 (indexes first, then dimension)
        let gathered = self.weights.index_select(&flat_ids, 0)?;

        // Reshape to [batch_size, seq_len, output_dim]
        let output_dim = self.weights.dims()[1];
        let reshaped = gathered.reshape(&[batch_size, seq_len, output_dim])?;

        // If input was 1D, squeeze out the batch dimension to match PyTorch behavior
        if is_1d {
            reshaped.squeeze(0)
        } else {
            Ok(reshaped)
        }
    }
}
