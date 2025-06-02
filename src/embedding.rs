use candle_core::{Device, Tensor};

pub struct Embedding {
    pub weights: Tensor,
}

// an emebdding layer is essentialy a single weight-matrix of shape(vocab_size * output_dim) plus a lookup operation at inference time
// we can allocate a tensor of shape[vocab_size*output_dim] to hold our weights
// then for a given batch of tokenIds, we can index-select along the 0th dimension of the embedding-weight matrix
impl Embedding {
    // creates a weight matrix
    pub fn new(vocab_size: i64, output_dim: i64, device: Device) -> Embedding {
        let weights = Tensor::randn(&[vocab_size, output_dim], device);
        Embedding { weights }
    }

    pub fn forward(&self, token_ids: &Tensor) -> Tensor {
        //   If `token_ids` has shape [batch_size, seq_len], then
        //   `weights.index_select(0, token_ids.flatten())` will yield a
        //   [batch_size * seq_len, output_dim] matrix. We then reshape to
        //   [batch_size, seq_len, output_dim].

        let (batch_size, seq_len) = {
            let shape = token_ids.dims();
            (shape[0], shape[1])
        };

        // flatten the token_ids
        let flat_ids = token_ids.view([-1]);

        // index_select on dim=0
        let gathered: Tensor = self.weights.index_select(0, &flat_ids);

        // reshape to [batch_size, seq_len, output)dim]
        let output_dim = self.weights.dims()[1];
        let out = gathered.view([batch_size, seq_len, output_dim]);

        out
    }
}
