use candle_core::{Device, Result, Tensor};

pub struct Embedding {
    pub weights: Tensor,
}

impl Embedding {
    // creates a weight matrix with mean 0, std of 1 and shape[vocab_size*output_dim
    pub fn new(vocab_size: usize, output_dim: usize, device: Device) -> Result<Embedding> {
        // this is actually faster than parallelizing this i think due to the overhead of thread scheduling
        let weights = Tensor::randn(
            0.0f32,
            1.0f32,
            &[vocab_size as usize, output_dim as usize],
            &device,
        )?;
        Ok(Embedding { weights })
    }

    pub fn from_weights(weights: Tensor) -> Result<Self> {
        let weight_dims = weights.dims();
        if weight_dims.len() != 2 {
            return Err(candle_core::Error::Msg("Weight tensor must be 2D".into()));
        }

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
        let flat_ids = token_ids.flatten_all()?;

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

    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weights]
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weights]
    }

    pub fn parameter_count(&self) -> usize {
        self.weights.elem_count()
    }
}
