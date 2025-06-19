use candle_core::{Device, Error, Tensor};

use crate::layers::{Dropout, Linear};

pub struct NeuralNet {
    pub w_query: Linear,
    pub w_key: Linear,
    pub w_value: Linear,
    pub dropout: Dropout,
    pub mask: Tensor,
}

impl NeuralNet {
    // initializes a new Neural Net with d_in*d_out Linear layers
    pub fn new(
        d_in: usize,
        d_out: usize,
        device: Device,
        context_length: usize,
        dropout: f32,
        bias: Option<bool>,
    ) -> Result<Self, Error> {
        let bias = bias.unwrap_or(false);

        // create q,k,v matrices and apply dropout
        let w_query = Linear::new(d_in, d_out, bias, &device)?;
        let w_key = Linear::new(d_in, d_out, bias, &device)?;
        let w_value = Linear::new(d_in, d_out, bias, &device)?;
        let dropout = Dropout::new(dropout);

        let mask = Self::create_causal_mask(context_length, &device)?;

        Ok(NeuralNet {
            w_query,
            w_key,
            w_value,
            dropout,
            mask,
        })
    }

    fn create_causal_mask(context_length: usize, device: &Device) -> Result<Tensor, Error> {
        let mut mask_data = Vec::with_capacity(context_length * context_length);

        for i in 0..context_length {
            for j in 0..context_length {
                mask_data.push(if j > i { 1.0f32 } else { 0.0f32 });
            }
        }

        Tensor::from_vec(mask_data, (context_length, context_length), device)
    }

    pub fn create_qkv_matrices(&self, inputs: &Tensor) -> Result<(Tensor, Tensor, Tensor), Error> {
        let queries = self.w_query.forward(inputs)?;
        let keys = self.w_key.forward(inputs)?;
        let values = self.w_value.forward(inputs)?;

        Ok((queries, keys, values))
    }

    pub fn get_weights(&self) -> (&Tensor, &Tensor, &Tensor) {
        (
            &self.w_query.weight,
            &self.w_key.weight,
            &self.w_value.weight,
        )
    }

    // needs work and optimization - simple implementation for now
    pub fn update_weights(
        &mut self,
        w_query: Linear,
        w_key: Linear,
        w_value: Linear,
    ) -> Result<(), Error> {
        self.w_query = w_query;
        self.w_key = w_key;
        self.w_value = w_value;
        Ok(())
    }

    pub fn compute_attention_scores(inputs: &Tensor, query: &Tensor) -> Result<Tensor, Error> {
        // compute dot product
        let attention_scores = inputs.matmul(&query.unsqueeze(1)?)?;
        let scores_flat = attention_scores.flatten_all()?;

        Ok(scores_flat)
    }

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

    // computes the context vector for a single query or index in an input
    pub fn compute_single_context_vector(
        inputs: &Tensor,
        attention_weights: &Tensor,
    ) -> Result<Tensor, Error> {
        // Weighted sum: sum over i of (attention_weights[i] * inputs[i])

        // Reshape attention weights from [seq_len] to [1, seq_len] for matrix multiplication
        let weights_reshaped = attention_weights.unsqueeze(0)?;

        // Matrix multiplication: [1, seq_len] × [seq_len, hidden_dim] = [1, hidden_dim]
        let context_matrix = weights_reshaped.matmul(inputs)?;

        // Flatten to get [hidden_dim] vector
        let context_vector = context_matrix.flatten_all()?;

        Ok(context_vector)
    }

    // computes context vector all queries at once or an entire input
    pub fn compute_context_matrix(
        inputs: &Tensor,
        attention_weights: &Tensor,
    ) -> Result<Tensor, Error> {
        let context_vectors = attention_weights.matmul(inputs)?;

        Ok(context_vectors)
    }

    pub fn softmax(input: &Tensor, dim: Option<usize>) -> Result<Tensor, Error> {
        // Softmax formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)

        // Apply exponential function element-wise
        let exp_values = input.exp()?;

        let softmax_result = match dim {
            Some(dimension) => {
                // Row-wise (or dimension-wise) softmax
                // Sum along the specified dimension while keeping dimensions
                let exp_sum = exp_values.sum_keepdim(dimension)?; // Use 'dimension', not 'dim'

                // Normalize: divide each element by the sum along that dimension
                exp_values.broadcast_div(&exp_sum)?
            }
            None => {
                // Global softmax (original behavior)
                // Sum all exponential values
                let exp_sum = exp_values.sum_all()?;

                // Normalize: divide each exp value by the total sum
                exp_values.broadcast_div(&exp_sum)?
            }
        };

        Ok(softmax_result)
    }

    pub fn compute_attention(inputs: &Tensor, query: &Tensor) -> Result<(Tensor, Tensor), Error> {
        // Step 1: Compute attention scores
        let scores = Self::compute_attention_scores(inputs, query)?;

        // Step 2: Apply softmax to get attention weights
        let weights = Self::softmax(&scores, Some(1))?;

        // Step 3: Compute context vector
        let context = Self::compute_context_matrix(inputs, &weights)?;

        Ok((weights, context))
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Handle both 2D [num_tokens, d_in] and 3D [batch, num_tokens, d_in]
        let input_shape = input.shape().dims();

        let (batch_size, num_tokens) = if input_shape.len() == 3 {
            (input_shape[0], input_shape[1])
        } else if input_shape.len() == 2 {
            (1, input_shape[0]) // Treat 2D as batch_size=1
        } else {
            return Err(Error::Msg("Input must be 2D or 3D tensor".into()));
        };

        let (queries, keys, values) = self.create_qkv_matrices(input)?;

        // PyTorch: keys.transpose(1, 2) - swap dimensions 1 and 2
        let keys_t = if keys.rank() == 3 {
            keys.transpose(1, 2)? // [batch, d_in, num_tokens]
        } else {
            keys.t()? // [d_in, num_tokens]
        };

        let attn_scores = queries.matmul(&keys_t)?;

        // Apply causal mask for the current num_tokens
        let masked_scores = self.apply_causal_mask_slice(&attn_scores, num_tokens)?;

        // Scale by sqrt(d_k)
        let d_k = keys.dim(keys.rank() - 1)? as f64;
        let scale = 1.0 / d_k.sqrt();
        let scaled_scores = (masked_scores * scale)?;

        // Apply softmax
        let attn_weights = Self::softmax(&scaled_scores, Some(scaled_scores.rank() - 1))?;

        // Apply dropout
        let attn_weights = self.dropout.forward(&attn_weights)?;

        // Compute context vectors
        let context_vecs = attn_weights.matmul(&values)?;

        Ok(context_vecs)
    }

    // adds in teh causal mask to the attention scores
    pub fn apply_causal_mask(attn_scores: &Tensor) -> Result<Tensor, Error> {
        let device = attn_scores.device();
        let shape = attn_scores.shape().dims();
        assert_eq!(shape.len(), 2, "expected [L, L] attn_scores");
        let L = shape[0];

        // 1) build float mask [L, L] where j > i → 1.0, else 0.0
        let mut mask_data = Vec::with_capacity(L * L);
        for i in 0..L {
            for j in 0..L {
                mask_data.push(if j > i { 1.0_f32 } else { 0.0_f32 });
            }
        }
        let float_mask = Tensor::from_vec(mask_data, (L, L), device)?;

        // 2) convert to bool mask via comparison
        //    (float_mask > 0.0) yields a boolean Tensor
        let bool_mask = float_mask.gt(0.0)?;

        // 3) build a -inf tensor of the same shape & dtype as attn_scores
        let neg_inf = Tensor::zeros_like(attn_scores)?.affine(0.0, f64::NEG_INFINITY)?;

        // 4) where(bool_mask) use -inf else original
        let masked = bool_mask.where_cond(&neg_inf, attn_scores)?;

        Ok(masked)
    }

    fn apply_causal_mask_slice(
        &self,
        attn_scores: &Tensor,
        num_tokens: usize,
    ) -> Result<Tensor, Error> {
        let device = attn_scores.device();
        let shape = attn_scores.shape().dims();

        // Slice the pre-computed mask to the current sequence length
        let mask_slice = if shape.len() == 3 {
            // For 3D: [batch, num_tokens, num_tokens]
            let batch_size = shape[0];
            let mask_2d = self
                .mask
                .narrow(0, 0, num_tokens)?
                .narrow(1, 0, num_tokens)?;
            mask_2d
                .unsqueeze(0)?
                .expand(&[batch_size, num_tokens, num_tokens])?
        } else {
            // For 2D: [num_tokens, num_tokens]
            self.mask
                .narrow(0, 0, num_tokens)?
                .narrow(1, 0, num_tokens)?
        };

        // Convert to boolean mask and apply
        let bool_mask = mask_slice.gt(0.0)?;
        let neg_inf = Tensor::zeros_like(attn_scores)?.affine(0.0, f64::NEG_INFINITY)?;

        bool_mask.where_cond(&neg_inf, attn_scores)
    }
}

// todo: creat a layer trait that all layers implement
// todo: when we do auto-grad, we will have to update this to store teh computed q,k,v tensors
