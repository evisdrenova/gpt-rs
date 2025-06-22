use candle_core::{Device, Error, Tensor};

use crate::layers::{Dropout, Linear};
/*

TODOs
1. when we do auto-grad, we will have to update this to store the computed q,k,v tensors
2. creat a moduel layer  trait that all layers implement
*/

pub struct MultiHeadAttention {
    pub w_query: Linear,
    pub w_key: Linear,
    pub w_value: Linear,
    pub dropout: Dropout,
    pub mask: Tensor,
    pub num_heads: usize,
    pub head_dim: usize,
    pub out_proj: Linear,
    pub d_out: usize,
}

impl MultiHeadAttention {
    pub fn new(
        d_in: usize,
        d_out: usize,
        context_length: usize,
        dropout: f32,
        num_heads: usize,
        bias: Option<bool>,
        device: Device,
    ) -> Result<Self, Error> {
        if d_out % num_heads != 0 {
            return Err(Error::Msg("d_out must be divisble by num_heads".into()));
        }

        let head_dim = d_out / num_heads;

        let bias = bias.unwrap_or(false);

        // create q,k,v matrices and apply dropout
        let w_query = Linear::new(d_in, d_out, bias, &device)?;
        let w_key = Linear::new(d_in, d_out, bias, &device)?;
        let w_value = Linear::new(d_in, d_out, bias, &device)?;

        let out_proj = Linear::new(d_out, d_out, bias, &device);

        let dropout_layer = Dropout::new(dropout);

        // apply causal mask
        let mask = Self::create_causal_mask(context_length, &device)?;

        Ok(MultiHeadAttention {
            w_query,
            w_key,
            w_value,
            dropout: dropout_layer,
            mask,
            num_heads,
            head_dim,
            out_proj: out_proj?,
            d_out,
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

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let input_shape: &[usize] = input.shape().dims();

        // parse input dimensions so we can use them later for reshaping
        let (b, num_tokens) = parse_batch_and_seq(input_shape)?;

        /* Apply linear transformations to create queries, keys, and values
        Input shape: [batch, seq_len, d_in]
        Output shape: [batch, seq_len, d_out] for each Q, K, V*/

        let (mut queries, mut keys, mut values) = self.create_qkv_matrices(input)?;

        // split out attention heads for parallel processing
        // Before: [batch, seq_len, d_out]
        // After: [batch, seq_len, num_heads, head_dim]
        keys = keys.reshape(&[b, num_tokens, self.num_heads, self.head_dim])?;
        queries = queries.reshape(&[b, num_tokens, self.num_heads, self.head_dim])?;
        values = values.reshape(&[b, num_tokens, self.num_heads, self.head_dim])?;

        // group all attention heads together so we can process them in parallel with a  single matrix operation instead of looping
        // Before: [batch, seq_len, num_heads, head_dim]
        // After: [batch, num_heads, seq_len, head_dim]
        keys = keys.transpose(1, 2)?.contiguous()?;
        queries = queries.transpose(1, 2)?.contiguous()?;
        values = values.transpose(1, 2)?.contiguous()?;

        // compute the attention scores
        let keys_t = keys.transpose(2, 3)?.contiguous()?;
        let attn_scores = queries.matmul(&keys_t)?;

        // Apply causal mask for the current num_tokens
        let masked_scores = self.apply_causal_mask_slice(&attn_scores, num_tokens)?;

        // scale scores to make up for dropout mask
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scaled_scores = (masked_scores * scale)?;

        // Apply softmax
        let attn_weights = Self::softmax(&scaled_scores, Some(scaled_scores.rank() - 1))?;

        // Apply dropout
        let attn_weights = self.dropout.forward(&attn_weights)?;

        // Compute context vectors
        let context_vecs = attn_weights.matmul(&values)?;

        // transpose and concat all heads back into a single representation
        let context_vecs = context_vecs.transpose(1, 2)?;
        let context_vecs = context_vecs
            .reshape(&[b, num_tokens, self.d_out])?
            .contiguous()?;

        // Apply output projection
        let context_vecs = self.out_proj.forward(&context_vecs)?;

        Ok(context_vecs)
    }

    pub fn create_qkv_matrices(&self, inputs: &Tensor) -> Result<(Tensor, Tensor, Tensor), Error> {
        let queries = self.w_query.forward(inputs)?;
        let keys = self.w_key.forward(inputs)?;
        let values = self.w_value.forward(inputs)?;

        Ok((queries, keys, values))
    }

    // this could probably be made way more efficieint like we shouldnt have to do an affine transform here
    fn apply_causal_mask_slice(
        &self,
        attn_scores: &Tensor,
        num_tokens: usize,
    ) -> Result<Tensor, Error> {
        let shape = attn_scores.shape().dims();

        // Slice the pre-computed mask to the current sequence length
        let mask_slice = if shape.len() == 4 {
            let batch_size = shape[0];
            let num_heads = shape[1];
            let mask_2d = self
                .mask
                .narrow(0, 0, num_tokens)?
                .narrow(1, 0, num_tokens)?;
            mask_2d
                .unsqueeze(0)?
                .unsqueeze(0)?
                .expand(&[batch_size, num_heads, num_tokens, num_tokens])?
        } else if shape.len() == 3 {
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

    pub fn softmax(input: &Tensor, dim: Option<usize>) -> Result<Tensor, Error> {
        // Softmax formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)

        // Apply exponential function element-wise
        let exp_values = input.exp()?;

        let softmax_result = match dim {
            Some(dimension) => {
                // Row-wise (or dimension-wise) softmax
                // Sum along the specified dimension while keeping dimensions
                let exp_sum = exp_values.sum_keepdim(dimension)?;
                // Normalize: divide each element by the sum along that dimension
                exp_values.broadcast_div(&exp_sum)?
            }
            None => {
                // Sum all exponential values
                let exp_sum = exp_values.sum_all()?;
                // Normalize: divide each exp value by the total sum
                exp_values.broadcast_div(&exp_sum)?
            }
        };

        Ok(softmax_result)
    }
}

pub fn parse_batch_and_seq(dims: &[usize]) -> Result<(usize, usize), Error> {
    match dims.len() {
        1 => Ok((1, dims[0])),
        2 => Ok((dims[0], dims[1])),
        3 => Ok((dims[0], dims[1])),
        _ => Err(Error::Msg(format!(
            "Expected 1D, 2D or 3D input, got {}D",
            dims.len()
        ))),
    }
}
