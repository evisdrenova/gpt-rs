use candle_core::{Device, Error, Tensor};

use crate::layers::{Dropout, Linear};
use crate::module_list::ModuleList;
/*

TODOs
1. when we do auto-grad, we will have to update this to store the computed q,k,v tensors
2. creat a layer trait that all layers implement
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

        // check if there are batches
        let (b, num_tokens) = if input_shape.len() == 3 {
            (input_shape[0], input_shape[1])
        } else if input_shape.len() == 2 {
            (1, input_shape[0])
        } else {
            return Err(Error::Msg("Input must be 2D or 3D tensor".into()));
        };

        // feed forward on the linear layer to create matrices
        let (mut queries, mut keys, mut values) = self.create_qkv_matrices(input)?;

        //reshapes the matrices by adding a num_heads dimension
        keys = keys.reshape(&[b, num_tokens, self.num_heads, self.head_dim])?;
        queries = queries.reshape(&[b, num_tokens, self.num_heads, self.head_dim])?;
        values = values.reshape(&[b, num_tokens, self.num_heads, self.head_dim])?;

        // PyTorch: keys.transpose(1, 2) - swap dimensions 1 and 2
        let keys_t = if keys.rank() == 3 {
            keys.transpose(1, 2)?
        } else {
            keys.t()?
        };

        let queries_t = if queries.rank() == 3 {
            queries.transpose(1, 2)?
        } else {
            queries.t()?
        };

        let values_t = if values.rank() == 3 {
            values.transpose(1, 2)?
        } else {
            values.t()?
        };

        // computes the dot product for each head
        let attn_scores = queries_t.matmul(&keys_t)?;

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
        let mut context_vecs = attn_weights.matmul(&values_t)?;

        context_vecs.transpose(1, 2);

        context_vecs
            .reshape(&[b, num_tokens, self.d_out])?
            .contiguous();

        context_vecs = self.out_proj.forward(&context_vecs)?;

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

// pub struct CausalAttention {
//     pub w_query: Linear,
//     pub w_key: Linear,
//     pub w_value: Linear,
//     pub dropout: Dropout,
//     pub mask: Tensor,
// }

// impl CausalAttention {
//     pub fn new(
//         d_in: usize,
//         d_out: usize,
//         context_length: usize,
//         dropout: f32,
//         bias: Option<bool>,
//         device: &Device,
//     ) -> Result<Self, Error> {
//         let bias = bias.unwrap_or(false);

//         // create q,k,v matrices and apply dropout
//         let w_query = Linear::new(d_in, d_out, bias, &device)?;
//         let w_key = Linear::new(d_in, d_out, bias, &device)?;
//         let w_value = Linear::new(d_in, d_out, bias, &device)?;
//         let dropout_layer = Dropout::new(dropout);

//         // apply causal mask
//         let mask = Self::create_causal_mask(context_length, &device)?;

//         Ok(CausalAttention {
//             w_query,
//             w_key,
//             w_value,
//             dropout: dropout_layer,
//             mask,
//         })
//     }

//     pub fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
//         let input_shape: &[usize] = input.shape().dims();

//         // check if there are batches
//         let num_tokens: usize = if input_shape.len() == 3 {
//             input_shape[1]
//         } else if input_shape.len() == 2 {
//             input_shape[0]
//         } else {
//             return Err(Error::Msg("Input must be 2D or 3D tensor".into()));
//         };

//         let (queries, keys, values) = self.create_qkv_matrices(input)?;

//         // PyTorch: keys.transpose(1, 2) - swap dimensions 1 and 2
//         let keys_t = if keys.rank() == 3 {
//             keys.transpose(1, 2)? // [batch, d_in, num_tokens]
//         } else {
//             keys.t()? // [d_in, num_tokens]
//         };

//         let attn_scores = queries.matmul(&keys_t)?;

//         // Apply causal mask for the current num_tokens
//         let masked_scores = self.apply_causal_mask_slice(&attn_scores, num_tokens)?;

//         // Scale by sqrt(d_k)
//         let d_k = keys.dim(keys.rank() - 1)? as f64;
//         let scale = 1.0 / d_k.sqrt();
//         let scaled_scores = (masked_scores * scale)?;

//         // Apply softmax
//         let attn_weights = Self::softmax(&scaled_scores, Some(scaled_scores.rank() - 1))?;

//         // Apply dropout
//         let attn_weights = self.dropout.forward(&attn_weights)?;

//         // Compute context vectors
//         let context_vecs = attn_weights.matmul(&values)?;

//         Ok(context_vecs)
//     }
// }
