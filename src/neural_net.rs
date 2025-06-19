use candle_core::{DType, Device, Error, Tensor};
use rand::{Rng, SeedableRng, rngs::StdRng};

// TODO:
// when we do auto-grad, we will have to update this to store teh computed q,k,v tensors

pub struct NeuralNet {
    pub w_query: Linear,
    pub w_key: Linear,
    pub w_value: Linear,
    pub dropout: Dropout,
    pub mask: Tensor,
    d_in: usize,
    d_out: usize,
    device: Device,
}

impl NeuralNet {
    // initializes a new Neural Net with d_in*d_out Linear layers
    pub fn new(
        d_in: usize,
        d_out: usize,
        device: Device,
        context_length: usize,
        dropout: f32,
        seed: Option<u64>,
        bias: Option<bool>,
    ) -> Result<Self, Error> {
        let bias = bias.unwrap_or(false);

        let (query_seed, key_seed, value_seed) = match seed {
            Some(base_seed) => {
                // If seed is provided, use deterministic offsets
                (Some(base_seed), Some(base_seed + 1), Some(base_seed + 2))
            }
            None => {
                // If no seed provided, use random seeds for each layer
                use rand::Rng;
                let mut rng = rand::rng();
                (
                    Some(rng.random::<u64>()),
                    Some(rng.random::<u64>()),
                    Some(rng.random::<u64>()),
                )
            }
        };

        let w_query = Linear::new(d_in, d_out, bias, &device, query_seed)?;
        let w_key = Linear::new(d_in, d_out, bias, &device, key_seed)?;
        let w_value = Linear::new(d_in, d_out, bias, &device, value_seed)?;
        let dropout = Dropout::new(dropout);

        let mask = Self::create_causal_mask(context_length, &device)?;

        Ok(NeuralNet {
            w_query,
            w_key,
            w_value,
            dropout,
            mask,
            d_in,
            d_out,
            device,
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
        println!("Input shape: {:?}", inputs.shape());
        println!("Query weight shape: {:?}", self.w_query.weight.shape());
        // use linear layer's forward to create the matrices
        let queries = self.w_query.forward(inputs)?;
        println!("after the query");
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
        println!("input shape:{:?}", input_shape);
        let (batch_size, num_tokens) = if input_shape.len() == 3 {
            (input_shape[0], input_shape[1])
        } else if input_shape.len() == 2 {
            (1, input_shape[0]) // Treat 2D as batch_size=1
        } else {
            return Err(Error::Msg("Input must be 2D or 3D tensor".into()));
        };

        println!(".5");

        println!("the input shape{:?}", input);

        let (queries, keys, values) = self.create_qkv_matrices(input)?;
        println!(".6");
        // PyTorch: keys.transpose(1, 2) - swap dimensions 1 and 2
        let keys_t = if keys.rank() == 3 {
            keys.transpose(1, 2)? // [batch, d_in, num_tokens]
        } else {
            keys.t()? // [d_in, num_tokens]
        };

        println!("1");
        let attn_scores = queries.matmul(&keys_t)?;

        // Apply causal mask for the current num_tokens
        let masked_scores = self.apply_causal_mask_slice(&attn_scores, num_tokens)?;
        println!("2");
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

    pub fn tril(size: usize, device: &Device) -> Result<Tensor, Error> {
        let mut mask_data = Vec::with_capacity(size * size);

        for row in 0..size {
            for col in 0..size {
                if col <= row {
                    mask_data.push(1.0f32);
                } else {
                    mask_data.push(0.0f32);
                }
            }
        }

        Tensor::from_vec(mask_data, (size, size), device)
    }

    pub fn triu(size: usize, diagonal: i32, device: &Device) -> Result<Tensor, Error> {
        let mut mask_data = Vec::with_capacity(size * size);

        for row in 0..size {
            for col in 0..size {
                // Upper triangular: col > row + diagonal
                if (col as i32) > (row as i32) + diagonal {
                    mask_data.push(1.0f32); // Upper triangle
                } else {
                    mask_data.push(0.0f32); // Lower triangle + diagonal
                }
            }
        }

        Tensor::from_vec(mask_data, (size, size), device)
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

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}
/// Applies an affine linear transformation to the incoming data: y = xA^T + b, where  x is the input tensor, A is a randomly intialized weight matrix and b is a bias term
/// * `in_features` - Size of input features
/// * `out_features` - Size of output features  
/// * `bias` - Whether to include bias term
/// * `device` - Device to create tensors on
/// * `seed` - Optional seed for reproducible initialization
impl Linear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
        seed: Option<u64>,
    ) -> Result<Self, Error> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::seed_from_u64(123), // no seed
        };

        // Glorot uniform initialization to maintain variance across forward/backward prop
        let std_dev = (2.0 / (in_features + out_features) as f64).sqrt() as f32;
        let bound = std_dev * (3.0_f32).sqrt();

        let weight_size = in_features * out_features;

        let mut weight_data: Vec<f32> = Vec::with_capacity(weight_size);
        for _ in 0..weight_size {
            weight_data.push(rng.random_range(-bound..bound));
        }

        // PyTorch: weight shape is [out_features, in_features]
        let weight = Tensor::from_vec(weight_data, (out_features, in_features), device)?;

        let bias = if bias {
            Some(Tensor::zeros(
                out_features,
                candle_core::DType::F32,
                device,
            )?)
        } else {
            None
        };

        Ok(Linear {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let input_shape = input.dims();

        println!("the input in the forward {:?}", input_shape);

        if input_shape.is_empty() {
            return Err(Error::Msg("Input tensor cannot be empty".into()));
        }

        let last_dim = input_shape[input_shape.len() - 1];
        if last_dim != self.in_features {
            return Err(Error::Msg(
                format!(
                    "Input last dimension {} doesn't match in_features {}",
                    last_dim, self.in_features
                )
                .into(),
            ));
        }

        println!("before transpose{:?}", self.weight);

        // transpose the weight matric to [in_features, out_features]
        let weight_t = self.weight.t()?;

        println!("after transpose{:?}", weight_t);
        // broadcast_matmul will auto-rerank matrices so they are compatible
        input.broadcast_matmul(&weight_t).and_then(|out| {
            // optionally add bias
            if let Some(b) = &self.bias {
                out.broadcast_add(b)
            } else {
                Ok(out)
            }
        })
    }

    // get a reference to the weight tensor
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
    // get a reference to the bias tensor
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    // check if layer has a bias
    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    // get the number of layer parameteres
    // weight params + bias params
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.in_features * self.out_features;
        let bias_params = if self.has_bias() {
            self.out_features
        } else {
            0
        };
        weight_params + bias_params
    }
}

pub struct Dropout {
    p: f32,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Dropout { p }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        // 1) get dims and total element count
        let dims = x.shape().dims();
        let n: usize = dims.iter().product();

        // 2) compute scale
        let keep_prob = 1.0 - self.p;
        let scale = 1.0 / keep_prob;

        // 3) build mask_vec
        let mut rng = rand::rng();
        let mut mask_vec = Vec::with_capacity(n);
        for _ in 0..n {
            let r: f32 = rng.random(); // in [0,1)
            if r < self.p {
                mask_vec.push(0.0);
            } else {
                mask_vec.push(scale);
            }
        }

        // 4) make mask Tensor
        let mask = Tensor::from_vec(mask_vec, dims, x.device())?;
        // apply the mask tensor to the input tensor
        Ok((x * mask)?)
    }
}

// todo: creat a layer trait that all layers implement
// todo: update the way that we set seed values so that we can just set it once per struct i.e. like torch.manualSeed(123)
