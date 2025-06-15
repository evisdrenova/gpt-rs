use candle_core::{Device, Error, Tensor};
use rand::{Rng, SeedableRng, rngs::StdRng};

pub struct NeuralNet {
    pub w_query: Tensor,
    pub w_key: Tensor,
    pub w_value: Tensor,
    pub d_in: usize,
    pub d_out: usize,
    pub device: Device,
    pub bias: bool,
}

impl NeuralNet {
    pub fn new(
        d_in: usize,
        d_out: usize,
        device: Device,
        seed: Option<u64>,
        bias: Option<bool>,
    ) -> Result<Self, Error> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::seed_from_u64(123), // default seed
        };

        // initialize with random weights between [0,1)
        // aims to be equivalent to torch.rand
        // we could prob parallelize this but then seeding gets weird since each thread gets their own RNG...
        let n = d_in * d_out;

        // fill vecs with
        let mut wq: Vec<f32> = vec![0f32; n];
        rng.fill(&mut wq[..]);

        let mut wk: Vec<f32> = vec![0f32; n];
        rng.fill(&mut wk[..]);

        let mut wv: Vec<f32> = vec![0f32; n];
        rng.fill(&mut wv[..]);

        let bias = bias.unwrap_or(false);

        // turn vec -> tensor
        let w_query = Tensor::from_vec(wq, (d_in, d_out), &device)?;
        let w_key = Tensor::from_vec(wk, (d_in, d_out), &device)?;
        let w_value = Tensor::from_vec(wv, (d_in, d_out), &device)?;

        Ok(NeuralNet {
            w_query,
            w_key,
            w_value,
            d_in,
            d_out,
            device,
            bias,
        })
    }

    pub fn create_qkv_matrices(&self, inputs: &Tensor) -> Result<(Tensor, Tensor, Tensor), Error> {
        let queries = inputs.matmul(&self.w_query)?;
        let keys = inputs.matmul(&self.w_key)?;
        let values = inputs.matmul(&self.w_value)?;

        Ok((queries, keys, values))
    }

    pub fn get_weights(&self) -> (&Tensor, &Tensor, &Tensor) {
        (&self.w_query, &self.w_key, &self.w_value)
    }

    // needs work and optimization - simple implementation for now
    pub fn update_weights(
        &mut self,
        w_query: Tensor,
        w_key: Tensor,
        w_value: Tensor,
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
        let keys = input.matmul(&self.w_key)?;
        let queries = input.matmul(&self.w_query)?;
        let values = input.matmul(&self.w_value)?;

        let keys_t = keys.t()?;

        let attn_scores = queries.matmul(&keys_t)?;

        let d_k = keys.dim(keys.rank() - 1)? as f64;
        let scale = 1.0 / d_k.sqrt();

        let scaled_scores = (attn_scores * scale)?;

        let attn_weights = NeuralNet::softmax(&scaled_scores, Some(scaled_scores.rank() - 1))?;

        let context_vecs = attn_weights.matmul(&values)?;

        Ok(context_vecs)
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
        // Standard deviation = sqrt(2.0 / (in_features + out_features))
        let std_dev = (2.0 / (in_features + out_features) as f64).sqrt() as f32;
        let bound = std_dev * (3.0_f32).sqrt();

        let weight_size = in_features * out_features;

        let mut weight_data: Vec<f32> = Vec::with_capacity(weight_size);
        for _ in 0..weight_size {
            weight_data.push(rng.random_range(-bound..bound));
        }

        let weight = Tensor::from_vec(weight_data, weight_size, device)?;

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

        // do forward pass by matmul the input tensor with the weight tensor
        let output = input.matmul(&self.weight)?;

        // Add bias if present
        match &self.bias {
            Some(bias) => output.broadcast_add(bias),
            None => Ok(output),
        }
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
