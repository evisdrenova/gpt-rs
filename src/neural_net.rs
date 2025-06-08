use candle_core::{Device, Error, Tensor};

pub struct NeuralNet {}

impl NeuralNet {
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

    pub fn compute_context_vector(
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

    pub fn softmax(input: &Tensor) -> Result<Tensor, Error> {
        // Softmax formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)

        // Apply exponential function element-wise
        let exp_values = input.exp()?;

        // Sum all exponential values
        let exp_sum = exp_values.sum_all()?;

        // Normalize: divide each exp value by the sum
        let softmax_result = exp_values.broadcast_div(&exp_sum)?;

        Ok(softmax_result)
    }

    pub fn attention(inputs: &Tensor, query: &Tensor) -> Result<(Tensor, Tensor), Error> {
        // Step 1: Compute attention scores
        let scores = Self::compute_attention_scores(inputs, query)?;

        // Step 2: Apply softmax to get attention weights
        let weights = Self::softmax(&scores)?;

        // Step 3: Compute context vector
        let context = Self::compute_context_vector(inputs, &weights)?;

        Ok((weights, context))
    }
}
