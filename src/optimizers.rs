use candle_core::{Error, Tensor};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AdamWOptimizer {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub step_count: usize,
    pub amsgrad: bool,

    // State for each parameter (using Var for gradient tracking)
    momentum_states: HashMap<usize, Tensor>, // First moment (m)
    velocity_states: HashMap<usize, Tensor>, // Second moment (v)
    max_velocity_states: HashMap<usize, Tensor>, // For AMSGrad variant
}

impl AdamWOptimizer {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            step_count: 0,
            amsgrad: false,
            momentum_states: HashMap::new(),
            velocity_states: HashMap::new(),
            max_velocity_states: HashMap::new(),
        }
    }

    pub fn zero_grad(&mut self) {
        // In Candle, this is typically handled automatically
        // or by clearing gradients on Var parameters
    }

    pub fn step(&mut self, parameters: Vec<&mut Tensor>) -> Result<(), Error> {
        self.step_count += 1;

        let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);

        for (param_idx, param) in parameters.iter_mut().enumerate() {
            // In Candle, gradients are accessed differently
            // For now, create a mock gradient to implement the AdamW logic
            let device = param.device();

            // Initialize states if not exists
            if !self.momentum_states.contains_key(&param_idx) {
                let zeros = Tensor::zeros_like(param)?;
                self.momentum_states.insert(param_idx, zeros.clone());
                self.velocity_states.insert(param_idx, zeros);
                if self.amsgrad {
                    self.max_velocity_states
                        .insert(param_idx, Tensor::zeros_like(param)?);
                }
            }

            let momentum = self.momentum_states.get(&param_idx).unwrap();
            let velocity = self.velocity_states.get(&param_idx).unwrap();

            // Update momentum: m = beta1 * m + (1 - beta1) * grad
            let momentum_new = (momentum * self.beta1)? + (&grad * (1.0 - self.beta1))?;

            // Update velocity: v = beta2 * v + (1 - beta2) * grad^2
            let grad_squared = grad.sqr()?;
            let velocity_new = (velocity * self.beta2)? + (&grad_squared * (1.0 - self.beta2))?;

            // Bias correction
            let m_hat = (&momentum_new / bias_correction1)?;
            let v_hat = (&velocity_new / bias_correction2)?;

            // For AMSGrad variant
            let v_hat_corrected = if self.amsgrad {
                let max_v = self.max_velocity_states.get(&param_idx).unwrap();
                let v_hat_max = v_hat.maximum(max_v)?;
                self.max_velocity_states
                    .insert(param_idx, v_hat_max.clone());
                v_hat_max
            } else {
                v_hat
            };

            // AdamW update
            let mut updated_param = param.clone();

            // Weight decay (applied directly to parameter)
            if self.weight_decay > 0.0 {
                let decay_factor = 1.0 - self.learning_rate * self.weight_decay;
                updated_param = (&updated_param * decay_factor)?;
            }

            // Gradient update: param = param - lr * m_hat / (sqrt(v_hat) + eps)
            let denominator = (v_hat_corrected.sqrt()? + self.epsilon)?;
            let update_term = (&m_hat / &denominator)? * self.learning_rate;
            updated_param = (updated_param - update_term)?;

            // Update the parameter tensor by copying the new values
            *param = updated_param;

            // Update states
            self.momentum_states.insert(param_idx, momentum_new);
            self.velocity_states.insert(param_idx, velocity_new);
        }

        Ok(())
    }
}

// Enhanced Parameter structure with weight decay flag
#[derive(Debug, Clone)]
pub struct Parameter {
    pub data: Vec<f32>,
    pub grad: Option<Vec<f32>>,
    pub requires_grad: bool,
    pub apply_weight_decay: bool, // Whether to apply weight decay to this parameter
}

impl Parameter {
    pub fn new(data: Vec<f32>, requires_grad: bool) -> Self {
        Self {
            data,
            grad: None,
            requires_grad,
            apply_weight_decay: true, // Default: apply weight decay
        }
    }

    pub fn new_with_weight_decay(
        data: Vec<f32>,
        requires_grad: bool,
        apply_weight_decay: bool,
    ) -> Self {
        Self {
            data,
            grad: None,
            requires_grad,
            apply_weight_decay,
        }
    }

    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            if let Some(ref mut grad) = self.grad {
                grad.fill(0.0);
            }
        }
    }
}
