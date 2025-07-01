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

    // State for each parameter
    momentum_states: HashMap<String, Vec<f32>>, // First moment (m)
    velocity_states: HashMap<String, Vec<f32>>, // Second moment (v)
    max_velocity_states: HashMap<String, Vec<f32>>, // For AMSGrad variant
}

impl AdamWOptimizer {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01, // Default weight decay for AdamW
            step_count: 0,
            amsgrad: false,
            momentum_states: HashMap::new(),
            velocity_states: HashMap::new(),
            max_velocity_states: HashMap::new(),
        }
    }

    pub fn with_params(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        amsgrad: bool,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step_count: 0,
            amsgrad,
            momentum_states: HashMap::new(),
            velocity_states: HashMap::new(),
            max_velocity_states: HashMap::new(),
        }
    }

    pub fn zero_grad(&mut self) {
        // In a real implementation, this would clear gradients on parameters
        // The exact implementation depends on your tensor/parameter system
    }

    pub fn step(&mut self, parameters: &mut HashMap<String, Parameter>) {
        self.step_count += 1;

        // Bias correction terms
        let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);

        for (name, param) in parameters.iter_mut() {
            if let Some(grad) = &param.grad {
                // Initialize states if not exists
                if !self.momentum_states.contains_key(name) {
                    self.momentum_states
                        .insert(name.clone(), vec![0.0; param.data.len()]);
                    self.velocity_states
                        .insert(name.clone(), vec![0.0; param.data.len()]);
                    if self.amsgrad {
                        self.max_velocity_states
                            .insert(name.clone(), vec![0.0; param.data.len()]);
                    }
                }

                let momentum = self.momentum_states.get_mut(name).unwrap();
                let velocity = self.velocity_states.get_mut(name).unwrap();

                // Update parameters using AdamW algorithm
                for i in 0..param.data.len() {
                    let g = grad[i];

                    // Update biased first moment estimate (momentum)
                    momentum[i] = self.beta1 * momentum[i] + (1.0 - self.beta1) * g;

                    // Update biased second raw moment estimate (velocity)
                    velocity[i] = self.beta2 * velocity[i] + (1.0 - self.beta2) * g * g;

                    // Compute bias-corrected estimates
                    let m_hat = momentum[i] / bias_correction1;
                    let v_hat = velocity[i] / bias_correction2;

                    // For AMSGrad variant, maintain max of v_hat
                    let v_hat_corrected = if self.amsgrad {
                        let max_v = self.max_velocity_states.get_mut(name).unwrap();
                        max_v[i] = max_v[i].max(v_hat);
                        max_v[i]
                    } else {
                        v_hat
                    };

                    // AdamW update: Apply weight decay directly to parameters (not gradients)
                    // param = param * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)

                    // Weight decay term (applied directly to parameter)
                    if self.weight_decay > 0.0 && param.apply_weight_decay {
                        param.data[i] *= 1.0 - self.learning_rate * self.weight_decay;
                    }

                    // Gradient-based update
                    let update =
                        self.learning_rate * m_hat / (v_hat_corrected.sqrt() + self.epsilon);
                    param.data[i] -= update;
                }
            }
        }
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    pub fn get_learning_rate(&self) -> f32 {
        self.learning_rate
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
