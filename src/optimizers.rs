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

        let bc1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step_count as i32);

        for (idx, param) in parameters.iter_mut().enumerate() {
            /* -----------------------------------------------------------
             * 1.  Fetch gradient; skip this param if there is none
             * -------------------------------------------------------- */
            let Some(grad) = param.grad()? else { continue };

            /* -----------------------------------------------------------
             * 2.  Lazily initialise state tensors on first encounter
             * --------------------------------------------------------- */
            if !self.momentum_states.contains_key(&idx) {
                let zeros = Tensor::zeros_like(param)?;
                self.momentum_states.insert(idx, zeros.clone());
                self.velocity_states.insert(idx, zeros);
                if self.amsgrad {
                    self.max_velocity_states
                        .insert(idx, Tensor::zeros_like(param)?);
                }
            }

            // clones are cheap (Arc) – needed for immutable ops
            let m_prev = self.momentum_states.get(&idx).unwrap().clone();
            let v_prev = self.velocity_states.get(&idx).unwrap().clone();

            /* -----------------------------------------------------------
             * 3.  Update first & second moments
             * --------------------------------------------------------- */
            let m_t = (&m_prev * self.beta1)? + (&grad * (1.0 - self.beta1))?;
            let g2 = grad.sqr()?;
            let v_t = (&v_prev * self.beta2)? + (&g2 * (1.0 - self.beta2))?;

            /* -----------------------------------------------------------
             * 4.  Bias-corrected estimates
             * --------------------------------------------------------- */
            let m_hat = (&m_t / bc1)?;
            let v_hat = (&v_t. / bc2)?;

            /* -----------------------------------------------------------
             * 5.  AMSGrad (optional) – maintain running max of v_hat
             * --------------------------------------------------------- */
            let v_hat_corr = if self.amsgrad {
                let v_max_prev = self.max_velocity_states.get(&idx).unwrap().clone();
                let v_max = v_hat.maximum(&v_max_prev)?;
                // store updated max
                self.max_velocity_states.insert(idx, v_max.clone());
                v_max
            } else {
                v_hat
            };

            /* -----------------------------------------------------------
             * 6.  Weight-decay term (decoupled, as in AdamW paper)
             * --------------------------------------------------------- */
            let mut param_t = if self.weight_decay != 0.0 {
                (param * (1.0 - self.learning_rate * self.weight_decay))?
            } else {
                param.clone()
            };

            /* -----------------------------------------------------------
             * 7.  Parameter update:  θ ← θ − α · m̂ / (√v̂ + ε)
             * --------------------------------------------------------- */
            let denom = (v_hat_corr.sqrt()? + self.epsilon)?;
            let update = (&m_hat / denom)? * self.learning_rate;
            param_t = (&param_t - update)?;

            /* -----------------------------------------------------------
             * 8.  Commit new parameter & moment states
             * --------------------------------------------------------- */
            *param = param_t; // in-place update
            self.momentum_states.insert(idx, m_t);
            self.velocity_states.insert(idx, v_t);
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
