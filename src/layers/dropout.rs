use candle_core::{Error, Tensor};
use rand::Rng;

pub struct Dropout {
    p: f32,
    training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Dropout { p, training: false }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        if !self.training || self.p == 0.0 {
            return Ok(x.clone());
        }

        // If dropout probability is 1.0, return zeros
        if self.p >= 1.0 {
            return Ok(Tensor::zeros_like(x)?);
        }

        // get dims and total element count
        let dims = x.shape().dims();
        let n: usize = dims.iter().product();

        // compute scale factor for values that don't get dropped out
        let keep_prob = 1.0 - self.p;
        let scale = 1.0 / keep_prob;

        // build mask_vec and push to vector when the rng number is less than the dropout probability
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

        let mask = Tensor::from_vec(mask_vec, dims, x.device())?;

        // apply the mask tensor to the input tensor
        Ok((x * mask)?)
    }
}
