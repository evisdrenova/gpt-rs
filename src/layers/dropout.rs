use candle_core::{Error, Tensor};
use rand::Rng;

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
