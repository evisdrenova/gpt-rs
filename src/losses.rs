use candle_core::{Error, Tensor};

use crate::activations::Activations;

pub fn cross_entropy_loss(logits: &Tensor, target: &Tensor) -> Result<Tensor, Error> {
    let log_probs = Activations::log_softmax(logits, Some(1))?;
    // gather log probabilities at the class indixes
    let idx = target.unsqueeze(1)?;

    let true_log_probs = log_probs.gather(&idx, 1)?;

    let flattened = true_log_probs.flatten_all()?;

    let mean_log_prob = flattened.mean(0)?;

    Ok(mean_log_prob.neg()?)
}
