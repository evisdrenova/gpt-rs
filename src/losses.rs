use candle_core::{Device, Error, Tensor};
use candle_nn::ops::log_softmax;

pub fn cross_entropy_loss(
    input: &Tensor,
    target: &Tensor,
    batch: &Tensor,
    device: Device,
) -> Result<Tensor, Error> {
    let log_probs = log_softmax(logits, 1)?;
    // gather log probabilities at the class indixes
    let idx = target.unsqueeze(1)?;

    let true_log_probs = log_probs.gather(&idx, 1)?;

    let flattened = true_log_probs.flatten_all()?;

    let mean_log_prob = flattened.mean(0)?;

    Ok(mean_log_prob.neg())
}
