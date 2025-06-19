use once_cell::sync::OnceCell;
use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};
use std::sync::Mutex;

static GLOBAL_RNG: OnceCell<Mutex<StdRng>> = OnceCell::new();

//init the global rng
pub fn seed(seed: u64) {
    let rng = StdRng::seed_from_u64(seed);
    // ignored if already initialized
    let _ = GLOBAL_RNG.set(Mutex::new(rng));
}

/// Obtain a locked handle to the global RNG
pub fn rng() -> std::sync::MutexGuard<'static, StdRng> {
    GLOBAL_RNG
        .get_or_init(|| {
            // this closure runs only once, on first access
            Mutex::new(StdRng::from_os_rng())
        })
        .lock()
        .expect("RNG mutex poisoned")
}
/// Draw a random f32 in [0,1)
pub fn random_f32() -> f32 {
    let mut r = rng();
    r.next_u32() as f32 / (u32::MAX as f32)
}

/// Draw a random f32 in [lo, hi)
pub fn random_range_f32(lo: f32, hi: f32) -> f32 {
    let mut r = rng();
    r.random_range(lo..hi)
}
