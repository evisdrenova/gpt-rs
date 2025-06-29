use rand::{Rng, SeedableRng, rng as thread_rng};
use rayon::prelude::*;
use std::cell::RefCell;

// Optional: For deterministic seeding per thread
thread_local! {
    static SEEDED_RNG: RefCell<Option<rand::rngs::StdRng>> = RefCell::new(None);
}

/// Initialize with a seed (optional - for reproducibility)
pub fn seed(seed: u64) {
    SEEDED_RNG.with(|rng_cell| {
        *rng_cell.borrow_mut() = Some(rand::rngs::StdRng::seed_from_u64(seed));
    });
}

/// Fast thread-local random f32 in [0,1)
pub fn random_f32() -> f32 {
    thread_rng().random()
}

/// Fast thread-local random f32 in [lo, hi)
pub fn random_range_f32(lo: f32, hi: f32) -> f32 {
    thread_rng().random_range(lo..hi)
}

/// Bulk generation for large tensors (MUCH FASTER)
pub fn random_vec_f32(count: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut rng_instance = thread_rng();
    (0..count)                    
        .map(|_| rng_instance.random_range(lo..hi))
        .collect()
}

/// Compatibility function - keep your existing API
pub fn rng() -> impl Rng {
    thread_rng()
}

pub fn random_vec_f32_parallel(count: usize, lo: f32, hi: f32) -> Vec<f32> {
    let chunk_size = 100_000; // 100k numbers per chunk

    (0..count)
        .into_par_iter()
        .chunks(chunk_size)
        .flat_map(|chunk| {
            let mut rng = thread_rng(); // Each thread gets its own RNG
            chunk
                .into_iter()
                .map(|_| rng.random_range(lo..hi))
                .collect::<Vec<_>>()
        })
        .collect()
}
