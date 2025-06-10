/*
To run benchmarks:

Run all benchmarks -> cargo bench
Run specific benchmark -> cargo bench attention_scores
Run with HTML report generation -> cargo bench -- --output-format html
Run and open HTML report -> cargo bench && open target/criterion/report/index.html
*/

use candle_core::{Device, Tensor};
use criterion::{Criterion, criterion_group, criterion_main};
use gpt_rs::neural_net::NeuralNet;
use std::hint::black_box;

fn create_test_data() -> Tensor {
    let data = vec![
        0.43f32, 0.15, 0.89, // Your (x^1)
        0.55, 0.87, 0.66, // journey (x^2)
        0.57, 0.85, 0.64, // starts (x^3)
        0.22, 0.58, 0.33, // with (x^4)
        0.77, 0.25, 0.10, // one (x^5)
        0.05, 0.80, 0.55, // step (x^6)
    ];

    Tensor::from_vec(data, (6, 3), &Device::Cpu).unwrap()
}

fn bench_attention_scores_loops(c: &mut Criterion) {
    let inputs = create_test_data();

    c.bench_function("attention_scores_nested_loops", |b| {
        b.iter(|| NeuralNet::compute_attention_scores_matrix(black_box(&inputs)).unwrap())
    });
}

fn bench_attention_scores_vectorized(c: &mut Criterion) {
    let inputs = create_test_data();

    c.bench_function("attention_scores_vectorized", |b| {
        b.iter(|| black_box(&inputs).matmul(&inputs.t().unwrap()).unwrap())
    });
}

fn bench_softmax(c: &mut Criterion) {
    let inputs = create_test_data();
    let attn_scores = inputs.matmul(&inputs.t().unwrap()).unwrap();

    c.bench_function("softmax", |b| {
        b.iter(|| NeuralNet::softmax(black_box(&attn_scores), Some(1)).unwrap())
    });
}

fn bench_context_vectors(c: &mut Criterion) {
    let inputs = create_test_data();
    let attn_scores = inputs.matmul(&inputs.t().unwrap()).unwrap();
    let attn_weights = NeuralNet::softmax(&attn_scores, Some(1)).unwrap();

    c.bench_function("context_vectors", |b| {
        b.iter(|| {
            NeuralNet::compute_context_matrix(black_box(&inputs), black_box(&attn_weights)).unwrap()
        })
    });
}

fn bench_full_attention_pipeline(c: &mut Criterion) {
    let inputs = create_test_data();

    c.bench_function("full_attention_pipeline_loops", |b| {
        b.iter(|| {
            let inputs = black_box(&inputs);
            let attn_scores = NeuralNet::compute_attention_scores_matrix(inputs).unwrap();
            let attn_weights = NeuralNet::softmax(&attn_scores, Some(1)).unwrap();
            NeuralNet::compute_context_matrix(inputs, &attn_weights).unwrap()
        })
    });

    c.bench_function("full_attention_pipeline_vectorized", |b| {
        b.iter(|| {
            let inputs = black_box(&inputs);
            let attn_scores = inputs.matmul(&inputs.t().unwrap()).unwrap();
            let attn_weights = NeuralNet::softmax(&attn_scores, Some(1)).unwrap();
            NeuralNet::compute_context_matrix(inputs, &attn_weights).unwrap()
        })
    });
}

fn bench_comparison_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_comparison");
    let inputs = create_test_data();

    group.bench_function("nested_loops", |b| {
        b.iter(|| NeuralNet::compute_attention_scores_matrix(black_box(&inputs)).unwrap())
    });

    group.bench_function("vectorized", |b| {
        b.iter(|| black_box(&inputs).matmul(&inputs.t().unwrap()).unwrap())
    });

    group.finish();
}

// Benchmark with different input sizes
fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_scaling");

    for size in [6, 12, 24, 48].iter() {
        let data: Vec<f32> = (0..*size * 3).map(|i| (i as f32) * 0.1).collect();
        let inputs = Tensor::from_vec(data, (*size, 3), &Device::Cpu).unwrap();

        group.bench_with_input(format!("nested_loops_size_{}", size), size, |b, _| {
            b.iter(|| NeuralNet::compute_attention_scores_matrix(black_box(&inputs)).unwrap())
        });

        group.bench_with_input(format!("vectorized_size_{}", size), size, |b, _| {
            b.iter(|| black_box(&inputs).matmul(&inputs.t().unwrap()).unwrap())
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_attention_scores_loops,
    bench_attention_scores_vectorized,
    bench_softmax,
    bench_context_vectors,
    bench_full_attention_pipeline,
    bench_comparison_group,
    bench_scaling
);
criterion_main!(benches);
