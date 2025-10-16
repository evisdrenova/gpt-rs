import torch
import time

def time_step(name, func):
    """Helper function to time operations"""
    start = time.perf_counter()
    result = func()
    end = time.perf_counter()
    print(f"{name}: {(end - start) * 1000:.3f} ms")
    return result

# Create input tensor
inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]  # step (x^6)
])

print("Input tensor:")
print(inputs)
print()

# Step 1: Compute attention scores matrix (nested loops)
def compute_attention_scores():
    attn_scores = torch.empty(6, 6)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)
    print("Attention scores:")
    print(attn_scores)
    return attn_scores

# Step 2: Apply row-wise softmax
def apply_softmax(attn_scores):
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print("Attention weights:")
    print(attn_weights)
    return attn_weights

# Step 3: Compute context vectors
def compute_context_vectors(attn_weights):
    all_context_vecs = attn_weights @ inputs
    print("All context vectors:")
    print(all_context_vecs)
    return all_context_vecs

# Time each step
print("=== TIMING EACH STEP ===\n")

attn_scores = time_step("Step 1 - Compute attention scores (nested loops)", compute_attention_scores)
print()

attn_weights = time_step("Step 2 - Apply softmax", lambda: apply_softmax(attn_scores))
print()

all_context_vecs = time_step("Step 3 - Compute context vectors (matmul)", lambda: compute_context_vectors(attn_weights))
print()

# Bonus: Compare with vectorized version
print("=== VECTORIZED COMPARISON ===\n")

def vectorized_attention_scores():
    attn_scores_vec = inputs @ inputs.T
    print("Vectorized attention scores:")
    print(attn_scores_vec)
    return attn_scores_vec

attn_scores_vec = time_step("Step 1 - Vectorized attention scores", vectorized_attention_scores)
print()

# Multiple runs for better accuracy
print("=== PERFORMANCE COMPARISON (1000 runs) ===\n")

num_runs = 1000

# Time nested loops
start = time.perf_counter()
for _ in range(num_runs):
    attn_scores = torch.empty(6, 6)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)
loop_time = time.perf_counter() - start

# Time vectorized version
start = time.perf_counter()
for _ in range(num_runs):
    attn_scores_vec = inputs @ inputs.T
vec_time = time.perf_counter() - start

# Time softmax
start = time.perf_counter()
for _ in range(num_runs):
    attn_weights = torch.softmax(attn_scores, dim=-1)
softmax_time = time.perf_counter() - start

# Time context computation
start = time.perf_counter()
for _ in range(num_runs):
    all_context_vecs = attn_weights @ inputs
context_time = time.perf_counter() - start

print(f"Nested loops (attention scores): {loop_time*1000:.3f} ms")
print(f"Vectorized (attention scores): {vec_time*1000:.3f} ms")
print(f"Softmax: {softmax_time*1000:.3f} ms")
print(f"Context vectors (matmul): {context_time*1000:.3f} ms")
print()
print(f"Speedup (vectorized vs loops): {loop_time/vec_time:.2f}x")
print(f"Total time (loops): {(loop_time + softmax_time + context_time)*1000:.3f} ms")
print(f"Total time (vectorized): {(vec_time + softmax_time + context_time)*1000:.3f} ms")