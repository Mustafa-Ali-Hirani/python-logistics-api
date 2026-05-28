# multi_head_attention.py
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def single_head_attention(Q, K, V):
    """Computes standard attention for a single head."""
    d_k = K.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    weights = softmax(scores)
    output = np.matmul(weights, V)
    return output, weights

def simulate_multi_head_attention():
    print("====================================================")
    print("      SIMULATING 2-HEAD SELF-ATTENTION (MHA)         ")
    print("====================================================")
    
    # 3 words, embedding dimension = 8 (instead of 4)
    # This allows us to split the 8-dimension vector into 2 heads of dimension 4
    np.random.seed(42)
    seq_len = 3
    d_model = 8
    num_heads = 2
    d_k = d_model // num_heads  # Dimension per head = 4
    
    Q = np.random.uniform(-1, 1, size=(seq_len, d_model))
    K = np.random.uniform(-1, 1, size=(seq_len, d_model))
    V = np.random.uniform(0, 10, size=(seq_len, d_model))
    
    print(f"Original Q Matrix (Shape {Q.shape}):\n{Q}")
    
    # 1. Split Q, K, V into 2 Heads
    print(f"\n[MHA Step 1] Splitting embedding dimension {d_model} into {num_heads} heads of dimension {d_k}...")
    
    # Head 1 gets the first 4 columns, Head 2 gets the last 4 columns
    Q1, Q2 = Q[:, :d_k], Q[:, d_k:]
    K1, K2 = K[:, :d_k], K[:, d_k:]
    V1, V2 = V[:, :d_k], V[:, d_k:]
    
    print(f" -> Head 1 Query Matrix Q1 (Shape {Q1.shape}):\n{Q1}")
    print(f" -> Head 2 Query Matrix Q2 (Shape {Q2.shape}):\n{Q2}")
    
    # 2. Compute Attention for each head independently
    print("\n[MHA Step 2] Computing attention for Head 1...")
    out1, weights1 = single_head_attention(Q1, K1, V1)
    
    print("\n[MHA Step 2] Computing attention for Head 2...")
    out2, weights2 = single_head_attention(Q2, K2, V2)
    
    # 3. Concatenate the outputs from both heads back into a single vector (d_model = 8)
    print("\n[MHA Step 3] Concatenating Head 1 and Head 2 outputs back together...")
    mha_output = np.concatenate((out1, out2), axis=1)
    
    print(f"\nFinal Consolidated Multi-Head Output (Shape {mha_output.shape}):\n{mha_output}")
    print("====================================================")

if __name__ == "__main__":
    simulate_multi_head_attention()