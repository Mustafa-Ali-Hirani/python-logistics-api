# lora_simulation.py
import numpy as np

def simulate_lora_layer():
    print("====================================================")
    print("         MATHEMATICAL SIMULATION OF LoRA             ")
    print("====================================================")
    
    # Let's simulate a weight matrix in a hidden layer
    # Input dimension (d) = 100, Output dimension (k) = 100
    d, k = 100, 100
    rank = 4  # Our low-rank dimension (r)
    
    # Simulate an input vector x (e.g., representation of word 'delayed')
    np.random.seed(42)
    x = np.random.uniform(-1, 1, size=(d, 1))
    
    # 1. Base Model Weight Matrix (W_0) - Frozen during training
    W_0 = np.random.normal(0, 0.1, size=(k, d))
    
    # 2. LoRA Matrices (A and B) - Trainable
    # Matrix A is initialized with random Gaussian numbers
    A = np.random.normal(0, 0.1, size=(rank, d))
    # Matrix B is initialized with zeros (ensuring LoRA adds exactly 0 at the start of training)
    B = np.zeros((k, rank))
    
    print(f"Original Weight Matrix W_0 parameters: {W_0.size} (Frozen)")
    print(f"LoRA Matrix A parameters            : {A.size} (Trainable)")
    print(f"LoRA Matrix B parameters            : {B.size} (Trainable)")
    print(f"Total Trainable parameters with LoRA : {A.size + B.size}")
    print(f"Parameter Reduction Ratio           : {100 - ((A.size + B.size) / W_0.size * 100):.2f}%")
    print("-" * 52)
    
    # 3. Compute Base Forward Pass (W_0 * x)
    base_output = np.matmul(W_0, x)
    
    # 4. Compute LoRA Forward Pass (B * A * x)
    # This represents the "adapted/fine-tuned" behavioral change
    # At the start of training, since B is zero, the adapter output is exactly zero
    adapter_output_untrained = np.matmul(B, np.matmul(A, x))
    
    # 5. Let's simulate that we have trained the model, meaning Matrix B now contains learned weights
    # (We simulate trained weights in B by populating it with small values)
    B_trained = np.random.normal(0, 0.05, size=(k, rank))
    adapter_output_trained = np.matmul(B_trained, np.matmul(A, x))
    
    # 6. Combine the outputs (y = W_0 * x + B * A * x)
    scaling_alpha = 8
    scale_factor = scaling_alpha / rank
    
    final_output_untrained = base_output + (scale_factor * adapter_output_untrained)
    final_output_trained = base_output + (scale_factor * adapter_output_trained)
    
    print(f"Base Output Vector (first 3 values):\n{base_output[:3]}")
    print(f"\nAdapter Output (Untrained, first 3 values):\n{adapter_output_untrained[:3]}")
    print(f"\nAdapter Output (Trained, first 3 values):\n{adapter_output_trained[:3]}")
    print(f"\nFinal Fine-Tuned Output Vector (first 3 values):\n{final_output_trained[:3]}")
    print("====================================================")

if __name__ == "__main__":
    simulate_lora_layer()