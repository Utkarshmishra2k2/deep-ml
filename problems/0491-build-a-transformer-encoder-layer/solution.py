# Cell 1
import numpy as np

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    # Your code here
    batch_size, seq_len, d_model = X.shape
    if num_heads <= 0:
        raise ValueError("num_heads must be a positive integer")
    if d_model % num_heads != 0:
        raise ValueError(
        f"d_model ({d_model}) must be divisible by "
        f"num_heads ({num_heads})"
        )
        
    head_dim = d_model // num_heads
    
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        shifted = x - np.max(x, axis = axis, keepdims = True)
        exp_x = np.exp(shifted)
        
        return exp_x / np.sum(exp_x, axis = axis, keepdims = True)
        
    def layer_norm(x: np.ndarray,gamma: np.ndarray,beta: np.ndarray) -> np.ndarray:
        mean = np.mean(x,axis = -1, keepdims = True)
        variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(variance + eps)
        return gamma * normalized + beta
        
    def split_heads(x: np.ndarray) -> np.ndarray:
        x = x.reshape(batch_size, seq_len, num_heads,head_dim)
        return x.transpose(0,2,1,3)
        
    Q = X @ weights["W_q"]
    K = X @ weights["W_k"]
    V = X @ weights["W_v"]
    
    Q = split_heads(Q)
    K = split_heads(K)
    V = split_heads(V)
    
    attention_scores = Q @ K.transpose(0, 1, 3, 2)
    attention_scores = attention_scores / np.sqrt(head_dim)
    
    attention_weights = softmax(attention_scores, axis=-1)
    
    head_outputs = attention_weights @ V
    
    concatenated = head_outputs.transpose(0, 2, 1, 3)
    
    concatenated = concatenated.reshape(batch_size, seq_len, d_model)
    
    attention_output = concatenated @ weights["W_o"]
    
    norm1 = layer_norm(X + attention_output,weights["gamma1"], weights["beta1"])
    
    hidden = norm1 @ weights["W1"] + weights["b1"]
    hidden = np.maximum(hidden, 0.0)  # ReLU
    ffn_output = hidden @ weights["W2"] + weights["b2"]
    output = layer_norm(
        norm1 + ffn_output,
        weights["gamma2"],
        weights["beta2"]
    )
    
    return output