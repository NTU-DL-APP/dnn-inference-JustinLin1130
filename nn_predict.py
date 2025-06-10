import numpy as np
import json

# === Activation functions ===
def relu(x):
    # Rectified Linear Unit: max(0, x)
    return np.maximum(0, x)

def softmax(x):
    # Numerically stable softmax
    if x.ndim == 1:
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    else:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# === Forward Pass ===
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            activation = cfg.get("activation")
            if activation == "relu":
                x = relu(x)
            elif activation == "softmax":
                x = softmax(x)
    return x

# === Public API ===
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)

# === Test block ===
if __name__ == "__main__":
    sample_input = np.array([[1.0, 2.0, 3.0]])

    print("ReLU:", relu(sample_input))
    print("Softmax:", softmax(sample_input))
