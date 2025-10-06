"""
Author: Aravind Voggu Reddy, Yash Sengupta

This script performs latent vector extraction using a TFLite-quantized encoder
and conducts clustering on the latent space to find representative patterns.
It includes robust quantization handling for inference, batch processing,
and an elbow-method approach to determine the optimal number of clusters.

Workflow Overview:
------------------
1. Quantization helpers
   - Functions `quantize_input` and `dequantize_output` handle conversions
     between float32 and int8/uint8 as required by the TFLite model.

2. TFLite model loading
   - `load_tflite_model` loads the interpreter, allocates tensors,
     and prints input/output details.

3. Batch inference
   - `run_tflite_batch` runs inference on any batch of inputs
     while automatically handling quantized I/O.

4. Optimal clustering
   - `find_optimal_clusters` applies KMeans clustering on latent vectors.
   - Uses the elbow method with `kneed.KneeLocator` to determine the best k.
   - Plots inertia (within-cluster SSE) vs. number of clusters.

5. Clustering experiment
   - Loads the TFLite encoder.
   - Prepares dataset (subset for speed).
   - Extracts latent vectors for the subset using the encoder.
   - Determines optimal number of clusters and trains final KMeans.
   - Saves cluster centroids and the fitted KMeans model for downstream tasks.
"""



import time
import numpy as np
import tflite_runtime.interpreter as tflite
from CompressNet.src_final.dataset import make_dataset
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# === 1. Quantization helpers ===
def quantize_input(input_data, input_detail):
    """Quantize float32 input to int8/uint8 if required."""
    scale, zero_point = input_detail['quantization']
    if scale == 0:  # no quantization
        return input_data.astype(input_detail['dtype'])
    return np.round(input_data / scale + zero_point).astype(input_detail['dtype'])

def dequantize_output(output_data, output_detail):
    """Dequantize int8/uint8 output back to float32 if required."""
    scale, zero_point = output_detail['quantization']
    if scale == 0:
        return output_data.astype(np.float32)
    return (output_data.astype(np.float32) - zero_point) * scale

# === 2. Load interpreter ===
def load_tflite_model(tflite_path):
    interpreter = tflite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Input details:", input_details)
    print("Output details:", output_details)
    return interpreter, input_details, output_details

# === 3. Flexible batch inference ===
def run_tflite_batch(interpreter, input_array):
    """Run TFLite inference on any batch of inputs."""
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    outputs = []
    for i in range(input_array.shape[0]):
        sample = np.expand_dims(input_array[i], axis=0)  # (1,H,W,C)

        # Quantize if necessary
        if input_details['dtype'] in (np.int8, np.uint8):
            q_input = quantize_input(sample, input_details)
            interpreter.set_tensor(input_details['index'], q_input)
        else:
            interpreter.set_tensor(input_details['index'], sample.astype(np.float32))

        # Run inference
        interpreter.invoke()

        # Get raw output
        raw_output = interpreter.get_tensor(output_details['index'])

        # Dequantize if necessary
        deq_output = dequantize_output(raw_output, output_details)
        outputs.append(deq_output[0])  # drop batch dim

    return np.stack(outputs, axis=0)


def find_optimal_clusters(latent_vectors, max_clusters=15):
    """
    Finds the optimal number of clusters using the elbow method.
    
    Args:
        latent_vectors (np.ndarray): The latent representations from the encoder.
        max_clusters (int): Maximum number of clusters to try.
    
    Returns:
        int: Optimal number of clusters
    """
    inertias = []
    cluster_range = range(1, max_clusters + 1)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(latent_vectors)
        inertias.append(kmeans.inertia_)  # inertia = sum of squared distances to nearest cluster center
    
    # Plot elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, inertias, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Within-cluster SSE)")
    plt.title("Elbow Method for Optimal k")
    plt.show()

    # Simple heuristic: choose k at the "elbow"
    # (or you can return inertias for manual inspection)
    # For automatic selection, we can use the "knee" detection method
    from kneed import KneeLocator
    kl = KneeLocator(cluster_range, inertias, curve="convex", direction="decreasing")
    
    return kl.knee if kl.knee else max_clusters  # fallback to max_clusters if not found


# === 4. Run clustering experiment ===
if __name__ == "__main__":
    # Load TFLite encoder
    interpreter, input_details, output_details = load_tflite_model(
        "saved_models/tiny-basic-02_3/encoder_quant_div.tflite"
    )

    # Prepare dataset (use subset to save time)
    train_x, _, _, _ = make_dataset(range(1, 4000))
    train_x = np.expand_dims(train_x, -1).astype(np.float32)  # (N,H,W,1)

    # Run encoder to extract latent vectors
    print("Extracting latent vectors...")
    latent_vectors = run_tflite_batch(interpreter, train_x[:4000])  # <-- adjust N
    print("Latent shape:", latent_vectors.shape)

    optimal_k = find_optimal_clusters(latent_vectors, max_clusters=15)
    print(f"Optimal number of clusters: {optimal_k}")

    # Train final KMeans with optimal clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(latent_vectors)
    
    np.save("quantized_centroid.npy", kmeans.cluster_centers_)
    print("Cluster centroids saved to quantized_centroid.npy")

    joblib.dump(kmeans, "quantized_kmeans.pkl")

