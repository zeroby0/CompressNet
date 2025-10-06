"""
Author: Aravind Voggu Reddy, Yash Sengupta

This script demonstrates running a quantized TFLite encoder on a batch of input images,
extracting latent vectors, and assigning cluster labels using a pre-trained KMeans model.

Workflow Overview:
------------------
1. TFLite Helpers
   - Functions to handle input quantization and output dequantization for TFLite models.

2. Batch inference function
   - Handles both batch-size=1 TFLite models and dynamic-batch models.
   - Returns dequantized latent vectors and total inference time.

3. Parameters
   - Define batch size for inference.

4. Load KMeans model
   - Load the pre-trained KMeans model for clustering latent vectors.

5. Load TFLite encoder
   - Load the quantized encoder in TFLite format.
   - Allocate tensors and fetch input/output details.

6. Prepare dataset
   - Load a subset of images for inference and expand dimensions to match model input.

7. Run TFLite inference batch-wise
   - Extract latent vectors for the batch and record inference time.

8. Run KMeans clustering
   - Predict cluster labels for the latent vectors.
   - Record clustering time and compute total per-sample time.
"""

import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite
from CompressNet.src_final.dataset import make_dataset
import tf_keras
import joblib
import time

# === TFLite helpers ===
def quantize_input(input_data, input_detail):
    scale, zero_point = input_detail['quantization']
    if scale == 0:
        return input_data.astype(input_detail['dtype'])
    return np.round(input_data / scale + zero_point).astype(input_detail['dtype'])

def dequantize_output(output_data, output_detail):
    scale, zero_point = output_detail['quantization']
    if scale == 0:
        return output_data.astype(np.float32)
    return (output_data.astype(np.float32) - zero_point) * scale

# === Run TFLite batch inference (handles batch-size=1 models automatically) ===
def run_tflite_batch(interpreter, batch, input_details, output_details):
    """
    Handles both fixed batch-size=1 TFLite models and dynamic-batch models.
    Returns dequantized latent vectors and total inference time.
    """
    outputs = []
    total_time = 0.0

    for i in range(batch.shape[0]):
        start_time = time.monotonic()
        sample = np.expand_dims(batch[i], axis=0).astype(np.float32)  # shape (1,H,W,C)
        # Quantize if needed
        if input_details['dtype'] in (np.int8, np.uint8):
            sample = quantize_input(sample, input_details)
        interpreter.set_tensor(input_details['index'], sample)

        
        interpreter.invoke()
        

        out = interpreter.get_tensor(output_details['index'])
        out = dequantize_output(out, output_details)
        outputs.append(out[0])  # remove batch dim
        end_time = time.monotonic()
        total_time += (end_time - start_time)

    return np.stack(outputs, axis=0), total_time

# === Parameters ===
BATCH_SIZE = 1024

# === Load KMeans model ===
kmeans_model = joblib.load("quantized_kmeans.pkl")
print("✅ Loaded KMeans model.")

# === Load TFLite encoder ===
interpreter = tflite.Interpreter(model_path="saved_models/tiny-basic-02_3/encoder_quant_div.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# === Prepare dataset ===
train_x, train_y1, train_y2,_ = make_dataset(range(1, 5000))
train_subset = np.expand_dims(train_x[:BATCH_SIZE], -1).astype(np.float32)

# === Run TFLite inference batch-wise ===
latent_vectors, inference_time = run_tflite_batch(interpreter, train_subset, input_details, output_details)

print(latent_vectors.shape)
# === Run KMeans clustering ===
start_time_c = time.monotonic()
cluster_labels = kmeans_model.predict(latent_vectors)
stop_time_c = time.monotonic()
cluster_inference_time = stop_time_c - start_time_c

total_time = inference_time + cluster_inference_time

print(total_time/BATCH_SIZE)
# print((0.0012894379906356335 * BATCH_SIZE) / total_time)
# print(f"Cluster labels for the batch: {cluster_labels}")
