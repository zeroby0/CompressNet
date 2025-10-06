"""
Author: Aravind Voggu Reddy, Yash Sengupta

This script performs post-training quantization-aware training (QAT) 
on a pre-trained encoder from CompressNet, followed by conversion to 
TFLite format with robust I/O quantization support. It also provides 
helper functions for inference that automatically handle quantized 
inputs/outputs and compare the teacher, QAT, and TFLite outputs.

Workflow Overview:
------------------
1. Load Pre-trained Model
   - Load the full trained Keras model from disk.
   - Extract the encoder component for QAT.

2. Dataset Preparation
   - Load a subset of training images using `make_dataset`.
   - Expand dims to match encoder input shape (N, H, W, 1).
   - Compute latent targets via the original encoder (teacher outputs).

3. Quantization-Aware Training (QAT)
   - Wrap the encoder using `tfmot.quantization.keras.quantize_model`.
   - Define a diversity loss to encourage high variance across latent features.
   - Train using a custom loop:
       - MSE loss between predicted latent and teacher latent
       - Diversity regularization (1e-3 factor)
       - Optimizer: Adam(1e-4)
       - Batch size: 32, epochs: 100

4. TFLite Conversion
   - Create a representative dataset generator for calibration.
   - Convert the QAT encoder to TFLite with `tf.lite.Optimize.DEFAULT`.
   - Optionally support full integer quantization (int8 I/O).

5. Robust TFLite Inference
   - Functions `quantize_input` and `dequantize_output` handle dtype conversion.
   - `run_tflite_inference_auto` runs inference on input arrays and automatically
     handles quantization/dequantization for float or integer TFLite models.

6. Comparison and Validation
   - Compare outputs of:
       - Teacher encoder (float)
       - Keras QAT encoder
       - TFLite model (dequantized)
   - Print small sample comparisons and MSE per sample.
   - Optionally compute per-dimension variance/std statistics.

Notes:
------
- The script assumes the encoder input shape is (256,256,1) and outputs a latent vector.
- `train_subset` is limited to a manageable number of samples to speed up QAT.
- The diversity loss encourages the latent representations to avoid collapse.
- TFLite conversion uses a representative dataset for better quantization scaling.
- Quantized I/O support ensures robust inference across different TFLite target platforms.
"""


import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
from pathlib import Path
import numpy as np
from CompressNet.src_final.dataset import make_dataset

# === Paths ===
saved_model_path = Path("saved_models/tiny-basic-02_3/best_ssimmae.keras")
tflite_path = Path("saved_models/tiny-basic-02_3/encoder_quant_div.tflite")

# === Step 1: Load full trained model and extract encoder ===
full_model = keras.models.load_model(saved_model_path)
encoder = full_model.get_layer("tiny_encoder")
encoder.summary()

# === Step 2: Prepare a subset of inputs ===
train_x, train_y1, train_y2, metrics = make_dataset(range(1, 4000))
train_subset = np.expand_dims(train_x[:4000], -1).astype('float32')

# Generate latent targets for self-distillation
latent_targets = encoder(train_subset)

# === Step 3: Create QAT encoder ===
qat_encoder = tfmot.quantization.keras.quantize_model(encoder)
optimizer = keras.optimizers.Adam(1e-4)

# === Step 4: Diversity loss ===
@tf.function
def diversity_loss(latent, epsilon=1e-6):
    var = tf.math.reduce_variance(latent, axis=0)
    return 1.0 / (tf.reduce_mean(var) + epsilon)

# === Step 5: Custom training loop for QAT encoder ===
epochs = 100
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices((train_subset, latent_targets)).shuffle(512).batch(batch_size)

for epoch in range(epochs):
    epoch_loss = 0
    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            pred_latent = qat_encoder(x_batch, training=True)
            mse_loss = tf.reduce_mean(tf.square(pred_latent - y_batch))
            loss = mse_loss + 1e-3 * diversity_loss(pred_latent)  # add diversity regularization
        grads = tape.gradient(loss, qat_encoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, qat_encoder.trainable_variables))
        epoch_loss += loss.numpy()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataset):.4f}")


# ------------------------
# Conversion + Robust TFLite inference (with calibration + I/O quantization handling)
# ------------------------

# 1) Representative dataset generator (for converter calibration)
def representative_dataset_generator(num_samples=200):
    # Use first `num_samples` examples from train_subset
    n = min(num_samples, train_subset.shape[0])
    for i in range(n):
        # Each yielded element must be a list/tuple matching model inputs
        # shape: (1, 256, 256, 1) float32
        inp = np.expand_dims(train_subset[i], axis=0).astype(np.float32)
        yield [inp]

# 2) Convert to TFLite with representative dataset (better scales)
converter = tf.lite.TFLiteConverter.from_keras_model(qat_encoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = lambda: representative_dataset_generator(256)

# Optional: force full integer quantization (uncomment if you want int8 I/O)
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8

tflite_model = converter.convert()
tflite_path.parent.mkdir(exist_ok=True, parents=True)
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"Converted and saved TFLite model to: {tflite_path}")

# 3) Helper functions to quantize/dequantize for the TFLite interpreter
def quantize_input(input_data, input_detail):
    # input_data: float32 numpy array with shape matching input_detail['shape'] (or batched)
    scale, zero_point = input_detail['quantization']
    if scale == 0:
        # No quantization parameters (model kept float32)
        return input_data.astype(input_detail['dtype'])
    q_dtype = input_detail['dtype']
    # quantize: q = round(input/scale) + zero_point
    q = np.round(input_data / scale + zero_point).astype(q_dtype)
    return q

def dequantize_output(output_data, output_detail):
    scale, zero_point = output_detail['quantization']
    if scale == 0:
        return output_data.astype(np.float32)
    return (output_data.astype(np.float32) - zero_point) * scale

# 4) Initialize interpreter and print I/O details
interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite input details:", input_details)
print("TFLite output details:", output_details)

# 5) Robust inference function that handles quantized I/O automatically
def run_tflite_inference_auto(interpreter, input_array):
    """
    Runs TFLite inference on input_array (shape: [N, H, W, C], float32).
    Automatically quantizes per input_details dtype and dequantizes outputs.
    Returns dequantized outputs as float32 numpy array of shape (N, out_dim).
    """
    input_det = interpreter.get_input_details()[0]
    output_det = interpreter.get_output_details()[0]

    outputs = []
    for i in range(len(input_array)):
        single = np.expand_dims(input_array[i], axis=0).astype(np.float32)  # shape (1,H,W,C)

        # Quantize input if necessary
        if input_det['dtype'] in (np.int8, np.uint8):
            q_input = quantize_input(single, input_det)
            interpreter.set_tensor(input_det['index'], q_input)
        else:
            # float32 model
            interpreter.set_tensor(input_det['index'], single)

        interpreter.invoke()

        out = interpreter.get_tensor(output_det['index'])
        # Dequantize output if necessary
        deq = dequantize_output(out, output_det)
        outputs.append(deq[0])  # remove batch dim

    return np.array(outputs)

# 6) Quick checks: compare teacher, Keras QAT outputs, and TFLite outputs
# Select small sample
N = 16
sample_inputs = train_subset[:N]  # shape (N, H, W, C)

# Teacher (float encoder)
teacher_latents = encoder(sample_inputs, training=False).numpy()
print("Teacher latents (float) shape:", teacher_latents.shape)

# Keras QAT encoder outputs (float, simulated quantization nodes present)
qat_keras_latents = qat_encoder(sample_inputs, training=False).numpy()
print("Keras QAT latents (float) shape:", qat_keras_latents.shape)

# TFLite outputs (dequantized to float)
tflite_latents = run_tflite_inference_auto(interpreter, sample_inputs)
print("TFLite dequantized latents shape:", tflite_latents.shape)

# 7) Print small samples for visual comparison
np.set_printoptions(precision=4, suppress=True)
print("\n--- Sample comparison (first 6 rows) ---")
for i in range(min(6, N)):
    print(f"Sample {i}:")
    print(" teacher :", np.round(teacher_latents[i], 4))
    print(" qat_keras:", np.round(qat_keras_latents[i], 4))
    print(" tflite  :", np.round(tflite_latents[i], 4))
    print("  mse(qat_keras vs teacher):", np.mean((qat_keras_latents[i]-teacher_latents[i])**2))
    print("  mse(tflite vs teacher):   ", np.mean((tflite_latents[i]-teacher_latents[i])**2))
    print("")

# 8) Optional: quick statistics (variance per-dim)
print("Teacher per-dim std:", np.round(np.std(teacher_latents, axis=0), 4))
print("QAT keras per-dim std:", np.round(np.std(qat_keras_latents, axis=0), 4))
print("TFLite per-dim std:", np.round(np.std(tflite_latents, axis=0), 4))

# End of block
