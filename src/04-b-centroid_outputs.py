"""
Author: Aravind Voggu Reddy, Yash Sengupta

This script takes pre-computed cluster centroids from the quantized encoder
and runs them through the decoder + CNN portion of the full model to produce
latent embeddings for downstream tasks (e.g., retrieval, similarity search,
or classification).

Workflow Overview:
------------------
1. Load cluster centroids
   - Load the NumPy array containing centroids extracted from KMeans clustering
     on encoder latent vectors.

2. Load full base model
   - Load the full trained Keras model (encoder+decoder+CNN) for inference.
   - Extract only the decoder and CNN parts if you only want embeddings
     from the cluster centroids.

3. Wrap decoder + CNN
   - Combine the decoder and CNN layers into a single sequential model
     for convenient forward pass on centroids.

4. Generate embeddings
   - Run centroids through decoder + CNN to produce fixed-size embeddings
     (e.g., 128-dimensional feature vectors).

5. Save embeddings
   - Store the resulting embeddings as a NumPy array for future use.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

# === 1. Load cent


import numpy as np
import tensorflow as tf
from pathlib import Path

# === 1. Load centroids ===
centroids = np.load("quantized_centroid.npy")   # shape (n_clusters, 10)

# === 2. Load full base model ===
saved_model_path = Path("saved_models/tiny-basic-02_3/bestvloss.keras")
base_model = tf.keras.models.load_model(saved_model_path)

# If your saved model is a full model with encoder+decoder+cnn,
# and you only need decoder+cnn, extract them like this:
decoder = base_model.get_layer("decoder")   # adjust name if needed
cnn     = base_model.get_layer("cnn_model")       # adjust name if needed

# Wrap decoder+cnn into a new model
decoder_cnn = tf.keras.Sequential([decoder, cnn])

# === 3. Run centroids through decoder+cnn ===
outputs = decoder_cnn.predict(centroids, batch_size=32)  # shape (n_clusters, 128)

# === 4. Save to npy ===
np.save("centroid_embeddings.npy", outputs)
print(f"Saved centroid embeddings: {outputs.shape}")
