"""
Author: Aravind Voggu Reddy, Yash Sengupta

This script implements a lightweight autoencoder + CNN framework for 
image compression metric prediction (CompressNet). It defines encoder, decoder, 
and CNN modules, combines them into a final model, prepares datasets, and 
performs custom training with normalized loss functions.

Workflow Overview:
------------------
1. **Encoder (`build_encoder`)**
   - Inputs: Grayscale image (default 256x256x1)
   - Four convolutional layers with increasing filter counts and stride 2
     to downsample the image.
   - Global average pooling reduces spatial dimensions while preserving features.
   - Dense layer produces latent vector (default size: 10).

2. **Decoder (`build_decoder`)**
   - Inputs: Latent vector
   - Dense layer reshaped to last encoder feature map
   - Four transposed convolution layers mirror encoder's downsampling
     to reconstruct the image to original size.

3. **CNN Feature Extractor (`build_cnn_model`)**
   - Three convolution + ReLU + max pooling blocks for feature extraction.
   - Dense layers with dropout for regularization.
   - Produces a 128-dimensional output vector for downstream tasks.

4. **Final Model (`build_final_model`)**
   - Combines encoder, decoder, and CNN in sequence.
   - Inputs: Image
   - Outputs: CNN features of reconstructed image.

5. **Dataset Preparation**
   - `make_dataset` (from CompressNet) loads images and their SSIM/file-size metrics.
   - `prepare_dataset` converts them to TensorFlow `Dataset` objects, batching and shuffling.

6. **Custom Loss Functions**
   - `fastloss`: Huber-like loss combining linear and quadratic errors.
   - `train_step` and `valid_step`:
       - Normalized MAE for SSIM and file size prediction
       - Reconstruction loss of autoencoder
       - Total loss = α*(MAE) + γ*(reconstruction)
       - Optional contrastive loss commented out

7. **Training Loop (`train_model`)**
   - Polynomial-decay learning rate with Adamax optimizer.
   - Tracks multiple metrics (train loss, validation loss, SSIM MAE, size MAE)
   - Saves best-performing models according to validation loss, SSIM MAE, and size MAE.
   - Records metrics and configurations in `results.txt` and `config.toml`.

8. **Usage**
   - Define batch size and epochs
   - Prepare train/validation datasets
   - Build final model and train

Note:
-----
- Inputs are expected as grayscale (single-channel) images.
- Target outputs include standardized SSIM and file size metrics.
- This script relies on the `CompressNet.src_final.dataset.make_dataset` function
  for dataset generation, which internally converts images to Oklab color space.
"""

import tensorflow as tf
from pathlib import Path
from CompressNet.src_final.dataset import make_dataset
import toml

def build_encoder(input_shape=(256, 256, 1), latent_dim=10):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Slightly larger filters but still lightweight
    x = tf.keras.layers.Conv2D(6, 3, strides=2, padding='same', activation='relu')(inputs)   # 128x128x6
    x = tf.keras.layers.Conv2D(8, 3, strides=2, padding='same', activation='relu')(x)        # 64x64x8
    x = tf.keras.layers.Conv2D(12, 3, strides=2, padding='same', activation='relu')(x)       # 32x32x12
    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding='same', activation='relu')(x)       # 16x16x16

    # Global average pooling to keep parameters low
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Small dense layer for latent representation
    latent = tf.keras.layers.Dense(latent_dim, name='latent_vector')(x)

    return tf.keras.models.Model(inputs, latent, name="tiny_encoder")



# Step 2: Decoder
def build_decoder(latent_dim=10, output_shape=(256, 256, 1)):
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,))

    # Mirror the encoder's last feature map (16x16x16)
    x = tf.keras.layers.Dense(16 * 16 * 16, activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((16, 16, 16))(x)

    # Reverse strides: 2 → 2 → 2 → 2
    x = tf.keras.layers.Conv2DTranspose(12, 3, strides=2, padding='same', activation='relu')(x)   # 32x32x12
    x = tf.keras.layers.Conv2DTranspose(8, 3, strides=2, padding='same', activation='relu')(x)    # 64x64x8
    x = tf.keras.layers.Conv2DTranspose(6, 3, strides=2, padding='same', activation='relu')(x)    # 128x128x6
    outputs = tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')(x) # 256x256x1

    return tf.keras.models.Model(latent_inputs, outputs, name="decoder")




# Step 3: CNN Model
def build_cnn_model(input_shape=(256, 256, 1)):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(x)

    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(x)

    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(24)(x)  # Was 8
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(128)(x)

    return tf.keras.models.Model(inputs, outputs, name="cnn_model")

# Step 4: Combine Encoder + Decoder + CNN
def build_final_model():
    encoder = build_encoder()
    decoder = build_decoder()
    cnn_model = build_cnn_model()

    inputs = tf.keras.layers.Input(shape=(256, 256, 1))
    latent = encoder(inputs)
    reconstructed = decoder(latent)
    cnn_output = cnn_model(reconstructed)

    return tf.keras.models.Model(inputs, cnn_output, name="final_model")

runtag = 'tiny-basic-02_3'

savedir = Path(f'saved_models/{runtag}/')
savedir.mkdir(exist_ok=True, parents=True)

train_x, train_y1, train_y2, metrics_train = make_dataset(range(1, 4000))
valid_x, valid_y1, valid_y2, _ = make_dataset(range(4000, 5000))

print(metrics_train)
# test_x, test_y1, test_y2 = make_dataset(range(700, 840))

# Prepare the datasets
def prepare_dataset(x, y1, y2, batch_size):
    y = tf.concat([y1, y2], axis=1)

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return dataset.shuffle(1000).batch(batch_size)


@tf.function
def fastloss(y_true, y_pred, delta=1.0):
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    abs_error = tf.abs(y_true - y_pred)

    quadratic = tf.square(abs_error) / (2 * delta)
    linear = (delta * abs_error)

    return tf.math.reduce_mean(tf.where(abs_error <= delta, linear, quadratic), axis=-1)


@tf.function
def train_step(model, optimizer, x, y, alpha=1.0, beta=0.1, gamma=0.5, lambda_recon=1e-5):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(x, training=True)

        # === Reconstruction ===
        encoded = model.get_layer("tiny_encoder")(x)
        reconstructed = model.get_layer("decoder")(encoded)
        recon_loss = tf.reduce_mean(tf.square(tf.expand_dims(x, axis=-1) - reconstructed))
        recon_loss = lambda_recon * recon_loss

        # === Split SSIM and File Size ===
        ssim_true, size_true = y[:, :64], y[:, 64:]
        ssim_pred, size_pred = predictions[:, :64], predictions[:, 64:]

        # === Normalized MAE ===
        mae_ssim = tf.reduce_mean(tf.abs(ssim_true - ssim_pred)) / (tf.math.reduce_std(ssim_true) + 1e-6)
        mae_size = tf.reduce_mean(tf.abs(size_true - size_pred)) / (tf.math.reduce_std(size_true) + 1e-6)

        # === Contrastive Loss (Negative Variance) ===
        # var_loss = -tf.math.reduce_mean(tf.math.reduce_variance(predictions, axis=0))

        # === Total Loss ===
        loss = alpha * (mae_ssim + mae_size) + gamma * recon_loss #+ beta * var_loss

    # Backpropagation
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, predictions



@tf.function
def valid_step(model, x, y, alpha=1.0, beta=0.1, gamma=0.5, lambda_recon=1e-5):
    # Forward pass
    predictions = model(x, training=False)

    # === Reconstruction ===
    encoded = model.get_layer("tiny_encoder")(x)
    reconstructed = model.get_layer("decoder")(encoded)
    recon_loss = tf.reduce_mean(tf.square(tf.expand_dims(x, axis=-1) - reconstructed))
    recon_loss = lambda_recon * recon_loss

    # === Split SSIM and File Size ===
    ssim_true, size_true = y[:, :64], y[:, 64:]
    ssim_pred, size_pred = predictions[:, :64], predictions[:, 64:]

    # === Normalized MAE ===
    mae_ssim = tf.reduce_mean(tf.abs(ssim_true - ssim_pred)) / (tf.math.reduce_std(ssim_true) + 1e-6)
    mae_size = tf.reduce_mean(tf.abs(size_true - size_pred)) / (tf.math.reduce_std(size_true) + 1e-6)

    # === Contrastive Loss (Negative Variance) ===
    # var_loss = -tf.math.reduce_mean(tf.math.reduce_variance(predictions, axis=0))

    # === Total Loss ===
    loss = alpha * (mae_ssim + mae_size) + gamma * recon_loss #+ beta * var_loss

    return loss, predictions



def train_model(model, train_data, valid_data, epochs):
    best_vloss = float("inf")
    best_sizemae = float("inf")
    best_ssimmae = float("inf")

    with open(savedir / "config.toml", "a") as conffile:
        toml.dump(model.get_config(), conffile)


    # LR best performer
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 10000, 1e-4) # LR1
    # lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 50000, 2e-5) # LR2
    # lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 50000, 1e-4) # LR4
    # lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 40000, 1e-5) # LR5
    # lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 60000, 1e-5) # LR6
    # lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 50000, 1e-6) # LR6
    # lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 50000, 5e-5) # LR7

    # lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 50000, 1e-4) # LR9
    
    optimizer = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)

    # Metrics
    train_loss_metric = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
    valid_loss_metric = tf.keras.metrics.Mean("valid_loss", dtype=tf.float32)
    ssim_metric = tf.keras.metrics.MeanAbsoluteError("ssim_mae")
    size_metric = tf.keras.metrics.MeanAbsoluteError("size_mae")

    for epoch in range(epochs):
        # Training loop
        for x_batch, y_batch in train_data:
            loss, predictions = train_step(model, optimizer, x_batch, y_batch)
            train_loss_metric.update_state(loss)

        # Validation loop
        for x_batch, y_batch in valid_data:
            loss, predictions = valid_step(model, x_batch, y_batch)
            valid_loss_metric.update_state(loss)

            ssim_metric.update_state(y_batch[:, :64], predictions[:, :64])
            size_metric.update_state(y_batch[:, 64:], predictions[:, 64:])

        result = (
            f"Epoch {epoch + 1}, "
            f"Train Loss: {train_loss_metric.result():.4f}, "
            f"Valid Loss: {valid_loss_metric.result():.4f}, "
            f"SSIM MAE: {ssim_metric.result():.4f}, "
            f"Size MAE: {size_metric.result():.4f}, "
            f"Best Vloss: {best_vloss:.4f}, "
            f"Best SSIM MAE: {best_ssimmae:.6f}, "
            f"Best Size MAE: {best_sizemae:.6f}"
        )

        # Print metrics
        print(result)

        with open(savedir / "results.txt", "a") as resfile:
            resfile.write(result + "\n")

        if valid_loss_metric.result() < best_vloss:
            model.save(savedir / "bestvloss.keras")
            best_vloss = valid_loss_metric.result()

        if size_metric.result() < best_sizemae:
            model.save(savedir / "best_sizemae.keras")
            best_sizemae = size_metric.result()

        if ssim_metric.result() < best_ssimmae:
            model.save(savedir / "best_ssimmae.keras")
            best_ssimmae = ssim_metric.result()

        # Reset metrics for next epoch
        train_loss_metric.reset_state()
        valid_loss_metric.reset_state()
        ssim_metric.reset_state()
        size_metric.reset_state()


# Usage example
batch_size = 32
epochs = 1000

# Prepare datasets
train_data = prepare_dataset(train_x, train_y1, train_y2, batch_size)
valid_data = prepare_dataset(valid_x, valid_y1, valid_y2, batch_size)

final_model = build_final_model()
final_model.summary()

train_model(final_model, train_data, valid_data, epochs)

