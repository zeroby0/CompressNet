# CompressNet: A Lightweight Auto-Encoder Design for Ultra-Fast Compression Parameter Estimation on Edge Devices

This repository contains all the training and inference scripts, along with the dataset used for experimentation. The `README.md` provides all the steps required to reproduce the results presented.

---

## 🚀 Setting Up the Environment

All dependencies have been listed in the `pyproject.toml` file inside the `uv-3.9` folder.  
You can install them using:

```
uv lock
```
## 📂 Downloading the Dataset
The dataset used is [RAISE6k](https://loki.disi.unitn.it/RAISE/getFile.php?p=6k).

Steps:

* Download the dataset and rename the images sequentially: 1.TIF, 2.TIF, 3.TIF, …
* Convert the .TIF images into .png format using any image editor. 
    (This is done to allow faster processing.)

## 🔄 Training Pipeline

The entire training process is divided into **five steps**:

---

### 1. Preparing the Dataset
- **Script**: `01-a-split_thresphold.py`  
- **Functionality**:
  - Takes images from the `dataset` folder.  
  - Resizes them into **256×256** images.  
  - Creates **16×16 patches** from each image.  
  - Applies all **66 threshold configurations** and classifies patches into respective threshold buckets.  
  - Estimates compression parameters for each configuration using:
    - `01-b-Huff.py`  
    - `01-c-oklib3.py`  
  - Stores the estimated parameters for every image in the dataset folder.  

---

### 2. Training the Model
- **Script**: `02-b-train.py`  
- **Functionality**:
  - Trains the **encoder + CNN model** using the prepared dataset.  

---

### 3. Quantization
- **Script**: `03-tflite_converter.py`  
- **Functionality**:
  - Quantizes the trained model.  
  - Retrains the quantized model.  
  - Converts the model into **TensorFlow Lite (`.tflite`) format** for deployment on edge devices.  

---

### 4. Clustering
- **Script**: `04-b-tflite_cluster.py`  
- **Functionality**:
  - Runs inference on training data using the **TFLite encoder model**.  
  - Extracts latent vectors.  
  - Clusters the latent representations for further analysis.  

---

### 5. Inference
- **Script**: `05-tflite_infer.py`  
- **Functionality**:
  - Performs inference using the quantized **TFLite model**.  

---

