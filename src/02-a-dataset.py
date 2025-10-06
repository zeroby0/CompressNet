"""
Author: Aravind Voggu Reddy, Yash Sengupta

This script generates the dataset for training CompressNet by parsing precomputed
image compression metrics and preparing tensors for model input.

Detailed Workflow:
------------------
1. **Parsing Compression Data**
   - `parse_data(file_path)` reads text files containing SSIM and compressed sizes 
     for each image and threshold configuration.
   - Filters out specific thresholds (e.g., (10,10) and (30,30)) that are not used.
   - Extracts SSIM and zlib-compressed sizes for each image patch.
   - Standardizes SSIM and size values (zero mean, unit variance) for training.
   - Returns SSIM array, size array, and original mean/std metrics.

2. **Preparing Image Tensors**
   - `make_dataset(indices, channel='l')` processes a list of image indices.
   - Loads each image from `../dataset/{index}.png`.
   - Converts RGB image to Oklab color space using `oklib.image_lsrgb_to_oklab`.
   - Extracts the requested channel (default: 'L' for luminance).
   - Converts the channel to a TensorFlow tensor.

3. **Combining Image Data with Metrics**
   - Calls `parse_data` to get standardized SSIM and size values for each image.
   - Stores:
       - `dstype_x`: TensorFlow tensors of image channel data.
       - `dstype_y1`: SSIM values (target for model training).
       - `dstype_y2`: Compressed sizes (alternative target for model training).
   - Stacks all tensors into batched datasets for model training.

4. **Visualization**
   - Computes histogram of SSIM values (`all_y1`) to analyze distribution.
   - Highlights the bin with maximum count in the histogram.
   - Saves histogram as `haha-f.png` for inspection.

5. **Main Execution**
   - Processes all images in the range 1–6000 by default.
   - Prints metrics and shapes of the dataset tensors.
   - Applies optional filtering (commented out) using `ds_filter.drop_close_to_median_tf`.

Summary:
--------
This script bridges raw image data and precomputed compression metrics to produce 
ready-to-use TensorFlow datasets. It supports luminance-channel extraction, metric 
standardization, and visualization of SSIM distributions, forming a crucial step in 
CompressNet training preparation.
"""


import re
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
import oklib3 as oklib

import matplotlib.pyplot as plt


def parse_data(file_path):
    ssims = []
    sizes = []

    with open(file_path, "r") as file:
        for line in file:
            # Extract values using regular expressions
            match = re.match(
                r"threshold=\((\d+),\s*(\d+)\),\s*ssim=([\d.]+),\s*orig=(\d+),\s*zlib=(\d+),\s*bz2=(\d+),\s*lzma=(\d+)",
                line,
            )

            if match:
                threshold = tuple(map(int, match.group(1, 2)))

                if threshold == (10, 10) or threshold == (30, 30):
                    continue

                ssim = (float(match.group(3)),)
                # orig =  int(match.group(4)),
                zlib =  int(match.group(5)),
                # bz2 = (float(match.group(6)),)
                # lzma =  int(match.group(7))

                # BZ2 seems to perform the best
                ssims.append(ssim)
                sizes.append(zlib)

    # ssims = np.round(np.asarray(ssims, dtype=np.float32).ravel() * 1000)
    # sizes = (
    #     np.round(np.asarray(sizes, dtype=np.float32).ravel() / 1000)
    # )  # Reduce precision to KB
    
    # Keep SSIM in [0,1] (no *1000)
    # ssims = np.asarray(ssims, dtype=np.float32).ravel()

    # Use log(KB) for sizes (better behaved than raw KBs)
    # sizes = np.log1p(np.asarray(sizes, dtype=np.float32).ravel() / 1000.0)  # log(size in KB + 1)
    
    sizes = np.asarray(sizes, dtype=np.float32).ravel()
    ssims = np.asarray(ssims, dtype=np.float32).ravel()
    
    ssim_mean = ssims.mean()
    size_mean = sizes.mean()
    ssim_std = ssims.std()
    size_std = sizes.std()
    
    ssims = (ssims - ssims.mean()) / ssims.std()
    sizes = (sizes - sizes.mean()) / sizes.std()


    
    # print("SSIM VALUES")
    # print(ssims)
    # print("SIZE VALUES")
    # print(sizes)
    # print("SIZE NOT IN MB ANYMORE")

    return ssims, sizes, (ssim_mean, ssim_std, size_mean, size_std)


def make_dataset(indices: list, channel = 'l'):
    dstype_x, dstype_y1, dstype_y2 = [], [], []

    for i in indices:
        image_path = Path(f'../dataset/{i}.png')

        if not image_path.exists(): continue

        rgb = np.asarray(Image.open(image_path))
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        oklab = oklib.image_lsrgb_to_oklab((r, g, b))
        oklab = [np.rint(okx).astype(np.uint8) for okx in oklab]

        channels = {
            'l': oklab[0],
            # 'a': oklab[1],
            # 'b': oklab[2],
        }

        image_tensor = tf.convert_to_tensor(channels[channel], dtype=tf.float32)

        ssims, sizes, metrics = parse_data(f'../dataset/{i}-{channel}.txt')


        ssims = tf.constant(ssims)
        sizes = tf.constant(sizes)

        dstype_x.append(image_tensor)
        dstype_y1.append(ssims)
        dstype_y2.append(sizes)


    dstype_x = tf.stack(dstype_x)
    dstype_y1 = tf.stack(dstype_y1)
    dstype_y2 = tf.stack(dstype_y2)

    return dstype_x, dstype_y1, dstype_y2, metrics



if __name__ == '__main__':
    from ds_filter import drop_close_to_median_tf

    # all_x, all_y1, all_y2, metrics = drop_close_to_median_tf(drop_fraction=0.8)(make_dataset)(range(1, 2))
    all_x, all_y1, all_y2, metrics = make_dataset(range(1, 6000))
    print(metrics)
    print("READ")
    print(all_x.shape, all_y1.shape, all_y2.shape)
    
    all_y1 = tf.reshape(all_y1, -1)
    all_y1 = all_y1 * 0.035792038 + 0.8997339
    all_y2 = tf.reshape(all_y2, -1)
    
    # print('ssim mean', tf.math.reduce_mean(all_y1))
    # print('size mean', tf.math.reduce_mean(all_y2))
    counts, bins, patches = plt.hist(all_y1, 50)
    plt.tight_layout()
    
    # Find bin with max count
    max_bin_index = np.argmax(counts)
    max_bin_center = (bins[max_bin_index] + bins[max_bin_index+1]) / 2
    max_count = counts[max_bin_index]

    # Highlight with a vertical line + annotation
    plt.axvline(max_bin_center, color='red', linestyle='--', linewidth=1.5)
    plt.text(max_bin_center, max_count, f'{max_bin_center:.2f}',
            ha='center', va='bottom', color='red', fontsize=8)

    plt.savefig('haha-f.png', dpi=600)
    plt.close()


