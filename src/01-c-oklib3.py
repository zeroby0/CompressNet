"""
Author: Aravind Voggu Reddy, Yash Sengupta

This module provides core image processing utilities for the CompressNet dataset 
generation and model pipeline. It implements color space conversion (sRGB ↔ Oklab), 
block-based Discrete Cosine Transform (DCT) with quantization, and helper functions 
for working with images as NumPy arrays.

Detailed Workflow:
------------------
1. **Color Space Conversion (sRGB ↔ Oklab)**
   - Implements linear-light sRGB to Oklab conversion (`lsrgb_to_oklab`).
   - Implements Oklab back to sRGB conversion (`oklab_to_lsrgb`).
   - Includes scaling of Oklab values to 0–255 range for image representation 
     (`image_lsrgb_to_oklab` and `image_oklab_to_lsrgb`).
   - Uses pre-calculated min/max ranges (`l_min`, `m_min`, `s_min`, etc.) 
     for normalization.

2. **Image Handling**
   - `read_image_rgb`: Reads an image file into separate R, G, B NumPy arrays.
   - `image_from_arrays`: Combines R, G, B arrays into a PIL Image.
   - `save_image_rgb`: Saves RGB arrays as an image file.
   - These provide a bridge between file-based images and NumPy-based operations.

3. **DCT and IDCT Operations**
   - `perform_dct`: Performs block-wise **Discrete Cosine Transform (DCT-II)** 
     with orthogonal normalization, supporting configurable stride/block size.
   - `perform_idct`: Performs the inverse DCT to reconstruct the image.
   - Both use `scipy.fft.dct` and `scipy.fft.idct`.

4. **Quantization Utilities**
   - `quantise_dct`: Applies JPEG-like quantization using a given quantization table 
     (scaled per block).
   - `unquantise_dct`: Reverses the quantization to reconstruct DCT coefficients.
   - These functions allow mimicking JPEG compression’s quantization step.

5. **Tile Extraction**
   - `split_channel_to_tiles`: Splits an image channel into non-overlapping square 
     tiles of a given size.
   - Returns a list of dictionaries with coordinates (`x`, `y`) and tile data.
   - Useful for patch-based compression and analysis.

6. **Quantization Table**
   - Includes a predefined JPEG-like quantization table (`dqt_90_dct_lum`) 
     used for luminance channels at quality level ~90.
   - This table is used as the base for scaling quantization in other scripts.

Summary:
--------
This module encapsulates all low-level operations needed to:
- Convert images into perceptually uniform Oklab space.
- Perform DCT/IDCT transforms.
- Apply quantization and unquantization.
- Handle image tiles for patch-based processing.

It serves as a fundamental building block for dataset creation, 
compression parameter estimation, and training pipeline of CompressNet.
"""


import numpy as np
import numpy.typing as npt

from pathlib import Path
from PIL import Image
from scipy.fft import dct, idct


l_min = 0.0
l_max = 6.341325663998628
l_range = l_max - l_min

m_min = -1.4831572863679208
m_max = 1.7515803991801318
m_range = m_max - m_min

s_min = -1.9755014508238542
s_max = 1.2591954894854225
s_range = s_max - s_min


nd_uint8 = npt.NDArray[np.uint8]
nd_float64 = npt.NDArray[np.float64]


# fmt: off
dqt_90_dct_lum = np.array([
    3,   2,   2,   3,   5,   8,  10,  12,
    2,   2,   3,   4,   5,  12,  12,  11,
    3,   3,   3,   5,   8,  11,  14,  11,
    3,   3,   4,   6,  10,  17,  16,  12,
    4,   4,   7,  11,  14,  22,  21,  15,
    5,   7,  11,  13,  16,  12,  23,  18,
   10,  13,  16,  17,  21,  24,  24,  21,
   14,  18,  19,  20,  22,  20,  20,  20,
], dtype=np.float64)
# fmt: on


def lsrgb_to_oklab(
    rgb: tuple[nd_uint8, nd_uint8, nd_uint8]
) -> tuple[nd_float64, nd_float64, nd_float64]:
    r, g, b = rgb

    l_ = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)

    okl = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    oka = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    okb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return (okl, oka, okb)


def oklab_to_lsrgb(
    oklab: tuple[nd_float64, nd_float64, nd_float64]
) -> tuple[nd_uint8, nd_uint8, nd_uint8]:
    okl, oka, okb = oklab

    l_ = okl + 0.3963377774 * oka + 0.2158037573 * okb
    m_ = okl - 0.1055613458 * oka - 0.0638541728 * okb
    s_ = okl - 0.0894841775 * oka - 1.2914855480 * okb

    l_ = np.power(l_, 3)
    m_ = np.power(m_, 3)
    s_ = np.power(s_, 3)

    r = +4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    b = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_

    r = np.rint(r).clip(0, 255).astype(np.uint8)
    g = np.rint(g).clip(0, 255).astype(np.uint8)
    b = np.rint(b).clip(0, 255).astype(np.uint8)

    return r, g, b


def image_lsrgb_to_oklab(
    rgb: tuple[nd_uint8, nd_uint8, nd_uint8]
) -> tuple[nd_float64, nd_float64, nd_float64]:
    # We convert the images to oklab, and then scale the numbers
    # to be between 0 and 255
    okl, oka, okb = lsrgb_to_oklab(rgb)

    okl = (okl - l_min) * 255 / l_range
    oka = (oka - m_min) * 255 / m_range
    okb = (okb - s_min) * 255 / s_range

    return (okl, oka, okb)


def image_oklab_to_lsrgb(
    oklab: tuple[nd_float64, nd_float64, nd_float64]
) -> tuple[nd_uint8, nd_uint8, nd_uint8]:
    okl, oka, okb = oklab

    okl = okl * l_range / 255
    oka = oka * m_range / 255
    okb = okb * s_range / 255

    okl = okl + l_min
    oka = oka + m_min
    okb = okb + s_min

    return oklab_to_lsrgb((okl, oka, okb))


def read_image_rgb(image_path: Path) -> tuple[nd_uint8, nd_uint8, nd_uint8]:
    image = np.asarray(Image.open(image_path).convert("RGB"))

    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # r = r[: r.shape[0] - r.shape[0] % 8, : r.shape[1] - r.shape[1] % 8]
    # g = g[: g.shape[0] - g.shape[0] % 8, : g.shape[1] - g.shape[1] % 8]
    # b = b[: b.shape[0] - b.shape[0] % 8, : b.shape[1] - b.shape[1] % 8]

    return r, g, b


def image_from_arrays(rgb: tuple[nd_uint8, nd_uint8, nd_uint8]):
    r, g, b = rgb

    image_cropped = np.zeros([r.shape[0], r.shape[1], 3], dtype=np.uint8)

    image_cropped[:, :, 0] = r
    image_cropped[:, :, 1] = g
    image_cropped[:, :, 2] = b

    return Image.fromarray(image_cropped)


def save_image_rgb(image_path: Path, rgb: tuple[nd_uint8, nd_uint8, nd_uint8]):
    image_from_arrays(rgb).save(image_path)


def perform_dct(im_any: nd_float64, stride: int = 8) -> nd_float64:
    d = lambda x: dct(x.reshape(x.size // stride, stride), norm='ortho', axis=1).reshape(x.shape)

    return d(d(im_any).T).T


def perform_idct(im_any: nd_float64, stride: int = 8) -> nd_float64:
    i = lambda x: idct(x.reshape(x.size // stride, stride), norm='ortho', axis=1).reshape(x.shape)

    return i(i(im_any).T).T


def quantise_dct(im_any: nd_float64, qtable: list[int], stride: int = 8) -> nd_float64:
    qtable = np.array(qtable).reshape((stride, stride))

    im_quant = np.zeros(im_any.shape, dtype=np.float64)

    for i in range(stride):
        row = im_any[i::stride]
        im_quant[i::stride] = (row.reshape(row.size // stride, stride) / qtable[i]).reshape(row.shape)
    
    return im_quant


def unquantise_dct(im_any: nd_float64, qtable: list[int], stride: int = 8) -> nd_float64:
    qtable = np.array(qtable).reshape((stride, stride))

    im_unquant = np.zeros(im_any.shape, dtype=np.float64)

    for i in range(stride):
        row = im_any[i::stride]
        im_unquant[i::stride] = (row.reshape(row.size // stride, stride) * qtable[i]).reshape(row.shape)
    
    return im_unquant


def split_channel_to_tiles(im_any: nd_uint8, tile_size: int = 64):
    tiles_any = []

    for i in np.r_[: im_any.shape[0] : tile_size]:
        for j in np.r_[: im_any.shape[1] : tile_size]:
            tiles_any.append({"x": i, "y": j, "data": im_any[i : i + tile_size, j : j + tile_size]})

    return tiles_any
