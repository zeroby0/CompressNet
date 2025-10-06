"""
Author: Aravind Voggu Reddy, Yash Sengupta

This script is used to create the dataset for CompressNet. It processes raw images
(from the RAISE6k dataset in .TIF format) into compressed patch-based representations 
and generates statistics required for training the compression parameter estimation model.

Detailed Workflow:
------------------
1. **Image Preprocessing**
   - Reads each `.TIF` image.
   - Performs a center crop to ensure the image is square.
   - Resizes it to a fixed dimension (default: 1024×1024).
   - Creates a smaller 256×256 version and saves it as `.png` in the dataset folder 
     (for faster downstream processing).

2. **Color Space Conversion**
   - Converts RGB image channels into the perceptual **Oklab** color space.
   - Only the **L (lightness)** channel is currently used for further processing.

3. **Variance Computation**
   - Splits the image into tiles of the maximum allowed size (16×16).
   - Computes the variance of each tile to measure local texture/complexity.
   - Variance maps are used for threshold-based adaptive quantization.

4. **DCT Transform and Quantization**
   - Performs block-based **DCT (Discrete Cosine Transform)** on the image.
   - Quantizes DCT coefficients using scaled JPEG-like quantization tables.
   - Supports multiple tile sizes: 4×4, 8×8, and 16×16.
   - Reconstructs the image using inverse DCT (IDCT) for comparison.

5. **Threshold-Based Adaptive Compression**
   - For each pair of thresholds (total: 66 configurations), selects which 
     tiles should use smaller DCT block sizes (4×4 or 8×8) instead of the default 16×16.
   - This allows adaptive compression where high-variance (detail-rich) regions 
     get finer tiles, and low-variance (smooth) regions use coarser tiles.

6. **Quality and Compression Metrics**
   - Reconstructs images for each threshold configuration.
   - Computes **SSIM (Structural Similarity Index)** between the original 
     and reconstructed channel.
   - Encodes DCT coefficient data using:
       - Raw binary storage
       - zlib compression
       - (placeholders for bz2 and lzma compression)
       - Custom **Huffman encoding**
   - Records file sizes for comparison.

7. **Output**
   - For each processed image:
       - Saves the resized `.png` image.
       - Creates a result text file (e.g., `image-l.txt`) containing threshold 
         configurations, SSIM scores, and compressed sizes for each method.

8. **Parallel Execution**
   - Uses Python’s `ProcessPoolExecutor` for parallel image processing.
   - Displays progress with `tqdm`.

Summary:
--------
The generated dataset includes compressed patch statistics and image quality 
metrics across multiple threshold and tile-size configurations. These results 
are later used to train the CompressNet model for ultra-fast compression 
parameter estimation on edge devices.
"""


import io
import tqdm
import numpy as np
import oklib3 as oklib
from PIL import Image
from scipy.ndimage import zoom
from pathlib import Path
import concurrent.futures


from skimage.metrics import structural_similarity
import zlib
import bz2
import lzma

from Huff import encode as huffman_encode


allowed_tile_sizes = sorted([4, 8, 16])


# fmt: off
thresholds = [
    (00, 00), (00, 10), (00, 20), (00, 30), (00, 40), (00, 50), (00, 60), (00, 70), (00, 80), (00, 90), (00, 100),
              (10, 10), (10, 20), (10, 30), (10, 40), (10, 50), (10, 60), (10, 70), (10, 80), (10, 90), (10, 100),
                        (20, 20), (20, 30), (20, 40), (20, 50), (20, 60), (20, 70), (20, 80), (20, 90), (20, 100),
                                  (30, 30), (30, 40), (30, 50), (30, 60), (30, 70), (30, 80), (30, 90), (30, 100),
                                            (40, 40), (40, 50), (40, 60), (40, 70), (40, 80), (40, 90), (40, 100),
                                                      (50, 50), (50, 60), (50, 70), (50, 80), (50, 90), (50, 100),
                                                                (60, 60), (60, 70), (60, 80), (60, 90), (60, 100),
                                                                          (70, 70), (70, 80), (70, 90), (70, 100),
                                                                                    (80, 80), (80, 90), (80, 100),
                                                                                              (90, 90), (90, 100),
                                                                                                       (100, 100),
]
# fmt: on


def get_variances(okx, maxtilesize):
    variances = np.zeros((okx.shape[0]//maxtilesize, okx.shape[1]//maxtilesize), dtype=np.float32)

    for i in np.r_[: okx.shape[0] : maxtilesize]:
        for j in np.r_[: okx.shape[1] : maxtilesize]:
            variances[i//maxtilesize, j//maxtilesize] = np.var(okx[i : i + maxtilesize, j : j + maxtilesize])
    
    return variances


def tiled_encdec_image(okx, tile_size):

    # Larger DCTs will have a more penalising qTable.
    # Sorry, I couldn't think of a better name
    penalty = 1
    if tile_size ==  4: penalty = 1
    if tile_size ==  8: penalty = 4
    if tile_size == 16: penalty = 16

    q_scaled = zoom(oklib.dqt_90_dct_lum.reshape((8, 8)), tile_size / 8.0, order=3)

    qmat = np.rint(q_scaled * penalty).astype(np.uint16)

    okx_dct = oklib.perform_dct(okx, stride=tile_size).astype(np.int16) # NEW: DCT to int16
    okx_dct_q = oklib.quantise_dct(okx_dct, qmat, stride=tile_size).astype(np.int16) # NEW: DCT to int16
    okx_dct_uq = oklib.unquantise_dct(np.rint(okx_dct_q), qmat, stride=tile_size)
    okx_idct = oklib.perform_idct(okx_dct_uq, stride=tile_size)

    return np.rint(okx_idct).astype(np.uint8), okx_dct_q


def read_crop_resize(img_path, size=1024):
    try:
        img = Image.open(img_path)
    except OSError:
        return
    
    w, h = img.size
    
    # Center crop to square
    s = min(w, h)
    s = (s // size) * size  # Ensure multiple of target size
    l = (w - s) // 2
    t = (h - s) // 2
    
    return img.crop((l, t, l + s, t + s)).resize((size, size), Image.LANCZOS)


def process_image(args):

    image_path, output_dir = args

    try:
        image_pil = read_crop_resize(image_path)
        if image_pil is None: return
        image_pil.resize((256, 256), Image.LANCZOS).save(output_dir / f'{image_path.stem}.png', "PNG")
    except OSError:
        return

    rgb = np.asarray(image_pil)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    oklab = oklib.image_lsrgb_to_oklab((r, g, b))
    oklab = [np.rint(okx).astype(np.uint8) for okx in oklab]

    channels = {
        'l': oklab[0],
        # 'a': oklab[1],
        # 'b': oklab[2],
    }

    for channel in channels:
        okx = channels[channel]

        variances_okx = get_variances(okx, max(allowed_tile_sizes))

        source_tiles_oklab_4, dctq_4   = tiled_encdec_image(okx, 4)
        source_tiles_oklab_8, dctq_8   = tiled_encdec_image(okx, 8)
        source_tiles_oklab_16, dctq_16 = tiled_encdec_image(okx, 16)

        # im_4 = Image.fromarray(source_tiles_oklab_4)
        # im_4.save(image_path.parent / channel / 'dct4.png')

        # im_8 = Image.fromarray(source_tiles_oklab_8)
        # im_8.save(image_path.parent / channel / 'dct8.png')

        # im_16 = Image.fromarray(source_tiles_oklab_16)
        # im_16.save(image_path.parent / channel / 'dct16.png')


        results = ''
        for threshold_8, threshold_4 in thresholds:
            okx_result = np.copy(source_tiles_oklab_16) # Initialise from DCT16

            # Tiles that pass threshold for 8 and 4
            thresholdpass_8  = variances_okx > np.percentile(variances_okx, threshold_8)
            thresholdpass_4  = variances_okx > np.percentile(variances_okx, threshold_4)

            # Actual image pixel indexes that need to be copied
            tiles_copymask_8  = thresholdpass_8.repeat(16, axis=0).repeat(16, axis=1)
            tiles_copymask_4  = thresholdpass_4.repeat(16, axis=0).repeat(16, axis=1)


            okx_result[tiles_copymask_8] = source_tiles_oklab_8[tiles_copymask_8]
            okx_result[tiles_copymask_4] = source_tiles_oklab_4[tiles_copymask_4]

            # im = Image.fromarray(okx_result)
            # im.save(image_path.parent / channel / f'({threshold_8}, {threshold_4}).png')

            # SSIM
            ssim = structural_similarity(okx, okx_result)

            # File size
            index_th_4 = np.where(thresholdpass_4.ravel())[0]
            index_th_8 = np.setdiff1d(np.where(thresholdpass_8.ravel())[0], index_th_4)

            dpixels_th_4  = dctq_4[tiles_copymask_4]
            dpixels_th_8  = dctq_8[tiles_copymask_8 & np.invert(tiles_copymask_4)]
            dpixels_th_16 = dctq_16[np.ones(dctq_16.shape, dtype=bool) & np.invert(tiles_copymask_8)]


            data = index_th_4.tobytes() \
                + index_th_8.tobytes() \
                    + dpixels_th_4.tobytes() \
                    + dpixels_th_8.tobytes() \
                    + dpixels_th_16.tobytes()
            
            len_zlib = len(zlib.compress(data))
            len_bz2 = 0 #len(bz2.compress(data))
            len_lzma = 0 #len(lzma.compress(data))

            huff_in = io.BytesIO()
            huff_in.write(data)
            huff_in.seek(0)
            huff_out = io.BytesIO()
            freq_table = huffman_encode(huff_in, huff_out)

            len_jpeg = len(huff_out.getvalue()) + len(freq_table) * 5 + 4

            results += f'threshold=({threshold_8}, {threshold_4}), ssim={ssim:.6f}, orig={len(data)}, zlib={len_zlib}, bz2={len_bz2}, lzma={len_lzma}, jpeg={len_jpeg}\n'
        
        with open(output_dir / f'{image_path.stem}-{channel}.txt', 'w') as resfile:
            resfile.write(results)
        


if __name__ == "__main__":
    input_path = Path("../RAISE_6K")
    output_path = Path("../dataset")
    output_path.mkdir(parents=True, exist_ok=True)

    images_in_corpus = sorted(
        [
            x
            for x in input_path.iterdir()
            if x.suffix in [".TIF"]
            and not (input_path / f"{x.stem}.aria2").exists()
            and not (output_path / f"{x.stem}-l.txt").exists()
        ]
    )

    print(len(images_in_corpus))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, (img, output_path)) for img in images_in_corpus]
        for _ in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass 
