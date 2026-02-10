"""
Change-detection toolbox (raster → binary change mask).


This module aims at detecting change between two images taken at different dates. It is mostly designed to work with Sentinel-1 preprocessed data 
but the user is free to feed it with any raster he dreams of. It provides all the bricks needed to:

  - load two coregistered rasters (that can contain multiple bands) with rasterio 
  - turn NoData values into NaN for robust numerical processing
  - align images by padding when they differ by at most 1 row/column
  - clip and normalize each band independently
  - compute a per-pixel dissimilarity map between the two images
  - split the dissimilarity map into tiles and apply Otsu thresholding per tile
  - reassemble the tiled masks into a global binary mask
  - filter the mask by local density and minimum area of connected components
  - (optionally) write the final mask as a GeoTIFF on disk via `main0`

The main entry point is `main(...)`, which returns the output raster profile
and the final binary mask (and can also save it to `out_path` if provided).

"""



########## IMPORTS ##########

# common
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# OTSU
from skimage.filters import threshold_otsu 

# filtering 
from scipy.ndimage import uniform_filter
from skimage.measure import label, regionprops 
from skimage.morphology import (closing, 
    opening, closing, erosion, dilation,
    disk, remove_small_objects, reconstruction)         ## added



########## NODATA HANDLING ##########

def to_nan(arr: np.ndarray, nodata_values=(-9999, -32768, -3.4028235e38)) -> np.ndarray:
    """
    Replace NoData values by NaN.

    Parameters
    ----------
    arr : np.ndarray
        Input array (any dtype).
    nodata_values : tuple of numbers, optional
        Values to treat as NoData and convert to NaN.

    Returns
    -------
    np.ndarray
        Float32 array with the same shape as `arr`, where all nodata_values have been replaced by NaN.
    """
    out = arr.astype("float32", copy=True) # we cast to float32 so that NaN is representable

    for nd in nodata_values:
        np.putmask(out, out == nd, np.nan) # np.putmask(array, mask, value) replaces array[mask] by value

    return out

########## DIMENSIONS ALIGNMENT BY PADDING ##########


def pad_right(img: np.ndarray, ncols: int = 1, fill_value: float = np.nan) -> np.ndarray:
    """
    Add one or several columns of pixels to the RIGHT side of an image.

    Parameters
    ----------
    img : np.ndarray
        3D (C, H, W) image (even if one band with rasterio opening 3D image with C=1).
    ncols : int, optional
        Number of columns to add (default 1).
    fill_value : float, optional
        Value used to fill the new pixels (NaN by default).

    Returns
    -------
    np.ndarray
        New image with `ncols` extra columns on the right.
    """
    if img.ndim == 3:
        C, H, W = img.shape
        pad_width = ((0, 0), (0, 0), (0, ncols)) # padding on width axis only: (C), (H), (W)
    else:
        raise ValueError("img must be 3D (C, H, W)")

    img_padded = np.pad(img, pad_width=pad_width, mode="constant", constant_values=fill_value)

    return img_padded


def pad_bottom(img: np.ndarray, nrows: int = 1, fill_value: float = np.nan) -> np.ndarray:
    """
    Add one or several rows of pixels at the BOTTOM of an image.

    Parameters
    ----------
    img : np.ndarray
        3D (C, H, W) image (even if one band with rasterio opening 3D image with C=1).
    nrows : int, optional
        Number of rows to add (default 1).
    fill_value : float, optional
        Value used to fill the new pixels (NaN by default).

    Returns
    -------
    np.ndarray
        New image with `nrows` extra rows at the bottom.
    """
    if img.ndim == 3:
        C, H, W = img.shape
        pad_width = ((0, 0), (0, nrows), (0, 0)) # padding on height axis only: (C), (H), (W)
    else:
        raise ValueError("img must 3D (C, H, W)")

    img_padded = np.pad(img, pad_width=pad_width, mode="constant", constant_values=fill_value)
    
    return img_padded


def align_by_padding(img1: np.ndarray, profile1: dict, img2: np.ndarray, profile2: dict, 
                     fill_value: float = np.nan, check_max_diff: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Align two images by adding columns (to the RIGHT) and/or rows (at the BOTTOM) to the smaller one so that both have the same size (H, W).

    We do not modify the profile here, otherwise we would also have to update
    the affine transform, which is not trivial. Later in the pipeline we will crop back to the original size to retrurn to img1's format.

    Parameters
    ----------
    img1 : np.ndarray     First image (2D or 3D).
    profile1 : dict       Rasterio profile for the first image.
    img2 : np.ndarray     Second image (2D or 3D).
    profile2 : dict       Rasterio profile for the second image.
    fill_value : float, optional      Value used for padding (NaN by default).
    check_max_diff : bool, optional   If True, raise an error if the initial height/width difference is greater than 1 row or 1 column.

    Returns
    -------
    img1_out : np.ndarray First image after padding (if any).
    img2_out : np.ndarray Second image after padding (if any).
    """

    # ----- Check that the two profiles are identical EXCEPT for keys directly or indirectly related to size -----

    keys_to_ignore = {"transform", "height", "width", "blockxsize", "blockysize", 'nodata'}                                         ## nodata added

    for key in profile1:
        if key in keys_to_ignore:
            continue  # skips directly to the next key
        if key not in profile2:
            raise ValueError(
                f"[ERROR] align_by_padding: key '{key}' is missing in the second profile.")
        if profile1[key] != profile2[key]:
            raise ValueError(
                f"[ERROR] align_by_padding: profiles differ on key '{key}'.\n"
                f"profile1[{key!r}] = {profile1[key]!r}\n"
                f"profile2[{key!r}] = {profile2[key]!r}\n"
                "Images are not spatially compatible for spatial alignment."
            )

    # ----- Helper to get current H, W -----
    def hw(arr: np.ndarray) -> tuple[int, int]:
        return arr.shape[-2], arr.shape[-1]  # (H, W)

    H1, W1 = hw(img1)
    H2, W2 = hw(img2)

    # Differences (positive => img2 larger than img1)
    dH = H2 - H1
    dW = W2 - W1

    if check_max_diff and (abs(dH) > 1 or abs(dW) > 1):
        raise ValueError(f"Size difference greater than 1 row/column: dH={dH}, dW={dW}")

    img1_out, img2_out = img1, img2

    # ----- Align height (rows): pad at the BOTTOM -----
    if dH > 0:
        # img1 is smaller in H -> pad img1
        img1_out = pad_bottom(img1_out, nrows=dH, fill_value=fill_value)
        H1 = H2  
    elif dH < 0:
        # img2 is smaller in H -> pad img2
        img2_out = pad_bottom(img2_out, nrows=-dH, fill_value=fill_value)
        H2 = H1  

    # ----- Align width (columns): pad to the RIGHT -----
    if dW > 0:
        # img1 is smaller in W -> pad img1
        img1_out = pad_right(img1_out, ncols=dW, fill_value=fill_value)
        W1 = W2
    elif dW < 0:
        # img2 is smaller in W -> pad img2
        img2_out = pad_right(img2_out, ncols=-dW, fill_value=fill_value)
        W2 = W1

    return img1_out, img2_out


########## CLIPPING AND NORMALIZATION OF BANDS ##########

def clip_percentiles(img: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """
    Clips extreme values of each band between percentiles p_low and p_high.
    Input: array (nb_bands, height, width)
    Output: array of the same shape
    Handles NaN values.
    """
    out = np.empty_like(img, dtype=float)

    for i in range(img.shape[0]):
        band = img[i]
        low = np.nanpercentile(band, p_low)
        high = np.nanpercentile(band, p_high)
        out[i] = np.clip(band, low, high)

    return out


def normalize_band(arr: np.ndarray) -> np.ndarray:
    """
    Normalizes a SINGLE band.
    Handles NaN values.
    """
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)

    if arr_max == arr_min:
        return np.zeros_like(arr, dtype=float)

    return (arr - arr_min) / (arr_max - arr_min)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalizes a multi-band image: calls normalize_band independently for each band (otherwise we would normalize different bands at the same time).
    """
    img_norm = np.empty_like(img, dtype=float) # creates empty image of same shape
    for i in range(img.shape[0]):
        img_norm[i] = normalize_band(img[i])
    return img_norm


########## DISSIMILARITY MATRIX ##########

def dissimilarity(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Gives a similarity index between two images between 0 and 1,
    where 1 = very similar pixels, and 0 = very different pixels.
    """

    # Basic verification
    if img1.shape != img2.shape:
        raise ValueError("Both images must have the same shape (bands, height, width)")

    # Vectorized computation of the Euclidean distance
    dist = np.sqrt(np.sum((img1 - img2) ** 2, axis=0))  # sum over the band axis -> for each pixel we compute Euclidean distance between the vectors formed by band values
    dist = dist / (np.sqrt(img1.shape[0]))  # if Euclidean distance is 0, dissimilarity = 0, if it is sqrt(nb of bands) (maximum by Pythagorean theorem), dissimilarity = 1
    return dist



def smart_dissimilarity(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:   ## added
    """
    Gives a similarity index between two images between 0 and 1,
    where 0 = very similar (or img2 > img1), and 1 = very different (img1 >> img2).
    Only positive differences (img1 - img2 > 0) are considered, so as to track precisely 
    the changes corresponding to a deforested area where the backscatter dropped 
    # to adapt   
    """

    # Basic verification
    if img1.shape != img2.shape:
        raise ValueError("Both images must have the same shape (bands, height, width)")

    # Keep only positive differences: if (img1 - img2) < 0, result is 0
    # This captures only the decrease in signal intensity
    pos_diff = np.maximum(0, img1 - img2)

    # Vectorized computation of the Euclidean distance on positive differences
   
    dist = np.sqrt(np.sum(pos_diff ** 2, axis=0))  # sum over the band axis (axis=0)
    dist = dist / (np.sqrt(img1.shape[0]) + 1)  # if Euclidean distance is 0, dissimilarity = 0, if it is sqrt(nb of bands) (maximum by Pythagorean theorem), dissimilarity = 1
    
    return dist


########## TILING ##########

def tile_image_2d(img2d: np.ndarray, n: int, fill_value=np.nan):
    """
    Cuts a 2D image (H, W) into n x n tiles of equal size.
    If H or W are not multiples of n, padding is added with `fill_value`
    on the bottom/right edges. This implies that analysis quality on the
    bottom and right borders is strongly degraded.

    Returns:
      tiles : np.ndarray with shape (n, n, tile_height, tile_width)
      meta  : dict with infos (tile_height, tile_width, pad_bottom, pad_right)
    """

    assert img2d.ndim == 2, "img2d must be 2D (H, W)."
    H, W = img2d.shape

    # Tile sizes (using ceil to avoid losing pixels → padding if needed)
    tile_height = int(np.ceil(H / n))
    tile_width  = int(np.ceil(W / n))

    # Padded dimensions so we fit exactly n * n tiles
    H_pad = tile_height * n
    W_pad = tile_width * n
    pad_bottom = H_pad - H
    pad_right  = W_pad - W

    # Bottom/right padding
    img_pad = np.pad(img2d, pad_width=((0, pad_bottom), (0, pad_right)), mode="constant", constant_values=fill_value)

    # Reshape (n*tile_height, n*tile_width) → (n, tile_height, n, tile_width) (memory just interpreted differently) then permute → (n, n, tile_height, tile_width)
    # Goal: tiles[i][j] → 2D tile (tile_height, tile_width)
    tiles = img_pad.reshape(n, tile_height, n, tile_width).transpose(0, 2, 1, 3)

    # Metadata dict (useful for reconstruction)
    meta = dict(tile_height=tile_height, tile_width=tile_width, pad_bottom=pad_bottom, pad_right=pad_right)

    return tiles, meta



########## THRESHOLDING ##########

def otsu_tile(img: np.ndarray, k: float = 1.0, nan_as_bg: bool = True, foreground: str = "high"):
    """
    Global Otsu on a 2D image (values expected in [0, 1]).

    Parameters
    ----------
    img : np.ndarray
        2D image. NaN allowed.
    k : float
        Threshold scaling factor.
    nan_as_bg : bool
        If True, NaN become background in the final mask (background).
        If False, NaN become object in the final mask (foreground).
    foreground : {"high", "low"}
        "high"  -> pixels > threshold are set to 1 ; "low" -> pixels <= threshold are set to 1.
        Here we consider "change" as the object, so usually "high".

    Returns
    -------
    mask : np.ndarray bool
        Binary mask.
    thr : float
        Otsu threshold computed on finite values.
    """
    if img.ndim != 2:
        raise ValueError("Image must be 2D")

    # valid values to estimate the threshold
    valid = img[np.isfinite(img)]
    if valid.size == 0:
        # all NaN: no usable threshold
        raise ValueError("Cannot compute Otsu threshold: the image contains only NaN values.")
    
    vmin, vmax = float(valid.min()), float(valid.max())
    if vmin == vmax:
        # degenerate case: all identical -> threshold = that value
        thr = vmin
    else:
        thr = k * float(threshold_otsu(valid))

    if foreground == "high":
        mask = img >= thr
    elif foreground == "low":
        mask = img <= thr
    else:
        raise ValueError("foreground must be 'high' or 'low'")

    if nan_as_bg:
        mask = np.where(np.isfinite(img), mask, False)  # np.where(condition, value_if_true, value_if_false)
        # more functional (steps more readable, debugging easier) than an in-place transform like np.putmask(mask, ~np.isfinite(img), False)
        # if nan_as_bg is False, then nodata (a non-zero value) becomes True so it is considered as object

    return mask.astype(bool), thr


def apply_otsu_to_tiles(tiles: np.ndarray, k: float = 1.0, *, nan_as_bg: bool = True, foreground: str = "high"):
    """
    Apply otsu_tile on each tile of an array (n, n, tile_h, tile_w)

    Returns
    -------
    masks : (n, n, tile_h, tile_w) bool: Binary mask for each tile
    thr_grid : (n, n) float32: Threshold per tile
    """
    if tiles.ndim != 4:
        raise ValueError("tiles must have shape (n_rows, n_cols, tile_h, tile_w)")

    n_r, n_c, th, tw = tiles.shape
    masks = np.empty_like(tiles, dtype=bool)
    thr_grid = np.empty((n_r, n_c), dtype=np.float32)

    for i in range(n_r):
        for j in range(n_c):
            m, thr = otsu_tile(tiles[i, j], k, nan_as_bg=nan_as_bg, foreground=foreground)
            masks[i, j] = m
            thr_grid[i, j] = thr

    return masks, thr_grid



########## REASSEMBLY ##########

def assemble_tiles_to_image(masks_4d: np.ndarray, meta: dict | None = None) -> np.ndarray:  # meta: dict | None = None, this notation is possible since python 3.11
    """
    Stitches a 4D array (n_rows, n_cols, tile_h, tile_w) back into a single 2D image.

    If `meta` is provided (with pad_bottom and pad_right), the padding added during tiling is removed.

    Parameters
    ----------
    masks_4d : np.ndarray bool
        One mask (2D array) per tile, shape: (n_rows, n_cols, tile_h, tile_w)
    meta : dict, optional
        Dictionary returned by tile_image_2d, dict(tile_height=tile_height, tile_width=tile_width,
        pad_bottom=pad_bottom, pad_right=pad_right)

    Returns
    -------
    mask_2d : np.ndarray bool
        Global 2D mask (without padding if meta is provided)
    """
    if masks_4d.ndim != 4:
        raise ValueError("masks_4d must have shape (n_rows, n_cols, tile_h, tile_w)")

    n_r, n_c, tile_h, tile_w = masks_4d.shape

    # Put tiles back in original order so they can be merged with reshape
    mask_padded = masks_4d.transpose(0, 2, 1, 3).reshape(n_r * tile_h, n_c * tile_w)

    # Remove padding if meta is provided
    if meta is not None:
        pad_bottom = int(meta.get("pad_bottom", 0))  # meta.get("pad_bottom", 0) reads the value if it exists, else 0 by default (no padding)
        pad_right = int(meta.get("pad_right", 0))    # same here
        if pad_bottom or pad_right:  # If at least one of the two is non-zero, we crop
            mask_padded = mask_padded[: mask_padded.shape[0] - pad_bottom,
                                      : mask_padded.shape[1] - pad_right]

    return mask_padded.astype(bool)


########## FILTERING ##########

def filter_dense_regions(mask: np.ndarray,
                          win_size: int = 30,
                          d: float = 0.5,
                          min_area: int = 4000, closings: bool = False) -> np.ndarray:
    """
    Filters a binary mask to keep only the zones:
      - located in a locally dense white region
      - with a sufficient size (minimum area)

    Parameters
    ----------
    mask : np.ndarray bool 0/1
        Binary image (True/1 = white).
    win_size : int
        Size of the side of the square window for local density computation (in pixels), if win_size even a convention is used to center the pixel in the window.
    d : float
        Minimum density of white in the window (0-1).
    min_area : int
        Minimum area (in pixels) for kept connected components.
    closings : bool
        If True, keep only pixels both dense AND originally white.

    Returns
    -------
    filt : np.ndarray bool
        Filtered binary mask.
    """

    # Ensure we have a float 0/1 array
    mask_float = mask.astype(float)

    # 1) Local density of white in a window of size win_size
    # mode="nearest" avoids border artifacts by duplicating edge pixels
    local_mean = uniform_filter(mask_float, size=win_size, mode="nearest")
    dense_mask = local_mean >= d  # pixels inside a dense zone

    # Optionally restricts to pixels originally white
    if closings:
        core = dense_mask & (mask_float > 0.5)
    else: core = dense_mask

    # 2) Filter by connected-component size
    labels = label(core, connectivity=2)  # label() converts binary mask in connected components, 8-neighborhood (diagonals included)
    filt = np.zeros_like(labels, dtype=np.uint8)  # GeoTIFF does not support native boolean → convert to uint8

    for region in regionprops(labels):  # regionprops analyses each connected component of label
        if region.area >= min_area:     # region.area = number of pixels in the connected component
            filt[labels == region.label] = True  # region.label = ID of the region

    filt = filt.astype(int)
    
    return filt



########## MAIN_dtod_test ##########

def main_dtod_test(path_img1: str, path_img2: str, n: int, k: float = 1.0, closings: bool = False, p: int = 30, d: float = 0.5, a: int = 500, out_path: str | None = None) -> tuple[dict, np.ndarray]:
    """
    ...
    If out_path is not None, writes the raster at the given path out_path
    """

    # ==== bands loading ====

    with rasterio.open(path_img1) as src1:
        img1 = src1.read()
        profile1 = src1.profile

    with rasterio.open(path_img2) as src2:
        img2 = src2.read()
        profile2 = src2.profile

    # ==== NaN handling, padding, clipping and normalization ====

    img1 = to_nan(img1)
    img2 = to_nan(img2)
    img1, img2 = align_by_padding(img1, profile1, img2, profile2)

    img1 = normalize_image(clip_percentiles(img1))
    img2 = normalize_image(clip_percentiles(img2))

    # ==== dissimilarity computation ====

    # dist = dissimilarity(img1, img2)
    dist = smart_dissimilarity(img1, img2)     ## added
    # dist = dissimilarity(img1, img2)

    # ==== tiling, thresholding and reassembly ====

    tiles, meta = tile_image_2d(dist, n)

    masks, thr_grid = apply_otsu_to_tiles(tiles, k)

    final = assemble_tiles_to_image(masks, meta)

    # ==== filtering ====

    filt = filter_dense_regions(final, p, d, a, closings)   
    # filt = closing(filt, footprint=disk(20))                  ## all filters except dense added
    # seed = erosion(filt, footprint=disk(20)) 
    # filt = reconstruction(seed, filt, method='dilation').astype(bool)
    # filt = opening(filt, footprint=disk(5))
    # filt = remove_small_objects(final, min_size=500)


    # We return to the initial size so that it matches the affine transform of the profile, which would be hard to update correctly; and we update the rest of the profile

    H1 = profile1["height"]
    W1 = profile1["width"]

    filt = filt[:H1, :W1]

    profile = profile1.copy()
    profile.update(dtype="uint8", nodata=0, count=1)   # update only what changed in the profile

    # ==== writing ====

    if out_path is not None:
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(filt.astype("uint8") * 255, 1)  # ensure we don't write a tif with boolean values directly

    return profile, filt
