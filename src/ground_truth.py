"""
Change-detection toolbox (evaluating performances of raster → binary change mask).
Utility functions for handling ground-truth masks and evaluating change-detection results.

This module provides two main blocks of functionality:

1. **Shapefile → Raster Mask Conversion**
   - Converts a polygon shapefile into a binary mask (0/1) aligned to a reference raster.
   - Ensures CRS consistency between the shapefile and the raster: USE THE IMG1 RASTER OF THE MAIN0 AS THE REFERENCE RASTER
   - Rasterizes polygons using rasterio's `rasterize`.
   - Optionally saves the output mask as a GeoTIFF.
   - Used to generate ground-truth masks for validation of the change-detection tool.

2. **Evaluation Metrics for Binary Masks**
   - Computes confusion matrix components: TP, TN, FP, FN.
   - Computes MCC, F1-score, precision, recall, accuracy and Kappa coefficient.
   - Designed to compare a ground-truth mask with a predicted mask.
   - Includes shape-consistency checks and safety against numerical overflow.

These tools are intended to be used in testing and validation workflows, typically inside
a Jupyter notebook or as part of automated evaluation scripts for change-detection models.

"""


########## IMPORTS ##########

# common
import rasterio
import numpy as np
import matplotlib.pyplot as plt


# shapefile_to_mask
import geopandas as gpd
from rasterio.features import rasterize



########## SHAPEFILE TO MASK ##########

def shapefile_to_mask(shp_path: str, ref_raster_path: str, out_path: str | None = None) -> np.ndarray:
    """
    Rasterizes a polygon shapefile into a binary mask (0/1),
    aligned on a reference raster.

    - shp_path : path to the shapefile (polygons)
    - ref_raster_path : reference GeoTIFF (size, transform, crs etc)
    - out_path : if not None, saves the mask as GeoTIFF using out_path as output file path

    Returns: mask (np.ndarray 2D, dtype=uint8)
    """

    # 1) Read reference raster
    with rasterio.open(ref_raster_path) as src:
        ref_transform = src.transform
        ref_crs = src.crs
        out_shape = (src.height, src.width)
        profile = src.profile

    # 2) Read shapefile
    gdf = gpd.read_file(shp_path)

    if gdf.empty or gdf.geometry.isna().all() or gdf.geometry.is_empty.all():
        raise ValueError(
            "[ERROR] shapefile contains no valid geometry "
            "(0 entities, null or empty geometries)."
        )

    # 3) Ensure both CRS match

    if gdf.crs != ref_crs:
        raise ValueError(
            f"[ERROR] CRS of shapefile ({gdf.crs}) differs from raster CRS ({ref_crs}).\n"
            f"Please reproject your shapefile to ({ref_crs}) before continuing."
        )

    # if we want the code to reproject:
        # if gdf.crs != ref_crs:
        #     print(
        #     f"[WARNING] Le CRS du shapefile ({gdf.crs}) est différent du CRS du raster ({ref_crs}). "
        #     f"Reprojection automatique en {ref_crs}.")
        #     gdf = gdf.to_crs(ref_crs)

        
    # 4) Prepare geometries for rasterization
    # each polygon is “burned” with value 1, creation of a tuple [(polygone1, 1), (polygone2, 1), (polygone3, 1), ...] as expected by rasterize()
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]  # skip None geometries (corrupted or empty polygon)

    # 5) Rasterization -> mask 0/1
    mask = rasterize(shapes=shapes, out_shape=out_shape, transform=ref_transform, fill=0, dtype="uint8")

    # 6) Optional save as GeoTIFF
    if out_path is not None:

        profile.update(dtype="uint8", count=1, nodata=0)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask.astype("uint8") * 255, 1)  # multiplies by 255 for visibility, writing on band 1

    return mask



########## METRICS ##########

def confusion_from_masks(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int]:
    """
    Computes TP, TN, FP, FN between two binary masks (0/1 or bool).
    y_true : reference mask (ground truth)
    y_pred : predicted mask

    Shapes must match. If shapes differ, raise an error.
    Returns: tp, tn, fp, fn (integers)
    """

    # ---- Safety check: shapes must match ----

    # if y_pred was generated using shapefile_to_mask(shp_path: str, ref_raster_path = path_img1) with path_img1 the first image of the main
    # both masks should have the same size

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"[ERROR] Masks have different shapes: y_true{y_true.shape}, y_pred{y_pred.shape}."
        )

    yt = np.asarray(y_true).astype(bool)
    yp = np.asarray(y_pred).astype(bool)

    tp = np.logical_and(yt, yp).sum()
    tn = np.logical_and(~yt, ~yp).sum()
    fp = np.logical_and(~yt, yp).sum()
    fn = np.logical_and(yt, ~yp).sum()

    return tp, tn, fp, fn

def metrics_from_masks(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Returns a dict with accuracy, precision, recall, F1, MCC and Kappa.
    """

    tp, tn, fp, fn = confusion_from_masks(y_true, y_pred)
    tp, tn, fp, fn = map(float, (tp, tn, fp, fn))  # avoids int32 overflow, in the computation of denom for instance
    total = tp + tn + fp + fn

    # avoid division by zero
    eps = 1e-9

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = (tp + tn) / (total + eps)

    # Matthews Correlation Coefficient
    denom_mcc = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
    mcc = ((tp * tn) - (fp * fn)) / denom_mcc

    # Cohen's Kappa
    # po: observed agreement (identical to accuracy)
    # pe: expected agreement by chance
    po = accuracy
    pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (total**2 + eps)
    kappa = (po - pe) / (1 - pe + eps)

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "F1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy)
    }