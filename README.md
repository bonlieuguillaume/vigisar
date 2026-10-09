# Sentinel-1 Change Detection Toolbox

This toolbox is designed for unsupervised deforestation detection using Sentinel-1 SAR imagery. It provides a pipeline processing orthorectified SAR under geotiff format into binary change masks. Any source of co-registered pre/post GeoTIFFs works; the **rosarium** repository is one that produces them from raw Sentinel-1 products (see *Input data* below).

---

## 📢 PROJECT STATUS & UPDATES
> **[IMPORTANT]** This project is under development. **Updates are incoming, particularly regarding the documentation. Until then, a power-point (in French) presenting the tool is already availbale in /assets/doc.**

---

## 🖼️ Workflow Illustration

| Ground Truth (Reference) | Result Mask (Prediction) |
|:---:|:---:|
| ![Truth](assets/display/paraguay_truth_mask_ex.png) | ![Result](assets/display/paraguay_result_mask_display_ex.png) |

*Example of clear deforestation between June and December in Gran Chaco, Paraguay.*

*Nicer visuals incoming: This space is reserved for the upcoming high-resolution workflow diagram.*

---

## 🚀 Very quick review of the main 

The toolbox is divided into two main modules:

### 1. Detection Pipeline (`main_dtod_test.py`)
The `main()` function handles the core logic:
- **Preprocessing:** Normalization and clipping of input rasters.
- **Dissimilarity:** Computing the difference between two dates (SAR intensity changes).
- **Tiling & Otsu:** Splitting the image into tiles to apply adaptive thresholding, allowing for local variations in backscatter.
- **Filtering:** Post-processing to remove noise and small objects using morphological operations and a handcrafted density filter.

### 2. Evaluation (`ground_truth.py`)
This module allows you to validate your results against reference data so as to test the pipeline performances:
- **Rasterization:** Convert polygon shapefiles into binary masks aligned with your imagery.
- **Metrics:** Compute statistical performance indicators including **F1-Score**, **MCC** (Matthews Correlation Coefficient), and **Kappa**.

---

## 🛰️ Input data

The detection takes two co-registered GeoTIFFs, one before and one after the event, **whatever produced them**: gamma0, coherence, or both, from SNAP, another toolbox or a provider. What it relies on:

- **the same layout in both files** — same bands in the same order, same CRS, same data type, and even the same storage options (driver, compression, tiling, interleave): the two rasterio profiles are compared key by key and must match, apart from the size, where one row or column of difference is tolerated (padded, then cropped back). Two files from different sources may need a `gdal_translate` to a common layout first;
- **co-registration** — pre and post on the same grid, pixel for pixel: the pipeline does not resample;
- **nodata declared** in the file (`nodata` tag), or one of -9999, -32768, -3.4e38 — those pixels become NaN and are ignored; an undeclared 0 would be taken as data;
- **any number of bands, of any kind** — each band is clipped and normalised on its own, then the dissimilarity is the distance between the pre and post band vectors of each pixel. The scale (linear or dB) is not imposed, but it changes the distribution the thresholding sees, so keep the one the parameters were tuned on.

**From raw Sentinel-1 products**, the sister repository **rosarium** covers the chain upstream — search, download, SNAP preprocessing — and writes exactly that:

- `python rosarium.py pre_post backscatter_coherence` — four SLC (two pre, two post) → `<name>_pre.tif` / `<name>_post.tif` with bands `gamma0_VH`, `gamma0_VV`, `coh_VH`, `coh_VV`;
- `python rosarium.py pre_post backscatter` — two GRD (pre, post) → the same two files with the two `gamma0` bands only.

Its outputs are in linear scale, with nodata = 0 declared on every band, on a 10 m UTM grid snapped on multiples of 10 m (`alignToStandardGrid`). Products generated before that grid alignment was switched on sit a fraction of a pixel off newer ones — regenerate them before mixing the two.

The preprocessing used to live here (`src/preprocess/`, `vigisar_graphs/`, `utils/parallel_download.py`); it moved to rosarium, where its documentation is (`features/snap_gpt/README.md` for the graphs, the GPT memory flags and the band conventions).


