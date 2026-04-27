# Sentinel-1 Change Detection Toolbox

This toolbox is designed for unsupervised deforestation detection using Sentinel-1 SAR imagery. It provides a pipeline processing orthorectified SAR under geotiff format (might be updated in the future to take in raw S1 SLC files as input) into binary change masks.

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


