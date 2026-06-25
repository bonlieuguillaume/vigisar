"""
Change-detection toolbox (raster → binary change mask).


This module aims at detecting change between two images taken at different dates. It is mostly designed to work with Sentinel-1 preprocessed data 
but the user is free to feed it with any raster he dreams of. It allows to :

  - load n+1 (1 pre, n post) coregistered rasters (that can contain multiple bands) with rasterio 
  - aggregate the information of the n post images to fine-tune the change detection between the two dates
    using the main_dtod_test function and an aggregation function that can be found in the README

The main entry point is `main_multidate_0(...)`, which returns the output raster profile
and the final binary mask (and can also save it to `out_path` if provided).

"""

########## IMPORTS ##########

# common
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Imports from src
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))  # add the project root directory to the PYTHONPATH
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import src.main_dtod_test as main_test

########## MAIN_multidate_test ##########

def main_multidate_test(path_img1: str, list_pathpost: list[str], n: int, k: float = 1.0, closings: bool = False, p: int = 30, d: float = 0.5, a: int = 3000, out_path_multidate : str | None = None) -> tuple[dict, np.ndarray]:
    """
    ...
    If out_path is not None, writes the raster at the given path out_path
    """
    
    diffs = []
    profile = main_test.main_dtod_test(path_img1, list_pathpost[0], n, k, closings, p, d, a, out_path=None)[0]

    for path_imgpost in list_pathpost:
        res1 = main_test.main_dtod_test(path_img1, path_imgpost, n, k, closings, p, d, a, out_path=None)
        res2 = main_test.main_dtod_test(list_pathpost[0], path_imgpost, n, k, closings, p, d, a, out_path=None)

        import matplotlib.pyplot as plt

    # --- BLOC DE VISUALISATION AJOUTÉ ---
        plt.figure(figsize=(6, 6))
        plt.imshow(res2[1], cmap="gray") # res2[1] est le ndarray du masque
        plt.title(f"Visualisation res2 pour : {os.path.basename(path_imgpost)}")
        plt.colorbar()
        plt.show()

        if res1[0] != profile or res2[0] != profile:
            raise ValueError(f"Profile mismatch detected!\n")
        
        diffs.append(np.abs(res1[1] - res2[1]))

        diff_visu = np.abs(res1[1] - res2[1]) # Calcul de l'écart
    
        plt.figure(figsize=(6, 6))
        plt.imshow(diff_visu, cmap="gray") # Affichage de la différence
        plt.title(f"Écart (abs) pour : {os.path.basename(path_imgpost)}")
        plt.colorbar(label="Intensité de la différence")
        plt.show()


    multidate = (1 / len(list_pathpost)) * sum(diffs)

    multidate = main_test.otsu_tile(multidate)[0]

    multidate.astype(int)

    if out_path_multidate is not None:
        with rasterio.open(out_path_multidate, "w", **profile) as dst:
            dst.write(multidate.astype("uint8") * 255, 1)  # ensure we don't write a tif with boolean values directly

    return profile, multidate