import subprocess
import os
import sys
import argparse
import xml.etree.ElementTree as ET
from typing import Optional

try:
    from .find_swaths_and_bursts import find_subswath
except ImportError:
    from find_swaths_and_bursts import find_subswath  # type: ignore[no-redef]

DEFAULT_GPT = r"C:\Program Files\esa-snap\bin\gpt.exe"

_PROJECT_ROOT     = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_GRAPHS_DIR       = os.path.join(_PROJECT_ROOT, "vigisar_graphs")
_PREPROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "preprocessed")
_TEMP_DIR         = os.path.join(_PREPROCESSED_DIR, "temp")

_GRAPH_BACKSCATTER     = os.path.join(_GRAPHS_DIR, "backscatter.xml")
_GRAPH_COHERENCE       = os.path.join(_GRAPHS_DIR, "coherence.xml")
_GRAPH_GATHERING       = os.path.join(_GRAPHS_DIR, "gathering.xml")
_GRAPH_BACKSCATTER_GRD = os.path.join(_GRAPHS_DIR, "backscatter_grd.xml")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_band_name(collocate_name: str) -> str:
    """Derive a short generic name from a Collocate output band name.

    Examples:
        Gamma0_IW2_VH_mst_17Aug2017_M  →  gamma0_VH
        coh_IW2_VV_05Aug2017_17Aug2017_S0  →  coh_VV
    """
    pol = "VH" if "_VH_" in collocate_name else ("VV" if "_VV_" in collocate_name else "")
    if collocate_name.startswith("Gamma0"):
        return f"gamma0_{pol}" if pol else "gamma0"
    if collocate_name.startswith("coh"):
        return f"coh_{pol}" if pol else "coh"
    return collocate_name


def _clean_geotiff(path: str, band_names: list[str]) -> None:
    """Remove extra flag bands added by SNAP's Collocate (collocationFlags) and
    rename the remaining bands.

    SNAP's Write operator appends flag bands after data bands regardless of the
    BandSelect node.  We keep only the first len(band_names) bands and discard
    the rest, then set the band descriptions in-place.
    """
    try:
        from osgeo import gdal
    except ImportError:
        print("Warning: osgeo.gdal not available — band cleanup skipped.", file=sys.stderr)
        return

    gdal.UseExceptions()
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    ds = gdal.Open(path)
    gdal.PopErrorHandler()
    if ds is None:
        print(f"Warning: could not open {path}.", file=sys.stderr)
        return

    n_expected = len(band_names)
    n_actual   = ds.RasterCount
    ds = None

    if n_actual > n_expected:
        tmp = path + ".tmp.tif"
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        gdal.Translate(tmp, path, bandList=list(range(1, n_expected + 1)))
        gdal.PopErrorHandler()
        os.replace(tmp, path)

    gdal.PushErrorHandler("CPLQuietErrorHandler")
    ds = gdal.Open(path, gdal.GA_Update)
    gdal.PopErrorHandler()
    if ds is None:
        return
    for i, name in enumerate(band_names, 1):
        ds.GetRasterBand(i).SetDescription(name)
    ds = None


def _add_swath_suffix(path: str, swath: str) -> str:
    """Insert _IW1 / _IW2 / _IW3 before the file extension."""
    base, ext = os.path.splitext(path)
    return f"{base}_{swath}{ext}"


def _read_dimap_band_names(dim_path: str) -> list[str]:
    root = ET.parse(dim_path).getroot()
    return [el.text for el in root.findall(".//Spectral_Band_Info/BAND_NAME")]


def _resolve_gathering_bands(
    input_backscatter: str,
    input_coh_pre: str,
    input_coh_post: str,
) -> tuple[list[str], list[str]]:
    """
    Derive which Collocate output bands belong to the pre-event and post-event
    products by reading band names from the three BEAM-DIMAP inputs.

    Collocate suffix convention (must match the gathering.xml sources order):
        input_backscatter → reference → ``_M``
        input_coh_pre     → first secondary → ``_S0``
        input_coh_post    → second secondary → ``_S1``

    SNAP's CreateStack always writes master bands before slave bands.
    Since run_backscatter is called with input1=pre2 (master) and input2=post1
    (slave), the first half of backscatter bands is always pre2 and the second
    half is always post1 — no date parsing required.

    Returns:
        (bands_pre, bands_post): lists of band names as they appear after
        Collocate, ready to be passed as comma-separated sourceBands parameters.

    Raises:
        ValueError: If the backscatter product does not contain an even number
            of bands (would indicate something other than one master + one slave).
    """
    coh_pre_bands  = _read_dimap_band_names(input_coh_pre)
    coh_post_bands = _read_dimap_band_names(input_coh_post)
    bs_bands       = _read_dimap_band_names(input_backscatter)

    n = len(bs_bands)
    if n % 2 != 0:
        raise ValueError(
            f"Expected an even number of bands in the backscatter product "
            f"(one master image + one slave image), got {n} in {input_backscatter}"
        )
    bs_pre  = bs_bands[:n // 2]   # master (pre2) — always listed first by SNAP
    bs_post = bs_bands[n // 2:]   # slave  (post1) — always listed second

    bands_pre  = [f"{b}_M"  for b in bs_pre]  + [f"{b}_S0" for b in coh_pre_bands]
    bands_post = [f"{b}_M"  for b in bs_post] + [f"{b}_S1" for b in coh_post_bands]

    return bands_pre, bands_post


def _run_gpt(gpt_path: str, graph_xml: str, params: dict) -> Optional[str]:
    """
    Internal helper: build a GPT command from a parameter dict and execute it.

    Parameters are passed as ``-Pkey=value`` flags, substituting ``${key}``
    placeholders in the XML graph.  Returns stdout on success, None on failure.

    Raises:
        FileNotFoundError: If gpt_path does not exist on the filesystem.
    """
    if not os.path.exists(gpt_path):
        raise FileNotFoundError(f"GPT executable not found at: {gpt_path}")

    # -e  → full Java stack trace on error
    # -c  → tile cache size (GPT ignores snap.properties)
    # -q  → worker thread count (GPT ignores snap.properties)
    command = [
        gpt_path,
        graph_xml,
        "-e",
        "-c", "16384M",
        "-q", "16",
    ]
    for key, value in params.items():
        command.append(f"-P{key}={value}")

    print(f"Running graph: {os.path.basename(graph_xml)}")
    try:
        process = subprocess.run(command, check=True, text=True)
        print("Processing completed successfully.")
        return process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error during graph execution:\n{e.stderr}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backscatter(
    input1: str,
    input2: str,
    aoi: str,
    output: Optional[str] = None,
    gpt_path: str = DEFAULT_GPT,
) -> list[Optional[str]]:
    """
    Run the backscatter graph on two Sentinel-1 SLC products.

    Processing chain:
        Apply-Orbit-File → TOPSAR-Split → ThermalNoiseRemoval → Calibration
        → TOPSAR-Deburst → CreateStack → Cross-Correlation → Warp
        → Speckle-Filter → Terrain-Correction → Subset → Write

    The subswath and burst range are determined automatically from ``aoi`` using
    ``find_subswath``.  If the AOI spans multiple subswaths the graph is run
    once per subswath and the outputs are suffixed with the subswath name
    (e.g. ``backscatter_IW2.dim``).

    Args:
        input1 (str): Path to the master Sentinel-1 SLC product (.zip or .SAFE).
            Should be the pre2 image so that the master bands appear first in the
            output stack (required by ``run_gathering``).
        input2 (str): Path to the secondary Sentinel-1 SLC product (.zip or .SAFE).
            Should be the post1 image.
        aoi (str): Area of interest as a WKT polygon in WGS84.  Used both to
            locate the correct subswath/burst range and to spatially clip the
            Terrain-Correction output via the Subset node.
        output (str, optional): Full path for the output product (.dim).
            Defaults to ``data/preprocessed/temp/backscatter.dim``.
            If multiple subswaths are found, the subswath name is inserted before
            the extension (e.g. ``backscatter_IW2.dim``).
        gpt_path (str): Absolute path to the SNAP GPT executable.

    Returns:
        list: One entry per processed subswath (stdout string or None on error).

    Raises:
        FileNotFoundError: If gpt_path does not exist.
        ValueError: If the AOI does not intersect any subswath in input1.
    """
    swaths = find_subswath(input1, aoi)
    if not swaths:
        raise ValueError(f"The AOI does not intersect any subswath in {input1}")

    base_path = output if output is not None else os.path.join(_TEMP_DIR, "backscatter.dim")
    os.makedirs(os.path.dirname(os.path.abspath(base_path)), exist_ok=True)

    results = []
    for swath in swaths:
        out = _add_swath_suffix(base_path, swath["subswath"]) if len(swaths) > 1 else base_path
        params = {
            "input1":      input1,
            "input2":      input2,
            "output":      out,
            "aoi":         aoi,
            "subswath":    swath["subswath"],
            "first_burst": str(swath["first_burst"]),
            "last_burst":  str(swath["last_burst"]),
        }
        results.append(_run_gpt(gpt_path, _GRAPH_BACKSCATTER, params))

    return results


def run_coherence(
    input1: str,
    input2: str,
    aoi: str,
    pair: Optional[str] = None,
    output: Optional[str] = None,
    gpt_path: str = DEFAULT_GPT,
) -> list[Optional[str]]:
    """
    Run the coherence graph on two Sentinel-1 SLC products.

    Processing chain:
        Apply-Orbit-File → TOPSAR-Split → Back-Geocoding
        → Enhanced-Spectral-Diversity → Coherence → TOPSAR-Deburst
        → Terrain-Correction → Subset → Write

    The subswath and burst range are determined automatically from ``aoi``.
    If the AOI spans multiple subswaths the graph is run once per subswath
    and the subswath name is inserted before the file extension
    (e.g. ``coherence_pre_IW1.dim``, ``coherence_pre_IW2.dim``).

    Two output modes depending on ``pair``:

    * **No pair** (standalone run): output is written to
      ``data/preprocessed/default/coh.dim``, or to ``output`` if given as a
      full path.  Useful for single-date coherence products or exploratory runs.

    * **With pair** (``"pre"`` or ``"post"``): output is written to
      ``data/preprocessed/temp/coherence_{pair}.dim``.  Use this mode when
      running the full pipeline (backscatter + coherence pre + coherence post +
      gathering) so that ``run_gathering`` can locate the files automatically.

    Args:
        input1 (str): Path to the master Sentinel-1 SLC product (.zip or .SAFE).
        input2 (str): Path to the secondary Sentinel-1 SLC product (.zip or .SAFE).
        aoi (str): Area of interest as a WKT polygon in WGS84.
        pair (str, optional): ``"pre"`` or ``"post"``.  When given, the output is
            placed in the temp folder with a ``_pre`` / ``_post`` suffix so that
            ``run_gathering`` can find it.  When omitted, the output goes to
            ``data/preprocessed/default/coh.dim`` (or the path given in
            ``output``).
        output (str, optional):
            * If ``pair`` is given: base name used in the temp folder filename,
              e.g. ``"zta1"`` → ``data/preprocessed/temp/zta1_pre.dim``.
              Defaults to ``"coherence"``.
            * If ``pair`` is not given: full path to the output file.
              Defaults to ``data/preprocessed/default/coh.dim``.
        gpt_path (str): Absolute path to the SNAP GPT executable.

    Returns:
        list: One entry per processed subswath (stdout string or None on error).

    Raises:
        FileNotFoundError: If gpt_path does not exist.
        ValueError: If pair is not ``"pre"``, ``"post"``, or None.
        ValueError: If the AOI does not intersect any subswath in input1.
    """
    if pair is not None and pair not in ("pre", "post"):
        raise ValueError(f"pair must be 'pre', 'post', or None, got {pair!r}")

    swaths = find_subswath(input1, aoi)
    if not swaths:
        raise ValueError(f"The AOI does not intersect any subswath in {input1}")

    if pair is None:
        if output is None:
            folder    = os.path.join(_PREPROCESSED_DIR, "default")
            base_path = os.path.join(folder, "coherence.dim")
        else:
            base_path = output
            folder    = os.path.dirname(os.path.abspath(base_path))
    else:
        base_name = output if output is not None else "coherence"
        folder    = _TEMP_DIR
        base_path = os.path.join(folder, f"{base_name}_{pair}.dim")

    os.makedirs(folder, exist_ok=True)

    results = []
    for swath in swaths:
        out = _add_swath_suffix(base_path, swath["subswath"]) if len(swaths) > 1 else base_path
        params = {
            "input1":      input1,
            "input2":      input2,
            "output":      out,
            "aoi":         aoi,
            "subswath":    swath["subswath"],
            "first_burst": str(swath["first_burst"]),
            "last_burst":  str(swath["last_burst"]),
        }
        results.append(_run_gpt(gpt_path, _GRAPH_COHERENCE, params))

    return results


def run_gathering(
    input_backscatter: str,
    input_coh_pre: str,
    input_coh_post: str,
    output: Optional[str] = None,
    gpt_path: str = DEFAULT_GPT,
) -> list[str]:
    """
    Run the gathering graph to collocate a backscatter product with two
    coherence stacks and split the result into pre-event and post-event products.

    Processing chain:
        3× Read → Collocate → BandSelect → Write (pre)
                           → BandSelect → Write (post)

    Band assignment is derived automatically from the BEAM-DIMAP band names
    of the three inputs — no XML editing required.

    Args:
        input_backscatter (str): Path to the backscatter stack (.dim), output of
            ``run_backscatter``.  input1=pre2 and input2=post1 must have been
            respected when running the backscatter graph so that master bands
            (pre2) appear first in the stack.
        input_coh_pre (str): Path to the pre-event coherence product (.dim),
            output of ``run_coherence`` with ``pair="pre"``.
        input_coh_post (str): Path to the post-event coherence product (.dim),
            output of ``run_coherence`` with ``pair="post"``.
        output (str, optional): Name for this processing run.  A subfolder with
            that name is created under ``data/preprocessed/`` and the two output
            GeoTIFFs are written there as ``<name>_pre.tif`` and
            ``<name>_post.tif``.  If None, outputs go to
            ``data/preprocessed/default/`` as ``pre.tif`` and ``post.tif``.
        gpt_path (str): Absolute path to the SNAP GPT executable.

    Returns:
        list[str]: Paths of the GeoTIFF files that were successfully written
            (``[pre.tif, post.tif]``).  Empty if GPT failed.

    Raises:
        FileNotFoundError: If gpt_path does not exist.
    """
    if output is None:
        folder      = os.path.join(_PREPROCESSED_DIR, "default")
        output_pre  = os.path.join(folder, "pre")
        output_post = os.path.join(folder, "post")
    elif os.sep in output or "/" in output:
        # Full path prefix — caller controls the directory (used by main_preprocess
        # in multi-swath mode to keep all swath files in the same folder).
        folder      = os.path.dirname(os.path.abspath(output))
        stem        = os.path.basename(output)
        output_pre  = os.path.join(folder, f"{stem}_pre")
        output_post = os.path.join(folder, f"{stem}_post")
    else:
        folder      = os.path.join(_PREPROCESSED_DIR, output)
        output_pre  = os.path.join(folder, f"{output}_pre")
        output_post = os.path.join(folder, f"{output}_post")

    os.makedirs(folder, exist_ok=True)
    reference_name = os.path.splitext(os.path.basename(input_backscatter))[0]
    bands_pre, bands_post = _resolve_gathering_bands(
        input_backscatter, input_coh_pre, input_coh_post
    )
    params = {
        "input1":        input_backscatter,
        "input2":        input_coh_pre,
        "input3":        input_coh_post,
        "output_pre":    output_pre,
        "output_post":   output_post,
        "reference_name": reference_name,
        "bands_pre":     ",".join(bands_pre),
        "bands_post":    ",".join(bands_post),
    }
    _run_gpt(gpt_path, _GRAPH_GATHERING, params)

    produced = []
    clean_pre  = [_clean_band_name(b) for b in bands_pre]
    clean_post = [_clean_band_name(b) for b in bands_post]
    for tif, names in [
        (output_pre  + ".tif", clean_pre),
        (output_post + ".tif", clean_post),
    ]:
        if os.path.exists(tif):
            _clean_geotiff(tif, names)
            produced.append(tif)

    return produced


def _split_grd_stack(dim_path: str, pre_path: str, post_path: str) -> list[str]:
    """
    Split a coregistered GRD stack (BEAM-DIMAP) into two GeoTIFFs.

    SNAP's CreateStack writes master bands before slave bands.  Since
    run_backscatter_grd is always called with input1=pre (master) and
    input2=post (slave), the first half of bands is pre and the second half
    is post — no date parsing required.

    Band names are read from the DIMAP XML.  If the "_mst" / "_slv" suffixes
    are present they are used to identify each group; otherwise the bands are
    split by position (first half / second half).
    """
    try:
        from osgeo import gdal
    except ImportError:
        print("Warning: osgeo.gdal not available — cannot split GRD stack.", file=sys.stderr)
        return []

    gdal.UseExceptions()

    band_names = _read_dimap_band_names(dim_path)
    n = len(band_names)

    mst_idx = [i + 1 for i, name in enumerate(band_names) if "_mst" in name.lower()]
    slv_idx = [i + 1 for i, name in enumerate(band_names) if "_slv" in name.lower()]

    if not mst_idx or not slv_idx:
        if n % 2 != 0:
            raise ValueError(
                f"Cannot split GRD stack: odd number of bands ({n}) in {dim_path}"
            )
        half = n // 2
        mst_idx = list(range(1, half + 1))
        slv_idx = list(range(half + 1, n + 1))

    mst_names = [_clean_band_name(band_names[i - 1]) for i in mst_idx]
    slv_names = [_clean_band_name(band_names[i - 1]) for i in slv_idx]

    produced = []
    for path, indices, names in [
        (pre_path,  mst_idx, mst_names),
        (post_path, slv_idx, slv_names),
    ]:
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        gdal.Translate(path, dim_path, bandList=indices, format="GTiff")
        gdal.PopErrorHandler()
        if os.path.exists(path):
            _clean_geotiff(path, names)
            produced.append(path)

    return produced


def run_backscatter_grd(
    pre: str,
    post: str,
    aoi: str,
    output: Optional[str] = None,
    gpt_path: str = DEFAULT_GPT,
) -> list[str]:
    """
    Run the GRD backscatter graph on two Sentinel-1 GRD products and write
    two separate GeoTIFFs (pre-event and post-event).

    Processing chain (single graph, both images together):
        Apply-Orbit-File → ThermalNoiseRemoval → Remove-GRD-Border-Noise
        → Calibration (×2) → CreateStack → Cross-Correlation → Warp
        → Speckle-Filter → Terrain-Correction → Subset → Write

    The two images are coregistered via Cross-Correlation + Warp so that they
    lie on the same pixel grid.  The output BEAM-DIMAP stack is then split
    into two GeoTIFFs: the master bands (pre) and the slave bands (post).

    Unlike the SLC pipeline there is no subswath/burst splitting — GRD products
    already cover the full swath and do not require TOPSAR-Split.

    Args:
        pre (str): Path to the pre-event Sentinel-1 GRD product (.zip or .SAFE).
            Used as the master image (reference for coregistration).
        post (str): Path to the post-event Sentinel-1 GRD product (.zip or .SAFE).
            Used as the slave image.
        aoi (str): Area of interest as a WKT polygon in WGS84.  Used to clip
            the output after terrain correction.
        output (str, optional): Controls where the two output GeoTIFFs are written.
            Follows the same convention as ``run_gathering``:

            * ``None``          → ``data/preprocessed/default/pre.tif`` and ``post.tif``
            * Simple name       → ``data/preprocessed/<name>/<name>_pre.tif`` and ``_post.tif``
            * Full path prefix  → ``<prefix>_pre.tif`` and ``<prefix>_post.tif``
              (the folder must already exist or will be created)
        gpt_path (str): Absolute path to the SNAP GPT executable.

    Returns:
        list[str]: Paths of the two GeoTIFFs that were successfully written
            (``[pre.tif, post.tif]``).  Empty if GPT failed.

    Raises:
        FileNotFoundError: If gpt_path does not exist.
    """
    if output is None:
        folder      = os.path.join(_PREPROCESSED_DIR, "default")
        output_pre  = os.path.join(folder, "pre.tif")
        output_post = os.path.join(folder, "post.tif")
    elif os.sep in output or "/" in output:
        folder      = os.path.dirname(os.path.abspath(output))
        stem        = os.path.basename(output)
        output_pre  = os.path.join(folder, f"{stem}_pre.tif")
        output_post = os.path.join(folder, f"{stem}_post.tif")
    else:
        folder      = os.path.join(_PREPROCESSED_DIR, output)
        output_pre  = os.path.join(folder, f"{output}_pre.tif")
        output_post = os.path.join(folder, f"{output}_post.tif")

    os.makedirs(folder, exist_ok=True)

    tmp_dim = os.path.join(_TEMP_DIR, "backscatter_grd.dim")
    os.makedirs(_TEMP_DIR, exist_ok=True)

    _run_gpt(gpt_path, _GRAPH_BACKSCATTER_GRD, {
        "input1": pre,
        "input2": post,
        "aoi":    aoi,
        "output": tmp_dim,
    })

    if not os.path.exists(tmp_dim):
        return []

    return _split_grd_stack(tmp_dim, output_pre, output_post)


def run_mosaic(inputs: list[str], output: str) -> str:
    """
    Merge a list of co-registered GeoTIFFs (one per subswath) into a single file.

    Intended for use after ``run_gathering`` when the AOI spans multiple subswaths:
    pass the per-swath pre (or post) GeoTIFFs and receive a single mosaicked file.

    If only one input is given the file is copied as-is (no Warp needed).

    Args:
        inputs (list[str]): Ordered list of GeoTIFF paths to mosaic.
        output (str): Output GeoTIFF path.

    Returns:
        str: Path of the written output file (same as ``output``).

    Raises:
        RuntimeError: If GDAL fails to build the mosaic.
        ImportError: If osgeo.gdal is not available.
    """
    if not inputs:
        raise ValueError("inputs must not be empty")

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)

    if len(inputs) == 1:
        import shutil
        shutil.copy2(inputs[0], output)
        return output

    try:
        from osgeo import gdal
    except ImportError:
        raise ImportError("osgeo.gdal is required for mosaicking")

    gdal.UseExceptions()
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    ds = gdal.Warp(output, inputs, format="GTiff", resampleAlg="near")
    gdal.PopErrorHandler()

    if ds is None:
        raise RuntimeError(f"Mosaic failed → {output!r}")
    ds = None
    return output


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SNAP GPT preprocessing graphs for the Vigisar pipeline.\n\n"
            "Subcommands (SLC pipeline):\n"
            "  backscatter      — Gamma0 backscatter via cross-correlation coregistration\n"
            "  coherence        — coherence via ESD coregistration\n"
            "  gathering        — collocate backscatter + coherence into pre/post products\n\n"
            "Subcommands (GRD pipeline):\n"
            "  backscatter-grd  — Gamma0 backscatter from two GRD products (no subswath split)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--gpt",
        default=DEFAULT_GPT,
        metavar="PATH",
        help=f"[optional] Path to the SNAP GPT executable (default: {DEFAULT_GPT!r})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- backscatter ---------------------------------------------------------
    p_bs = subparsers.add_parser(
        "backscatter",
        parents=[common],
        help="Gamma0 backscatter stack via cross-correlation coregistration",
        description=(
            "Process two Sentinel-1 SLC acquisitions through orbit correction, "
            "TOPSAR split, thermal noise removal, radiometric calibration, deburst, "
            "cross-correlation coregistration, speckle filtering, and terrain correction.\n\n"
            "The subswath and burst range are determined automatically from --aoi.\n"
            "If the AOI spans multiple subswaths, the graph runs once per subswath\n"
            "and each output is suffixed with the subswath name (e.g. backscatter_IW2.dim).\n\n"
            "Pass input1=pre2 and input2=post1 so that the master bands (pre2) appear\n"
            "first in the output stack, as required by the gathering step."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_bs.add_argument("--input1", required=True, metavar="PATH",
                      help="[required] Master SLC product — should be pre2 (.zip or .SAFE)")
    p_bs.add_argument("--input2", required=True, metavar="PATH",
                      help="[required] Secondary SLC product — should be post1 (.zip or .SAFE)")
    p_bs.add_argument("--aoi", required=True, metavar="WKT",
                      help=(
                          "[required] Area of interest as a WKT polygon in WGS84.  Used to locate "
                          'the correct subswath/burst range and to clip the output.  '
                          'Must be quoted: --aoi "POLYGON ((-54.1 4.1, ...))"'
                      ))
    p_bs.add_argument("--output", default=None, metavar="PATH",
                      help=(
                          "[optional] Output product path (.dim).  "
                          "Defaults to data/preprocessed/temp/backscatter[_IWx].dim."
                      ))

    # -- coherence -----------------------------------------------------------
    p_coh = subparsers.add_parser(
        "coherence",
        parents=[common],
        help="Interferometric coherence via ESD coregistration",
        description=(
            "Process two Sentinel-1 SLC acquisitions through orbit correction, "
            "TOPSAR split, back-geocoding, Enhanced Spectral Diversity, coherence "
            "estimation, deburst, terrain correction, and spatial clipping.\n\n"
            "The subswath and burst range are determined automatically from --aoi.\n"
            "If the AOI spans multiple subswaths the graph runs once per subswath\n"
            "and each output is suffixed with its name (e.g. coherence_pre_IW2.dim).\n\n"
            "Two output modes:\n"
            "  No --pair : standalone run → data/preprocessed/default/coh[_IWx].dim\n"
            "              (or the path given with --output)\n"
            "  --pair pre/post : pipeline run → data/preprocessed/temp/coherence_pre[_IWx].dim\n"
            "              Use this mode when the output will be fed into gathering."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_coh.add_argument("--input1", required=True, metavar="PATH",
                       help="[required] Master SLC product (.zip or .SAFE)")
    p_coh.add_argument("--input2", required=True, metavar="PATH",
                       help="[required] Secondary SLC product (.zip or .SAFE)")
    p_coh.add_argument("--aoi", required=True, metavar="WKT",
                       help=(
                           "[required] Area of interest as a WKT polygon in WGS84.  "
                           'Must be quoted: --aoi "POLYGON ((-54.1 4.1, ...))"'
                       ))
    p_coh.add_argument("--pair", default=None, choices=["pre", "post"],
                       help=(
                           "[optional] Event period: 'pre' (pre1+pre2) or 'post' (post1+post2).  "
                           "When given, the output goes to data/preprocessed/temp/ with "
                           "a _pre/_post suffix so that gathering can locate it.  "
                           "Omit for a standalone coherence run."
                       ))
    p_coh.add_argument("--output", default=None, metavar="NAME_OR_PATH",
                       help=(
                           "[optional] With --pair: base name in the temp folder "
                           "(e.g. 'zta1' → temp/zta1_pre.dim). "
                           "Without --pair: full output path "
                           "(default: data/preprocessed/default/coherence.dim). "
                           "In both modes, if the AOI spans multiple subswaths the "
                           "subswath name is inserted before the extension "
                           "(e.g. coherence_IW1.dim, coherence_IW2.dim)."
                       ))

    # -- gathering -----------------------------------------------------------
    p_ga = subparsers.add_parser(
        "gathering",
        parents=[common],
        help="Collocate backscatter and coherence stacks into pre/post GeoTIFFs",
        description=(
            "Collocate a backscatter stack with a pre-event and a post-event "
            "coherence product, then split the result into two GeoTIFF files.\n\n"
            "Intended for use after running backscatter + coherence --pair pre + "
            "coherence --pair post on the same AOI.  Band assignment is derived "
            "automatically from the BEAM-DIMAP band names."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_ga.add_argument("--input-backscatter", required=True, metavar="PATH",
                      help="[required] Backscatter stack (.dim), output of the backscatter graph")
    p_ga.add_argument("--input-coh-pre", required=True, metavar="PATH",
                      help="[required] Pre-event coherence (.dim), output of coherence --pair pre")
    p_ga.add_argument("--input-coh-post", required=True, metavar="PATH",
                      help="[required] Post-event coherence (.dim), output of coherence --pair post")
    p_ga.add_argument("--output", default=None, metavar="NAME",
                      help=(
                          "[optional] Run name.  Creates data/preprocessed/<name>/ and writes "
                          "<name>_pre.tif and <name>_post.tif inside it.  "
                          "Defaults to data/preprocessed/default/pre.tif and post.tif."
                      ))

    # -- backscatter-grd -----------------------------------------------------
    p_grd = subparsers.add_parser(
        "backscatter-grd",
        parents=[common],
        help="Gamma0 backscatter from two GRD products (no subswath split required)",
        description=(
            "Process two Sentinel-1 GRD acquisitions through orbit correction, thermal\n"
            "noise removal, border noise removal, radiometric calibration, cross-correlation\n"
            "coregistration, speckle filtering, terrain correction, and spatial clipping.\n\n"
            "Both images are processed together in a single graph run.  The output is split\n"
            "into two GeoTIFFs: <name>_pre.tif (master/pre image) and <name>_post.tif\n"
            "(slave/post image), each with Gamma0_VH and Gamma0_VV bands.\n\n"
            "Unlike the SLC pipeline there is no subswath/burst detection step — GRD products\n"
            "already cover the full swath."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_grd.add_argument("--pre",    required=True, metavar="PATH",
                       help="[required] Pre-event GRD product (.zip or .SAFE) — used as master")
    p_grd.add_argument("--post",   required=True, metavar="PATH",
                       help="[required] Post-event GRD product (.zip or .SAFE) — used as slave")
    p_grd.add_argument("--aoi",    required=True, metavar="WKT",
                       help=(
                           "[required] Area of interest as a WKT polygon in WGS84.  "
                           'Must be quoted: --aoi "POLYGON ((-54.1 4.1, ...))"'
                       ))
    p_grd.add_argument("--output", default=None, metavar="NAME",
                       help=(
                           "[optional] Run name.  Creates data/preprocessed/<name>/ and writes "
                           "<name>_pre.tif and <name>_post.tif inside it.  "
                           "If omitted, writes pre.tif and post.tif to data/preprocessed/default/."
                       ))

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "backscatter":
        run_backscatter(
            input1=args.input1,
            input2=args.input2,
            aoi=args.aoi,
            output=args.output,
            gpt_path=args.gpt,
        )
    elif args.command == "coherence":
        run_coherence(
            input1=args.input1,
            input2=args.input2,
            aoi=args.aoi,
            pair=args.pair,
            output=args.output,
            gpt_path=args.gpt,
        )
    elif args.command == "gathering":
        run_gathering(
            input_backscatter=args.input_backscatter,
            input_coh_pre=args.input_coh_pre,
            input_coh_post=args.input_coh_post,
            output=args.output,
            gpt_path=args.gpt,
        )
    elif args.command == "backscatter-grd":
        tifs = run_backscatter_grd(
            pre=args.pre,
            post=args.post,
            aoi=args.aoi,
            output=args.output,
            gpt_path=args.gpt,
        )
        for path in tifs:
            print(path)


if __name__ == "__main__":
    main()
