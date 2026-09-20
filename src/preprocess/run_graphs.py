import subprocess
import os
import sys
import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

try:
    from .polygon_to_swaths_bursts import get_intersecting_bursts, parse_polygon
except ImportError:
    from polygon_to_swaths_bursts import get_intersecting_bursts, parse_polygon  # type: ignore[no-redef]

DEFAULT_GPT = r"C:\Program Files\esa-snap\bin\gpt.exe"


# ---------------------------------------------------------------------------
# GPT memory / performance settings
# ---------------------------------------------------------------------------
# Every graph run goes through _run_gpt, which passes these four settings on
# the gpt command line.  A command-line flag always overrides what SNAP has in
# gpt.vmoptions and ~/.snap/etc/snap.properties, so what is set here (or given
# to the CLIs) is what actually runs — the SNAP GUI settings do not apply.
# `gpt --diag` prints the values a bare gpt would use.
#
# The defaults below are the ones a 32 GB / 8-core (16-thread) machine runs
# comfortably.  How to choose them for another machine:
#
#   xmx        Java heap ceiling (-Xmx).  Everything gpt holds — the tile
#              cache AND the working arrays of the operators (coregistration,
#              Back-Geocoding, ESD keep whole bursts and the DEM in memory) —
#              must fit under it.  About 2/3 of the physical RAM, leaving the
#              rest to the OS, Python and the GeoTIFF tools.  Too low: a Java
#              OutOfMemoryError on large AOIs.  Too high: the machine swaps,
#              and the JVM can die on a native allocation (hs_err_pid*.log).
#
#   cache      Tile cache (-c), lives INSIDE the heap.  Keeps computed tiles so
#              that downstream operators do not recompute them.  A ceiling, not
#              a need: too small only costs time (recomputation), it never
#              crashes — whereas a big cache always fills up and starves the
#              operators.  1/4 to 1/3 of the heap is plenty for these graphs.
#
#   threads    Tiles computed in parallel (-q).  Working memory of the tiled
#              operators grows with it.  Up to the number of hardware threads;
#              the number of physical cores is the sweet spot when memory is
#              tight (SNAP scales poorly beyond ~8 threads anyway).  Lowering
#              it is the second lever after the cache on very large AOIs.
#
#   tile_size  Edge of the square tiles, in pixels.  Keep a power of two
#              (256, 512, 1024): it matches the block size of the files on disk
#              and of the pyramid levels, so every tile maps to whole blocks.
#              512 is SNAP's default and there is rarely a reason to change it;
#              1024 lowers the per-tile overhead on big rasters at the cost of
#              more memory per thread.
#
# Neither cache nor threads can shrink what the coregistration operators hold
# for a given AOI: if a large AOI does not fit, lower the cache first, then the
# threads, and if it still fails the heap (hence the machine) is the limit.

DEFAULT_XMX       = "21G"
DEFAULT_CACHE     = "8192M"
DEFAULT_THREADS   = 16
DEFAULT_TILE_SIZE = 512


@dataclass
class GptOptions:
    """Memory / performance settings passed to every gpt call (see above).

    ``xmx`` and ``cache`` are Java size strings (``"21G"``, ``"8192M"``).
    """
    xmx: str = DEFAULT_XMX
    cache: str = DEFAULT_CACHE
    threads: int = DEFAULT_THREADS
    tile_size: int = DEFAULT_TILE_SIZE

    def to_args(self) -> list[str]:
        """The gpt command-line flags for these settings.

        ``-J<opt>`` hands the option to the JVM itself (heap, and system
        properties, which is how snap.properties keys are overridden);
        ``-c`` / ``-q`` are gpt's own flags.
        """
        return [
            f"-J-Xmx{self.xmx}",
            f"-J-Dsnap.jai.defaultTileSize={self.tile_size}",
            "-c", self.cache,
            "-q", str(self.threads),
        ]


DEFAULT_GPT_OPTIONS = GptOptions()


def add_gpt_options(parser: argparse.ArgumentParser) -> None:
    """Add --xmx / --cache / --threads / --tile-size to a CLI parser."""
    g = parser.add_argument_group(
        "GPT memory / performance",
        "Override SNAP's settings for this run (a flag always wins over "
        "gpt.vmoptions and snap.properties).  Rules of thumb: xmx ~ 2/3 of the "
        "RAM; cache 1/4-1/3 of xmx (too small only costs time, too big starves "
        "the operators); threads <= hardware threads, physical cores when memory "
        "is tight; tile-size a power of two, 512 unless you know why.",
    )
    g.add_argument("--xmx", default=DEFAULT_XMX, metavar="SIZE",
                   help=f"Java heap ceiling, e.g. 16G (default: {DEFAULT_XMX})")
    g.add_argument("--cache", default=DEFAULT_CACHE, metavar="SIZE",
                   help=f"tile cache, inside the heap, e.g. 4096M (default: {DEFAULT_CACHE})")
    g.add_argument("--threads", default=DEFAULT_THREADS, type=int, metavar="N",
                   help=f"tiles computed in parallel (default: {DEFAULT_THREADS})")
    g.add_argument("--tile-size", default=DEFAULT_TILE_SIZE, type=int, metavar="PX",
                   help=f"tile edge in pixels, power of two (default: {DEFAULT_TILE_SIZE})")


def gpt_options_from_args(args: argparse.Namespace) -> GptOptions:
    """Build a GptOptions from a namespace produced with add_gpt_options."""
    return GptOptions(
        xmx=args.xmx, cache=args.cache, threads=args.threads, tile_size=args.tile_size
    )

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
    """Remove extra flag bands added by SNAP's Collocate (collocationFlags),
    rename the remaining bands, and declare 0.0 as the NoData value.

    SNAP's Write operator appends flag bands after data bands regardless of the
    BandSelect node.  We keep only the first len(band_names) bands and discard
    the rest, then set the band descriptions in-place.

    SNAP fills masked pixels (sea via nodataValueAtSea, out-of-swath areas)
    with the band no-data value 0.0, but that declaration is lost when the
    BEAM-DIMAP bands are converted to GeoTIFF, so it is re-set here on every
    band (a linear Gamma0 or coherence of exactly 0.0 does not occur in
    practice, so this is safe).
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
        band = ds.GetRasterBand(i)
        band.SetDescription(name)
        band.SetNoDataValue(0.0)
    ds = None


def _add_swath_suffix(path: str, swath: str) -> str:
    """Insert _IW1 / _IW2 / _IW3 before the file extension."""
    base, ext = os.path.splitext(path)
    return f"{base}_{swath}{ext}"


def _aoi_to_wkt(aoi: str) -> str:
    """Normalise the AOI to an inline WKT string, the only form GPT accepts.

    ``aoi`` may be an inline WKT string or a path to a WKT / GeoJSON file
    (see ``polygon_to_swaths_bursts.parse_polygon``); the graphs' Subset node
    reads ``${aoi}`` as WKT, so a file is parsed and re-serialised here.
    """
    return parse_polygon(aoi).wkt


def polygon_to_swaths_bursts(product_path: str, aoi: str, coarse: bool = True) -> list[dict]:
    """
    Find which Sentinel-1 IW subswath(es) and burst range intersect the AOI.

    Thin wrapper around ``polygon_to_swaths_bursts.get_intersecting_bursts``
    (the module of the same name) that reshapes its ``{swath: [burst numbers]}``
    summary into the triplets TOPSAR-Split expects (``subswath``,
    ``first_burst``, ``last_burst``).  See ``readme_polygon_to_swaths_bursts.md``
    for how the footprints are rebuilt from the annotation XML.

    Why ``coarse`` defaults to True here (the module itself defaults to False):
    the footprints are rebuilt from the geolocation grid, whose rows sit on the
    burst boundaries, so consecutive bursts *touch* without overlapping and
    the outline is only accurate to ~1 km near the edges — whereas the real
    valid data of neighbouring bursts overlap by about a kilometre.  An AOI
    whose edge falls in that band can therefore need a burst the strict test
    misses.  In this pipeline the cost of the two errors is very asymmetric:
    the AOI is also the Subset clip applied after terrain correction, so a
    missing burst does not raise anything — it leaves a nodata hole inside the
    final GeoTIFF, which surfaces much later as spurious "changes" in the
    detection.  An extra burst only costs a few seconds of processing and is
    stitched cleanly by TOPSAR-Deburst.  Dilating the footprints by ~2 km
    before the test (``coarse=True``) buys the recall at that price.

    Args:
        product_path (str): Sentinel-1 SLC product (.zip archive or .SAFE directory).
        aoi (str): Area of interest in lon/lat WGS84 — inline WKT, or a path
            to a WKT / GeoJSON file.
        coarse (bool): Dilate the footprints before the test (default True,
            see above).  Set False to reproduce the module's strict result.

    Returns:
        List of dicts, one per intersecting subswath, sorted by subswath, e.g.::

            [{"subswath": "IW2", "first_burst": 3, "last_burst": 5}]

        Empty if no subswath intersects the AOI.
    """
    _, summary = get_intersecting_bursts(product_path, aoi, coarse=coarse)
    return [
        {"subswath": swath, "first_burst": min(bursts), "last_burst": max(bursts)}
        for swath, bursts in sorted(summary.items())
    ]


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

    Band names produced by SNAP (dates, ``_mst``/``_slv``, subswath) are not
    reliable, so nothing is parsed from them.  Instead the names Collocate will
    produce are *predicted* from the input band names + a fixed suffix, and
    the pre/post split is done purely by band position.

    Collocate suffix convention (must match the gathering.xml sources order
    AND ``referenceProductName`` = backscatter, see the header comment in
    ``vigisar_graphs/gathering.xml``):
        input_backscatter → reference → ``_M``
        input_coh_pre     → first secondary → ``_S0``
        input_coh_post    → second secondary → ``_S1``

    SNAP's CreateStack always writes master bands before slave bands.
    Since run_backscatter is called with input1=pre2 (master) and input2=post1
    (slave), the first half of backscatter bands is always pre2 and the second
    half is always post1 — no date parsing required.

    Hidden assumptions (breaking any of them will NOT raise an error, the
    products will just be silently mislabelled):
        * The three ``<sourceProduct>`` of the Collocate node in gathering.xml
          keep the order backscatter / coh_pre / coh_post.  Swapping the two
          coherence inputs swaps pre and post coherence (same band count).
        * ``run_backscatter`` keeps input1=pre2, input2=post1, and the first
          ``<sourceProduct>`` of CreateStack in backscatter.xml is input1.
        * The backscatter stack contains exactly master + slave bands; the
          even-count check below cannot detect two extra bands.

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


def _run_gpt(
    gpt_path: str,
    graph_xml: str,
    params: dict,
    gpt_options: GptOptions = DEFAULT_GPT_OPTIONS,
) -> bool:
    """
    Internal helper: build a GPT command from a parameter dict and execute it.

    Parameters are passed as ``-Pkey=value`` flags, substituting ``${key}``
    placeholders in the XML graph.  Memory / performance flags come from
    ``gpt_options`` (see the GptOptions section at the top of this module).
    Returns True on success, False on failure.

    Raises:
        FileNotFoundError: If gpt_path does not exist on the filesystem.
    """
    if not os.path.exists(gpt_path):
        raise FileNotFoundError(f"GPT executable not found at: {gpt_path}")

    # -e → full Java stack trace on error; the rest: heap, tile size, cache,
    # threads — every one of them overrides SNAP's own configuration files.
    command = [gpt_path, graph_xml, "-e", *gpt_options.to_args()]
    for key, value in params.items():
        command.append(f"-P{key}={value}")

    print(f"Running graph: {os.path.basename(graph_xml)}")
    try:
        subprocess.run(command, check=True, text=True)
        print("Processing completed successfully.")
        return True
    except subprocess.CalledProcessError:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backscatter(
    input1: str,
    input2: str,
    aoi: str,
    output: Optional[str] = None,
    gpt_path: str = DEFAULT_GPT,
    gpt_options: GptOptions = DEFAULT_GPT_OPTIONS,
) -> list[Optional[str]]:
    """
    Run the backscatter graph on two Sentinel-1 SLC products.

    Processing chain:
        Apply-Orbit-File → TOPSAR-Split → ThermalNoiseRemoval → Calibration
        → TOPSAR-Deburst → CreateStack → Cross-Correlation → Warp
        → Speckle-Filter → Terrain-Correction → Subset → Write

    The subswath and burst range are determined automatically from ``aoi`` using
    ``polygon_to_swaths_bursts`` (coarse mode, see its docstring).  If the AOI
    spans multiple subswaths the graph is run once per subswath and the outputs
    are suffixed with the subswath name (e.g. ``backscatter_IW2.dim``).

    Args:
        input1 (str): Path to the master Sentinel-1 SLC product (.zip or .SAFE).
            Should be the pre2 image so that the master bands appear first in the
            output stack (required by ``run_gathering``).
        input2 (str): Path to the secondary Sentinel-1 SLC product (.zip or .SAFE).
            Should be the post1 image.
        aoi (str): Area of interest in lon/lat WGS84 — inline WKT, or a path to
            a WKT / GeoJSON file.  Used both to locate the correct subswath/burst
            range and to spatially clip the Terrain-Correction output via the
            Subset node.
        output (str, optional): Full path for the output product (.dim).
            Defaults to ``data/preprocessed/temp/backscatter.dim``.
            If multiple subswaths are found, the subswath name is inserted before
            the extension (e.g. ``backscatter_IW2.dim``).
        gpt_path (str): Absolute path to the SNAP GPT executable.
        gpt_options (GptOptions): Heap / cache / threads / tile size for gpt
            (see the top of this module).

    Returns:
        list: One entry per processed subswath (stdout string or None on error).

    Raises:
        FileNotFoundError: If gpt_path does not exist.
        ValueError: If the AOI does not intersect any subswath in input1.
    """
    swaths = polygon_to_swaths_bursts(input1, aoi)
    if not swaths:
        raise ValueError(f"The AOI does not intersect any subswath in {input1}")
    aoi_wkt = _aoi_to_wkt(aoi)

    base_path = output if output is not None else os.path.join(_TEMP_DIR, "backscatter.dim")
    os.makedirs(os.path.dirname(os.path.abspath(base_path)), exist_ok=True)

    results = []
    for swath in swaths:
        out = _add_swath_suffix(base_path, swath["subswath"]) if len(swaths) > 1 else base_path
        params = {
            "input1":      input1,
            "input2":      input2,
            "output":      out,
            "aoi":         aoi_wkt,
            "subswath":    swath["subswath"],
            "first_burst": str(swath["first_burst"]),
            "last_burst":  str(swath["last_burst"]),
        }
        results.append(_run_gpt(gpt_path, _GRAPH_BACKSCATTER, params, gpt_options))

    return results


def run_coherence(
    input1: str,
    input2: str,
    aoi: str,
    pair: Optional[str] = None,
    output: Optional[str] = None,
    gpt_path: str = DEFAULT_GPT,
    gpt_options: GptOptions = DEFAULT_GPT_OPTIONS,
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
        aoi (str): Area of interest in lon/lat WGS84 — inline WKT, or a path to
            a WKT / GeoJSON file.
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
        gpt_options (GptOptions): Heap / cache / threads / tile size for gpt
            (see the top of this module).

    Returns:
        list: One entry per processed subswath (stdout string or None on error).

    Raises:
        FileNotFoundError: If gpt_path does not exist.
        ValueError: If pair is not ``"pre"``, ``"post"``, or None.
        ValueError: If the AOI does not intersect any subswath in input1.
    """
    if pair is not None and pair not in ("pre", "post"):
        raise ValueError(f"pair must be 'pre', 'post', or None, got {pair!r}")

    swaths = polygon_to_swaths_bursts(input1, aoi)
    if not swaths:
        raise ValueError(f"The AOI does not intersect any subswath in {input1}")
    aoi_wkt = _aoi_to_wkt(aoi)

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
            "aoi":         aoi_wkt,
            "subswath":    swath["subswath"],
            "first_burst": str(swath["first_burst"]),
            "last_burst":  str(swath["last_burst"]),
        }
        results.append(_run_gpt(gpt_path, _GRAPH_COHERENCE, params, gpt_options))

    return results


def run_gathering(
    input_backscatter: str,
    input_coh_pre: str,
    input_coh_post: str,
    output: Optional[str] = None,
    gpt_path: str = DEFAULT_GPT,
    gpt_options: GptOptions = DEFAULT_GPT_OPTIONS,
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
        gpt_options (GptOptions): Heap / cache / threads / tile size for gpt
            (see the top of this module).

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
    _run_gpt(gpt_path, _GRAPH_GATHERING, params, gpt_options)

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

    data_dir = dim_path.replace(".dim", ".data")
    produced = []
    for path, indices, names in [
        (pre_path,  mst_idx, mst_names),
        (post_path, slv_idx, slv_names),
    ]:
        # BEAM-DIMAP stores each band as <name>.img + <name>.hdr (ENVI) inside
        # the .data/ directory. GDAL cannot open the .dim XML directly, so we
        # open individual .img files and merge them into one GeoTIFF via a VRT.
        raw_names = [band_names[i - 1] for i in indices]
        img_paths = [os.path.join(data_dir, f"{bn}.img") for bn in raw_names]

        vrt = gdal.BuildVRT("", img_paths, separate=True)
        if vrt is None:
            continue
        gdal.Translate(path, vrt, format="GTiff")
        vrt = None

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
    gpt_options: GptOptions = DEFAULT_GPT_OPTIONS,
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
        aoi (str): Area of interest in lon/lat WGS84 — inline WKT, or a path to
            a WKT / GeoJSON file.  Used to clip the output after terrain
            correction.
        output (str, optional): Controls where the two output GeoTIFFs are written.
            Follows the same convention as ``run_gathering``:

            * ``None``          → ``data/preprocessed/default/pre.tif`` and ``post.tif``
            * Simple name       → ``data/preprocessed/<name>/<name>_pre.tif`` and ``_post.tif``
            * Full path prefix  → ``<prefix>_pre.tif`` and ``<prefix>_post.tif``
              (the folder must already exist or will be created)
        gpt_path (str): Absolute path to the SNAP GPT executable.
        gpt_options (GptOptions): Heap / cache / threads / tile size for gpt
            (see the top of this module).

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
    tmp_data = os.path.join(_TEMP_DIR, "backscatter_grd.data")
    os.makedirs(_TEMP_DIR, exist_ok=True)

    # Remove any partial output from a previous failed run so SNAP starts clean.
    import shutil
    for path in (tmp_dim, tmp_data):
        if os.path.exists(path):
            try:
                shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
            except PermissionError:
                print(f"Warning: cannot delete {path} (file locked by another process). "
                      "Close SNAP GUI if it has this file open.", file=sys.stderr)

    success = _run_gpt(gpt_path, _GRAPH_BACKSCATTER_GRD, {
        "input1": pre,
        "input2": post,
        "aoi":    _aoi_to_wkt(aoi),
        "output": tmp_dim,
    }, gpt_options)

    if not success or not os.path.exists(tmp_dim):
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
    ds = gdal.Warp(output, inputs, format="GTiff", resampleAlg="near",
                   srcNodata=0.0, dstNodata=0.0)
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
    add_gpt_options(common)

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
    p_bs.add_argument("--aoi", required=True, metavar="WKT_OR_FILE",
                      help=(
                          "[required] Area of interest in lon/lat WGS84: an inline WKT polygon "
                          '(must be quoted: --aoi "POLYGON ((-54.1 4.1, ...))") or a path to a '
                          "WKT / GeoJSON file.  Used to locate the correct subswath/burst range "
                          "and to clip the output."
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
    p_coh.add_argument("--aoi", required=True, metavar="WKT_OR_FILE",
                       help=(
                           "[required] Area of interest in lon/lat WGS84: an inline WKT polygon "
                           '(must be quoted: --aoi "POLYGON ((-54.1 4.1, ...))") or a path to a '
                           "WKT / GeoJSON file."
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
            "(slave/post image), each with gamma0_VH and gamma0_VV bands.\n\n"
            "Unlike the SLC pipeline there is no subswath/burst detection step — GRD products\n"
            "already cover the full swath."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_grd.add_argument("--pre",    required=True, metavar="PATH",
                       help="[required] Pre-event GRD product (.zip or .SAFE) — used as master")
    p_grd.add_argument("--post",   required=True, metavar="PATH",
                       help="[required] Post-event GRD product (.zip or .SAFE) — used as slave")
    p_grd.add_argument("--aoi",    required=True, metavar="WKT_OR_FILE",
                       help=(
                           "[required] Area of interest in lon/lat WGS84: an inline WKT polygon "
                           '(must be quoted: --aoi "POLYGON ((-54.1 4.1, ...))") or a path to a '
                           "WKT / GeoJSON file."
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
    gpt_options = gpt_options_from_args(args)

    if args.command == "backscatter":
        run_backscatter(
            input1=args.input1,
            input2=args.input2,
            aoi=args.aoi,
            output=args.output,
            gpt_path=args.gpt,
            gpt_options=gpt_options,
        )
    elif args.command == "coherence":
        run_coherence(
            input1=args.input1,
            input2=args.input2,
            aoi=args.aoi,
            pair=args.pair,
            output=args.output,
            gpt_path=args.gpt,
            gpt_options=gpt_options,
        )
    elif args.command == "gathering":
        run_gathering(
            input_backscatter=args.input_backscatter,
            input_coh_pre=args.input_coh_pre,
            input_coh_post=args.input_coh_post,
            output=args.output,
            gpt_path=args.gpt,
            gpt_options=gpt_options,
        )
    elif args.command == "backscatter-grd":
        tifs = run_backscatter_grd(
            pre=args.pre,
            post=args.post,
            aoi=args.aoi,
            output=args.output,
            gpt_path=args.gpt,
            gpt_options=gpt_options,
        )
        for path in tifs:
            print(path)


if __name__ == "__main__":
    main()
