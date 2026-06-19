import subprocess
import os
import sys
import argparse
from typing import Optional

DEFAULT_GPT = r"C:\Program Files\esa-snap\bin\gpt.exe"

_PROJECT_ROOT     = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_GRAPHS_DIR       = os.path.join(_PROJECT_ROOT, "vigisar_graphs")
_PREPROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "preprocessed")
_TEMP_DIR         = os.path.join(_PREPROCESSED_DIR, "temp")

_GRAPH_BACKSCATTER = os.path.join(_GRAPHS_DIR, "backscatter.xml")
_GRAPH_COHERENCE   = os.path.join(_GRAPHS_DIR, "coherence.xml")
_GRAPH_GATHERING   = os.path.join(_GRAPHS_DIR, "gathering.xml")


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


def run_backscatter(
    input1: str,
    input2: str,
    output: Optional[str] = None,
    aoi: Optional[str] = None,
    gpt_path: str = DEFAULT_GPT,
) -> Optional[str]:
    """
    Run the backscatter graph on two Sentinel-1 SLC products.

    Processing chain:
        Apply-Orbit-File → TOPSAR-Split → ThermalNoiseRemoval → Calibration
        → TOPSAR-Deburst → CreateStack → Cross-Correlation → Warp
        → Speckle-Filter → Terrain-Correction → Subset → Write

    The output stack contains Gamma0 bands for both input acquisitions,
    coregistered to the master (input1) geometry.

    The TOPSAR-Split subswath and burst indices are currently fixed in the XML
    and must be edited manually there until a dedicated parameter is added.

    Args:
        input1 (str): Path to the master Sentinel-1 SLC product (.zip or .SAFE).
        input2 (str): Path to the secondary Sentinel-1 SLC product (.zip or .SAFE).
        output (str, optional): Path where the output product should be written (.dim).
            Defaults to ``data/preprocessed/temp/backscatter.dim``.
            When omitted, the file is overwritten on each run.
        aoi (str, optional): Area of interest as a WKT polygon fed to the Subset
            node (e.g. ``"POLYGON ((lon1 lat1, lon2 lat1, ...))"``).
            If None the Subset node receives an empty geometry, which causes SNAP
            to skip spatial clipping and write the full extent.
        gpt_path (str): Absolute path to the SNAP GPT executable.
            Defaults to ``C:\\Program Files\\esa-snap\\bin\\gpt.exe``.

    Returns:
        str: Standard output of the GPT process if successful.
        None: If the process exits with a non-zero return code.

    Raises:
        FileNotFoundError: If gpt_path does not exist.
    """
    if output is None:
        output = os.path.join(_TEMP_DIR, "backscatter.dim")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    params = {
        "input1": input1,
        "input2": input2,
        "output": output,
        "aoi": aoi if aoi is not None else "",
    }
    return _run_gpt(gpt_path, _GRAPH_BACKSCATTER, params)


def run_coherence(
    input1: str,
    input2: str,
    pair: str,
    output: Optional[str] = None,
    aoi: Optional[str] = None,
    gpt_path: str = DEFAULT_GPT,
) -> Optional[str]:
    """
    Run the coherence graph on two Sentinel-1 SLC products.

    This function must be called twice per full pipeline run: once for the
    pre-event pair (pre1 + pre2) and once for the post-event pair
    (post1 + post2).  The ``pair`` argument distinguishes the two runs and
    is used to build the output filename.

    Processing chain:
        Apply-Orbit-File → TOPSAR-Split → Back-Geocoding
        → Enhanced-Spectral-Diversity → Coherence → TOPSAR-Deburst
        → Terrain-Correction → Subset → Write

    The TOPSAR-Split subswath and burst indices are currently fixed in the XML
    and must be edited manually there until a dedicated parameter is added.

    Args:
        input1 (str): Path to the master Sentinel-1 SLC product (.zip or .SAFE).
        input2 (str): Path to the secondary Sentinel-1 SLC product (.zip or .SAFE).
        pair (str): Event period this pair belongs to — ``"pre"`` for the
            pre-event pair (pre1 + pre2) or ``"post"`` for the post-event pair
            (post1 + post2).  Used as a suffix in the output filename.
        output (str, optional): Base name for the output file.  The pair suffix
            is always appended, so passing ``"mysite"`` with ``pair="pre"``
            produces ``data/preprocessed/temp/mysite_pre.dim``.  If None the
            base name defaults to ``"coherence"``, giving
            ``data/preprocessed/temp/coherence_pre.dim`` or
            ``data/preprocessed/temp/coherence_post.dim``.
        aoi (str, optional): Area of interest as a WKT polygon fed to the Subset
            node (e.g. ``"POLYGON ((lon1 lat1, lon2 lat1, ...))"``).
            If None the Subset node receives an empty geometry, which causes SNAP
            to skip spatial clipping and write the full extent.
        gpt_path (str): Absolute path to the SNAP GPT executable.
            Defaults to ``C:\\Program Files\\esa-snap\\bin\\gpt.exe``.

    Returns:
        str: Standard output of the GPT process if successful.
        None: If the process exits with a non-zero return code.

    Raises:
        FileNotFoundError: If gpt_path does not exist.
        ValueError: If pair is not ``"pre"`` or ``"post"``.
    """
    if pair not in ("pre", "post"):
        raise ValueError(f"pair must be 'pre' or 'post', got {pair!r}")
    base = output if output is not None else "coherence"
    output = os.path.join(_TEMP_DIR, f"{base}_{pair}.dim")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    params = {
        "input1": input1,
        "input2": input2,
        "output": output,
        "aoi": aoi if aoi is not None else "",
    }
    return _run_gpt(gpt_path, _GRAPH_COHERENCE, params)


def run_gathering(
    input_gamma: str,
    input_coh_pre: str,
    input_coh_post: str,
    output: Optional[str] = None,
    gpt_path: str = DEFAULT_GPT,
) -> Optional[str]:
    """
    Run the gathering graph to collocate a backscatter product with two
    coherence stacks and split the result into pre-event and post-event products.

    Processing chain:
        3× Read → Collocate → BandSelect → Write (pre)
                           → BandSelect → Write (post)

    Note: The BandSelect nodes have band name lists that are currently fixed in
    the XML and depend on the acquisition dates encoded in the band names.
    They must be updated manually in the XML for each new dataset.

    Args:
        input_gamma (str): Path to the Gamma0 backscatter stack (.dim), output
            of ``run_backscatter``.
        input_coh_pre (str): Path to the pre-event coherence product (.dim),
            output of ``run_coherence`` for the pre-event pair.
        input_coh_post (str): Path to the post-event coherence product (.dim),
            output of ``run_coherence`` for the post-event pair.
        output (str, optional): Name for this processing run.  A subfolder with
            that name is created under ``data/preprocessed/`` and the two output
            GeoTIFFs are written there as ``<name>_pre.tif`` and
            ``<name>_post.tif``.  If None, outputs are written to
            ``data/preprocessed/default/`` as ``pre.tif`` and ``post.tif``.
        gpt_path (str): Absolute path to the SNAP GPT executable.
            Defaults to ``C:\\Program Files\\esa-snap\\bin\\gpt.exe``.

    Returns:
        str: Standard output of the GPT process if successful.
        None: If the process exits with a non-zero return code.

    Raises:
        FileNotFoundError: If gpt_path does not exist.
    """
    if output is None:
        folder = os.path.join(_PREPROCESSED_DIR, "default")
        output_pre  = os.path.join(folder, "pre")
        output_post = os.path.join(folder, "post")
    else:
        folder = os.path.join(_PREPROCESSED_DIR, output)
        output_pre  = os.path.join(folder, f"{output}_pre")
        output_post = os.path.join(folder, f"{output}_post")

    os.makedirs(folder, exist_ok=True)
    params = {
        "input1": input_gamma,
        "input2": input_coh_pre,
        "input3": input_coh_post,
        "output_pre": output_pre,
        "output_post": output_post,
    }
    return _run_gpt(gpt_path, _GRAPH_GATHERING, params)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SNAP GPT preprocessing graphs for the Vigisar pipeline.\n\n"
            "Three subcommands are available, one per processing graph:\n"
            "  backscatter  — Gamma0 backscatter via cross-correlation coregistration\n"
            "  coherence    — coherence via ESD coregistration\n"
            "  gathering    — collocate backscatter + coherence into pre/post products\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--gpt",
        default=DEFAULT_GPT,
        metavar="PATH",
        help=f"Path to the SNAP GPT executable (default: {DEFAULT_GPT!r})",
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
            "cross-correlation coregistration, speckle filtering, terrain correction, "
            "and optional spatial subsetting.\n\n"
            "The output is a BEAM-DIMAP stack containing Gamma0 bands for both "
            "acquisitions, coregistered to the master (--input1) geometry."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_bs.add_argument("--input1", required=True, metavar="PATH",
                      help="Master Sentinel-1 SLC product (.zip or .SAFE)")
    p_bs.add_argument("--input2", required=True, metavar="PATH",
                      help="Secondary Sentinel-1 SLC product (.zip or .SAFE)")
    p_bs.add_argument("--output", default=None, metavar="PATH",
                      help=(
                          "Output product path (.dim). "
                          "Defaults to data/preprocessed/temp/backscatter.dim "
                          "(overwritten on each run if omitted)."
                      ))
    p_bs.add_argument("--aoi", default=None, metavar="WKT",
                      help=(
                          "Area of interest as a WKT polygon passed to the Subset node "
                          "(e.g. \"POLYGON ((lon1 lat1, lon2 lat1, ...))\"). "
                          "Omit to skip spatial clipping."
                      ))

    # -- coherence -----------------------------------------------------------
    p_coh = subparsers.add_parser(
        "coherence",
        parents=[common],
        help="Interferometric coherence via ESD coregistration",
        description=(
            "Process two Sentinel-1 SLC acquisitions through orbit correction, "
            "TOPSAR split, back-geocoding, Enhanced Spectral Diversity, coherence "
            "estimation, deburst, terrain correction, and optional spatial subsetting."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_coh.add_argument("--input1", required=True, metavar="PATH",
                       help="Master Sentinel-1 SLC product (.zip or .SAFE)")
    p_coh.add_argument("--input2", required=True, metavar="PATH",
                       help="Secondary Sentinel-1 SLC product (.zip or .SAFE)")
    p_coh.add_argument("--pair", required=True, choices=["pre", "post"],
                       help=(
                           "Event period this pair belongs to: 'pre' for the pre-event "
                           "pair (pre1 + pre2), 'post' for the post-event pair (post1 + post2)."
                       ))
    p_coh.add_argument("--output", default=None, metavar="NAME",
                       help=(
                           "Base name for the output file.  The pair suffix is always appended: "
                           "'mysite' + --pair pre → data/preprocessed/temp/mysite_pre.dim.  "
                           "Defaults to 'coherence', giving coherence_pre.dim or coherence_post.dim."
                       ))
    p_coh.add_argument("--aoi", default=None, metavar="WKT",
                       help=(
                           "Area of interest as a WKT polygon passed to the Subset node "
                           "(e.g. \"POLYGON ((lon1 lat1, lon2 lat1, ...))\"). "
                           "Omit to skip spatial clipping."
                       ))

    # -- gathering -----------------------------------------------------------
    p_ga = subparsers.add_parser(
        "gathering",
        parents=[common],
        help="Collocate backscatter and coherence stacks into pre/post GeoTIFFs",
        description=(
            "Collocate a Gamma0 backscatter stack with a pre-event and a post-event "
            "coherence product, then split the result into two GeoTIFF files named "
            "<output>_pre.tif and <output>_post.tif.\n\n"
            "Note: the BandSelect nodes inside the XML have band name lists that are "
            "dataset-specific and must be updated manually in the XML for each new "
            "dataset (acquisition dates are encoded in the band names)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_ga.add_argument("--input-gamma", required=True, metavar="PATH",
                      help="Gamma0 backscatter stack (.dim), output of the backscatter graph")
    p_ga.add_argument("--input-coh-pre", required=True, metavar="PATH",
                      help="Pre-event coherence product (.dim), output of the coherence graph")
    p_ga.add_argument("--input-coh-post", required=True, metavar="PATH",
                      help="Post-event coherence product (.dim), output of the coherence graph")
    p_ga.add_argument("--output", default=None, metavar="NAME",
                      help=(
                          "Name for this processing run.  A subfolder data/preprocessed/<name>/ "
                          "is created and the outputs are written as <name>_pre.tif and "
                          "<name>_post.tif inside it.  "
                          "Omit to write pre.tif and post.tif into data/preprocessed/default/."
                      ))

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "backscatter":
        run_backscatter(
            input1=args.input1,
            input2=args.input2,
            output=args.output,
            aoi=args.aoi,
            gpt_path=args.gpt,
        )
    elif args.command == "coherence":
        run_coherence(
            input1=args.input1,
            input2=args.input2,
            pair=args.pair,
            output=args.output,
            aoi=args.aoi,
            gpt_path=args.gpt,
        )
    elif args.command == "gathering":
        run_gathering(
            input_gamma=args.input_gamma,
            input_coh_pre=args.input_coh_pre,
            input_coh_post=args.input_coh_post,
            output=args.output,
            gpt_path=args.gpt,
        )


if __name__ == "__main__":
    main()
