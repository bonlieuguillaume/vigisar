import argparse
import os

try:
    from .run_graphs import (
        run_backscatter, run_coherence, run_gathering, run_mosaic,
        polygon_to_swaths_bursts, GptOptions, add_gpt_options, gpt_options_from_args,
        DEFAULT_GPT, DEFAULT_GPT_OPTIONS, _TEMP_DIR, _PREPROCESSED_DIR,
    )
except ImportError:
    from run_graphs import (  # type: ignore[no-redef]
        run_backscatter, run_coherence, run_gathering, run_mosaic,
        polygon_to_swaths_bursts, GptOptions, add_gpt_options, gpt_options_from_args,
        DEFAULT_GPT, DEFAULT_GPT_OPTIONS, _TEMP_DIR, _PREPROCESSED_DIR,
    )


def main_preprocess(
    pre1: str,
    pre2: str,
    post1: str,
    post2: str,
    aoi: str,
    output_name: str,
    gpt_path: str = DEFAULT_GPT,
    gpt_options: GptOptions = DEFAULT_GPT_OPTIONS,
) -> dict:
    """
    Full Vigisar SAR preprocessing pipeline.

    Runs all graphs in the correct order and produces two final GeoTIFFs:
    one pre-event product and one post-event product, each containing
    Gamma0 backscatter and interferometric coherence bands.

    Image roles:
        pre1  — earliest acquisition (coherence reference for pre pair)
        pre2  — second acquisition  (coherence secondary for pre pair;
                                     master for backscatter stack)
        post1 — third acquisition   (slave for backscatter stack;
                                     coherence reference for post pair)
        post2 — latest acquisition  (coherence secondary for post pair)

    Processing steps:
        1. Detect subswath(s) and burst range from AOI
        2. Backscatter stack  : pre2 × post1
        3. Pre-event coherence: pre1 × pre2
        4. Post-event coherence: post1 × post2
        5. Gathering per subswath → per-swath GeoTIFFs
        6. Mosaic  (only when the AOI spans multiple subswaths)

    Args:
        pre1 (str): Earliest SLC product (.zip or .SAFE).
        pre2 (str): Second SLC product.
        post1 (str): Third SLC product.
        post2 (str): Latest SLC product.
        aoi (str): Area of interest in lon/lat WGS84 — inline WKT, or a path
            to a WKT / GeoJSON file.
        output_name (str): Label for this run (e.g. ``"zta1"``), or a path.

            * Simple name (``"zta1"``) — a folder ``data/preprocessed/zta1/``
              is created and the products are written inside it.
            * Path (``"data/preprocessed/zta6/zta6_slc"``) — the products are
              written inside that directory (created if needed), using the
              last segment (``zta6_slc``) as filename prefix.
        gpt_path (str): Path to the SNAP GPT executable.
        gpt_options (GptOptions): Heap / cache / threads / tile size handed to
            every gpt call (see the top of ``run_graphs.py`` for how to
            choose them).

    Returns:
        dict: ``{"pre": <path>, "post": <path>}`` — absolute paths of the
        two final GeoTIFFs.

    Raises:
        ValueError: If no subswath intersects the AOI.
    """
    # 1. Detect subswaths once (pre2 as reference product). Coarse mode: an
    # AOI on a burst seam also gets the neighbouring burst — a missing burst
    # would leave a silent nodata hole in the final GeoTIFFs, an extra one
    # costs seconds (see polygon_to_swaths_bursts in run_graphs).
    swaths = polygon_to_swaths_bursts(pre2, aoi)
    if not swaths:
        raise ValueError(f"No subswath intersects the given AOI in {pre2!r}")

    multi = len(swaths) > 1

    # 2-4. Per-pair processing (each function loops over swaths internally)
    run_backscatter(pre2, post1, aoi, gpt_path=gpt_path, gpt_options=gpt_options)
    run_coherence(pre1, pre2,   aoi, pair="pre",  gpt_path=gpt_path, gpt_options=gpt_options)
    run_coherence(post1, post2, aoi, pair="post", gpt_path=gpt_path, gpt_options=gpt_options)

    # 5. Gathering — all outputs go into the same output folder
    if os.sep in output_name or "/" in output_name:
        out_dir = os.path.abspath(output_name)
        prefix  = os.path.basename(os.path.normpath(output_name))
    else:
        out_dir = os.path.join(_PREPROCESSED_DIR, output_name)
        prefix  = output_name
    os.makedirs(out_dir, exist_ok=True)

    pre_tifs:  list[str] = []
    post_tifs: list[str] = []

    for swath in swaths:
        iw     = swath["subswath"]
        suffix = f"_{iw}" if multi else ""

        bs_path       = os.path.join(_TEMP_DIR, f"backscatter{suffix}.dim")
        coh_pre_path  = os.path.join(_TEMP_DIR, f"coherence_pre{suffix}.dim")
        coh_post_path = os.path.join(_TEMP_DIR, f"coherence_post{suffix}.dim")

        # Pass a full path prefix so run_gathering writes directly into out_dir
        # (single-swath: zta1/zta1, multi-swath: zta1/zta1_IW1, zta1/zta1_IW2)
        gather_prefix = os.path.join(out_dir, f"{prefix}{suffix}")
        tifs = run_gathering(bs_path, coh_pre_path, coh_post_path,
                             output=gather_prefix, gpt_path=gpt_path,
                             gpt_options=gpt_options)

        if len(tifs) >= 1:
            pre_tifs.append(tifs[0])
        if len(tifs) >= 2:
            post_tifs.append(tifs[1])

    # 6. Single-swath: gathering already wrote the final files
    if not multi:
        return {"pre": pre_tifs[0], "post": post_tifs[0]}

    # Multi-swath: mosaic the per-swath tiles (all already in out_dir)

    final_pre  = os.path.join(out_dir, f"{prefix}_pre.tif")
    final_post = os.path.join(out_dir, f"{prefix}_post.tif")

    run_mosaic(pre_tifs,  final_pre)
    run_mosaic(post_tifs, final_post)

    return {"pre": final_pre, "post": final_post}


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Full Vigisar SAR preprocessing pipeline.\n\n"
            "Runs backscatter, coherence (pre + post), gathering, and mosaic\n"
            "in the correct order from four raw Sentinel-1 SLC acquisitions.\n\n"
            "Image roles:\n"
            "  pre1  — earliest acquisition\n"
            "  pre2  — second acquisition  (master for backscatter)\n"
            "  post1 — third acquisition   (slave for backscatter)\n"
            "  post2 — latest acquisition\n\n"
            "Outputs (written to data/preprocessed/<NAME>/):\n"
            "  <NAME>_pre.tif  — pre-event product\n"
            "                    bands: gamma0_VH (pre2), gamma0_VV (pre2),\n"
            "                           coh_VH (pre1×pre2), coh_VV (pre1×pre2)\n"
            "  <NAME>_post.tif — post-event product\n"
            "                    bands: gamma0_VH (post1), gamma0_VV (post1),\n"
            "                           coh_VH (post1×post2), coh_VV (post1×post2)\n\n"
            "If --output is a path instead of a simple name, the products are\n"
            "written inside that directory (created if needed) and its last\n"
            "segment is used as filename prefix, e.g.:\n"
            "  --output data/preprocessed/zta6/zta6_slc\n"
            "  → data/preprocessed/zta6/zta6_slc/zta6_slc_pre.tif and zta6_slc_post.tif\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pre1",   required=True, metavar="PATH", help="[required] Earliest SLC product (.zip or .SAFE)")
    parser.add_argument("--pre2",   required=True, metavar="PATH", help="[required] Second SLC product (.zip or .SAFE)")
    parser.add_argument("--post1",  required=True, metavar="PATH", help="[required] Third SLC product (.zip or .SAFE)")
    parser.add_argument("--post2",  required=True, metavar="PATH", help="[required] Latest SLC product (.zip or .SAFE)")
    parser.add_argument("--aoi",    required=True, metavar="WKT_OR_FILE",
                        help=(
                            "[required] Area of interest in lon/lat WGS84: an inline WKT polygon "
                            '(must be quoted: --aoi "POLYGON ((-54.1 4.1, ...))") or a path to a '
                            "WKT / GeoJSON file.  Used to locate the subswath/burst range and "
                            "to clip the outputs."
                        ))
    parser.add_argument("--output", required=True, metavar="NAME_OR_PATH",
                        help=(
                            "[required] Run label or output path.  "
                            "Simple name: creates data/preprocessed/<NAME>/ and writes "
                            "<NAME>_pre.tif and <NAME>_post.tif inside it.  "
                            "Path (e.g. data/preprocessed/zta6/zta6_slc): creates that "
                            "directory if needed and writes zta6_slc_pre.tif and "
                            "zta6_slc_post.tif inside it."
                        ))
    parser.add_argument("--gpt",    default=DEFAULT_GPT, metavar="PATH",
                        help=f"[optional] Path to the SNAP GPT executable (default: {DEFAULT_GPT!r})")
    add_gpt_options(parser)

    args = parser.parse_args()
    result = main_preprocess(
        pre1=args.pre1,
        pre2=args.pre2,
        post1=args.post1,
        post2=args.post2,
        aoi=args.aoi,
        output_name=args.output,
        gpt_path=args.gpt,
        gpt_options=gpt_options_from_args(args),
    )
    print(f"Pre-event product : {result['pre']}")
    print(f"Post-event product: {result['post']}")


if __name__ == "__main__":
    main()
