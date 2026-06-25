import argparse
import os

try:
    from .run_graphs import (
        run_backscatter, run_coherence, run_gathering, run_mosaic,
        DEFAULT_GPT, _TEMP_DIR, _PREPROCESSED_DIR,
    )
    from .find_swaths_and_bursts import find_subswath
except ImportError:
    from run_graphs import (  # type: ignore[no-redef]
        run_backscatter, run_coherence, run_gathering, run_mosaic,
        DEFAULT_GPT, _TEMP_DIR, _PREPROCESSED_DIR,
    )
    from find_swaths_and_bursts import find_subswath  # type: ignore[no-redef]


def main_preprocess(
    pre1: str,
    pre2: str,
    post1: str,
    post2: str,
    aoi: str,
    output_name: str,
    gpt_path: str = DEFAULT_GPT,
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
        aoi (str): Area of interest as a WKT polygon in WGS84.
        output_name (str): Label for this run (e.g. ``"zta1"``).
            A folder ``data/preprocessed/<output_name>/`` is created.
        gpt_path (str): Path to the SNAP GPT executable.

    Returns:
        dict: ``{"pre": <path>, "post": <path>}`` — absolute paths of the
        two final GeoTIFFs.

    Raises:
        ValueError: If no subswath intersects the AOI.
    """
    # 1. Detect subswaths once (pre2 as reference product)
    swaths = find_subswath(pre2, aoi)
    if not swaths:
        raise ValueError(f"No subswath intersects the given AOI in {pre2!r}")

    multi = len(swaths) > 1

    # 2-4. Per-pair processing (each function loops over swaths internally)
    run_backscatter(pre2, post1, aoi, gpt_path=gpt_path)
    run_coherence(pre1, pre2,   aoi, pair="pre",  gpt_path=gpt_path)
    run_coherence(post1, post2, aoi, pair="post", gpt_path=gpt_path)

    # 5. Gathering — all outputs go into the same output_name/ folder
    out_dir = os.path.join(_PREPROCESSED_DIR, output_name)
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
        gather_prefix = os.path.join(out_dir, f"{output_name}{suffix}")
        tifs = run_gathering(bs_path, coh_pre_path, coh_post_path,
                             output=gather_prefix, gpt_path=gpt_path)

        if len(tifs) >= 1:
            pre_tifs.append(tifs[0])
        if len(tifs) >= 2:
            post_tifs.append(tifs[1])

    # 6. Single-swath: gathering already wrote the final files
    if not multi:
        return {"pre": pre_tifs[0], "post": post_tifs[0]}

    # Multi-swath: mosaic the per-swath tiles (all already in out_dir)

    final_pre  = os.path.join(out_dir, f"{output_name}_pre.tif")
    final_post = os.path.join(out_dir, f"{output_name}_post.tif")

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
            "                    bands: Gamma0_VH (pre2), Gamma0_VV (pre2),\n"
            "                           coh_VH (pre1×pre2), coh_VV (pre1×pre2)\n"
            "  <NAME>_post.tif — post-event product\n"
            "                    bands: Gamma0_VH (post1), Gamma0_VV (post1),\n"
            "                           coh_VH (post1×post2), coh_VV (post1×post2)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pre1",   required=True, metavar="PATH", help="[required] Earliest SLC product (.zip or .SAFE)")
    parser.add_argument("--pre2",   required=True, metavar="PATH", help="[required] Second SLC product (.zip or .SAFE)")
    parser.add_argument("--post1",  required=True, metavar="PATH", help="[required] Third SLC product (.zip or .SAFE)")
    parser.add_argument("--post2",  required=True, metavar="PATH", help="[required] Latest SLC product (.zip or .SAFE)")
    parser.add_argument("--aoi",    required=True, metavar="WKT",
                        help=(
                            "[required] Area of interest as a WKT polygon in WGS84.  "
                            'Must be quoted: --aoi "POLYGON ((-54.1 4.1, ...))"'
                        ))
    parser.add_argument("--output", required=True, metavar="NAME",
                        help="[required] Run label — creates data/preprocessed/<NAME>/")
    parser.add_argument("--gpt",    default=DEFAULT_GPT, metavar="PATH",
                        help=f"[optional] Path to the SNAP GPT executable (default: {DEFAULT_GPT!r})")

    args = parser.parse_args()
    result = main_preprocess(
        pre1=args.pre1,
        pre2=args.pre2,
        post1=args.post1,
        post2=args.post2,
        aoi=args.aoi,
        output_name=args.output,
        gpt_path=args.gpt,
    )
    print(f"Pre-event product : {result['pre']}")
    print(f"Post-event product: {result['post']}")


if __name__ == "__main__":
    main()
