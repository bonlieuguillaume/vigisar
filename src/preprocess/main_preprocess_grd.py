import argparse
import os

try:
    from .run_graphs import run_backscatter_grd, DEFAULT_GPT, _PREPROCESSED_DIR
except ImportError:
    from run_graphs import run_backscatter_grd, DEFAULT_GPT, _PREPROCESSED_DIR  # type: ignore[no-redef]


def main_preprocess_grd(
    pre: str,
    post: str,
    aoi: str,
    output_name: str,
    gpt_path: str = DEFAULT_GPT,
) -> dict:
    """
    GRD-only Vigisar SAR preprocessing pipeline.

    Runs a single SNAP graph on two Sentinel-1 GRD products and produces two
    final GeoTIFFs: one pre-event and one post-event, each containing Gamma0
    backscatter bands (VH and VV).

    Unlike the SLC pipeline, no subswath detection is needed — GRD products
    already cover the full swath.  Both images are processed together through
    orbit correction, thermal noise removal, border noise removal, radiometric
    calibration, cross-correlation coregistration, speckle filtering, terrain
    correction, and spatial clipping.

    Image roles:
        pre  — pre-event acquisition (master for coregistration)
        post — post-event acquisition (slave)

    Processing steps:
        1. Run the backscatter_grd graph (both images in one pass)
        2. Split the coregistered stack into pre and post GeoTIFFs

    Args:
        pre (str): Path to the pre-event Sentinel-1 GRD product (.zip or .SAFE).
        post (str): Path to the post-event Sentinel-1 GRD product (.zip or .SAFE).
        aoi (str): Area of interest as a WKT polygon in WGS84.
        output_name (str): Label for this run (e.g. ``"zta1"``), or a path.

            * Simple name (``"zta1"``) — a folder ``data/preprocessed/zta1/``
              is created and the products are written inside it.
            * Path (``"data/preprocessed/zta6/zta6_grd"``) — the products are
              written inside that directory (created if needed), using the
              last segment (``zta6_grd``) as filename prefix.
        gpt_path (str): Path to the SNAP GPT executable.

    Returns:
        dict: ``{"pre": <path>, "post": <path>}`` — absolute paths of the
        two final GeoTIFFs.

            - ``<output_name>_pre.tif``:  Gamma0_VH and Gamma0_VV from the pre image
            - ``<output_name>_post.tif``: Gamma0_VH and Gamma0_VV from the post image

    Raises:
        RuntimeError: If the GPT graph fails to produce output files.
    """
    if os.sep in output_name or "/" in output_name:
        out_dir = os.path.abspath(output_name)
        prefix  = os.path.basename(os.path.normpath(output_name))
    else:
        out_dir = os.path.join(_PREPROCESSED_DIR, output_name)
        prefix  = output_name
    os.makedirs(out_dir, exist_ok=True)

    gather_prefix = os.path.join(out_dir, prefix)
    tifs = run_backscatter_grd(pre, post, aoi, output=gather_prefix, gpt_path=gpt_path)

    if len(tifs) < 2:
        raise RuntimeError(
            "GRD preprocessing failed: expected two output GeoTIFFs but got "
            f"{len(tifs)}.  Check GPT logs above for details."
        )

    return {"pre": tifs[0], "post": tifs[1]}


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "GRD-only Vigisar SAR preprocessing pipeline.\n\n"
            "Runs orbit correction, thermal noise removal, border noise removal,\n"
            "radiometric calibration, cross-correlation coregistration, speckle\n"
            "filtering, terrain correction, and spatial clipping on two Sentinel-1\n"
            "GRD acquisitions.\n\n"
            "Image roles:\n"
            "  pre  — pre-event acquisition (master for coregistration)\n"
            "  post — post-event acquisition (slave)\n\n"
            "Outputs (written to data/preprocessed/<NAME>/):\n"
            "  <NAME>_pre.tif  — pre-event product\n"
            "                    bands: Gamma0_VH (pre), Gamma0_VV (pre)\n"
            "  <NAME>_post.tif — post-event product\n"
            "                    bands: Gamma0_VH (post), Gamma0_VV (post)\n\n"
            "If --output is a path instead of a simple name, the products are\n"
            "written inside that directory (created if needed) and its last\n"
            "segment is used as filename prefix, e.g.:\n"
            "  --output data/preprocessed/zta6/zta6_grd\n"
            "  → data/preprocessed/zta6/zta6_grd/zta6_grd_pre.tif and zta6_grd_post.tif\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pre",    required=True, metavar="PATH",
                        help="[required] Pre-event GRD product (.zip or .SAFE)")
    parser.add_argument("--post",   required=True, metavar="PATH",
                        help="[required] Post-event GRD product (.zip or .SAFE)")
    parser.add_argument("--aoi",    required=True, metavar="WKT",
                        help=(
                            "[required] Area of interest as a WKT polygon in WGS84.  "
                            'Must be quoted: --aoi "POLYGON ((-54.1 4.1, ...))"'
                        ))
    parser.add_argument("--output", required=True, metavar="NAME_OR_PATH",
                        help=(
                            "[required] Run label or output path.  "
                            "Simple name: creates data/preprocessed/<NAME>/ and writes "
                            "<NAME>_pre.tif and <NAME>_post.tif inside it.  "
                            "Path (e.g. data/preprocessed/zta6/zta6_grd): creates that "
                            "directory if needed and writes zta6_grd_pre.tif and "
                            "zta6_grd_post.tif inside it."
                        ))
    parser.add_argument("--gpt",    default=DEFAULT_GPT, metavar="PATH",
                        help=f"[optional] Path to the SNAP GPT executable (default: {DEFAULT_GPT!r})")

    args = parser.parse_args()
    result = main_preprocess_grd(
        pre=args.pre,
        post=args.post,
        aoi=args.aoi,
        output_name=args.output,
        gpt_path=args.gpt,
    )
    print(f"Pre-event product : {result['pre']}")
    print(f"Post-event product: {result['post']}")


if __name__ == "__main__":
    main()
