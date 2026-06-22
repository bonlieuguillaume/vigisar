import argparse
import glob
import os
import xml.etree.ElementTree as ET
import zipfile
from typing import Optional

from shapely import wkt as shapely_wkt
from shapely.geometry import MultiPoint


def find_subswath(product_path: str, aoi_wkt: str) -> list[dict]:
    """
    Find which Sentinel-1 IW subswath(es) and burst range intersect a given AOI.

    Parses the geolocation grid and burst list from the annotation XML files
    embedded in the product (one file per subswath).  Works with both .zip
    archives and unpacked .SAFE directories.

    The function performs two successive intersection tests:
      1. A fast subswath-level test using the convex hull of all geolocation
         grid points for that subswath.
      2. A per-burst test using only the grid points belonging to each burst,
         to determine the tightest first/last burst range that covers the AOI.

    Args:
        product_path (str): Path to the Sentinel-1 SLC product
            (.zip archive or .SAFE directory).
        aoi_wkt (str): Area of interest as a WKT polygon in WGS84
            (e.g. ``"POLYGON ((-54.1 4.1, -53.8 4.1, ...))"``).

    Returns:
        List of dicts, one entry per intersecting subswath, e.g.::

            [{"subswath": "IW2", "first_burst": 3, "last_burst": 5}]

        Returns an empty list if no subswath intersects the AOI.

    Raises:
        ValueError: If the WKT geometry cannot be parsed.
        FileNotFoundError: If product_path does not exist.
    """
    if not os.path.exists(product_path):
        raise FileNotFoundError(f"Product not found: {product_path}")

    aoi = shapely_wkt.loads(aoi_wkt)
    results = []

    for iw in ("IW1", "IW2", "IW3"):
        xml_content = _read_annotation(product_path, iw)
        if xml_content is None:
            continue

        root = ET.fromstring(xml_content)

        # --- Geolocation grid: list of (lon, lat, line_index) ---------------
        grid_points = []
        for pt in root.findall(".//geolocationGridPoint"):
            grid_points.append((
                float(pt.find("longitude").text),
                float(pt.find("latitude").text),
                int(pt.find("line").text),
            ))

        if not grid_points:
            continue

        # --- Fast subswath-level intersection check --------------------------
        subswath_hull = MultiPoint(
            [(lon, lat) for lon, lat, _ in grid_points]
        ).convex_hull
        if not subswath_hull.intersects(aoi):
            continue

        # --- Per-burst intersection check ------------------------------------
        lines_per_burst = int(root.find(".//linesPerBurst").text)
        n_bursts = len(root.findall(".//burst"))

        first_burst = last_burst = None
        for i in range(n_bursts):
            start = i * lines_per_burst
            end = (i + 1) * lines_per_burst
            burst_pts = [
                (lon, lat)
                for lon, lat, ln in grid_points
                if start <= ln < end
            ]
            if len(burst_pts) < 3:
                continue
            if MultiPoint(burst_pts).convex_hull.intersects(aoi):
                if first_burst is None:
                    first_burst = i + 1  # SNAP burst indices are 1-based
                last_burst = i + 1

        if first_burst is not None:
            results.append({
                "subswath": iw,
                "first_burst": first_burst,
                "last_burst": last_burst,
            })

    return results


def _read_annotation(product_path: str, subswath: str) -> Optional[str]:
    """
    Return the VV annotation XML content for the given subswath, or None if
    the subswath is not present in the product.

    VV is used because it is always present in dual-pol (VV+VH) products.
    For VH-only products the filter would need to be adjusted.
    """
    iw = subswath.lower()  # "iw1", "iw2", "iw3"

    if product_path.endswith(".zip"):
        with zipfile.ZipFile(product_path) as zf:
            candidates = [
                name for name in zf.namelist()
                if f"-{iw}-slc-vv-" in name
                and name.endswith(".xml")
                and "/annotation/" in name
                and "calibration" not in name
                and "/rfi/" not in name
            ]
            if not candidates:
                return None
            return zf.read(candidates[0]).decode("utf-8")

    # .SAFE directory
    pattern = os.path.join(product_path, "annotation", f"*-{iw}-slc-vv-*.xml")
    files = [f for f in glob.glob(pattern) if "calibration" not in f]
    if not files:
        return None
    with open(files[0], encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find which Sentinel-1 IW subswath(es) and burst range intersect "
            "a given area of interest.\n\n"
            "The results can be used directly as --subswath, --first-burst and "
            "--last-burst arguments when running the backscatter or coherence xml "
            "graphs (once those parameters are wired up)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image",
        required=True,
        metavar="PATH",
        help="Path to the Sentinel-1 SLC product (.zip archive or .SAFE directory)",
    )
    parser.add_argument(
        "--aoi",
        required=True,
        metavar="WKT",
        help='Area of interest as a WKT polygon in WGS84 (e.g. "POLYGON ((-54.1 4.1, ...))")',
    )
    args = parser.parse_args()

    results = find_subswath(args.image, args.aoi)

    if not results:
        print("No subswath intersects the given AOI.")
        return

    for r in results:
        print(f"Subswath    : {r['subswath']}")
        print(f"First burst : {r['first_burst']}")
        print(f"Last burst  : {r['last_burst']}")
        print()


if __name__ == "__main__":
    main()
