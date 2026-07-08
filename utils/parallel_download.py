"""
Parallel download of Sentinel-1 .SAFE products from CDSE via rclone.

Given a list of remote product paths (data/data_raw/list.txt), this builds an
rclone filter file, runs a single parallelized `rclone copy` covering every
product at once, then flattens the resulting date/mission tree so every
.SAFE folder ends up directly under data/data_raw/<folder>/.
"""

import argparse
import os
import subprocess
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_RAW_DIR = _PROJECT_ROOT / "data" / "data_raw"

DEFAULT_REMOTE = "cdse:eodata"
DEFAULT_FOLDER = "vrac"
DEFAULT_TRANSFERS = 8
DEFAULT_MULTI_THREAD_STREAMS = 8


def _normalize_remote_path(raw: str) -> str:
    """Turn a raw S3-style product path into a path relative to the `eodata` bucket root."""
    parts = [p for p in raw.strip().split("/") if p and p != "**"]
    if parts and parts[0].lower() == "eodata":
        parts = parts[1:]
    return "/" + "/".join(parts)


def _raw_path_to_filter_line(raw: str) -> str:
    return f"+ {_normalize_remote_path(raw)}/**"


def build_filter_lines(raw_lines: list) -> list:
    """
    Convert list.txt lines into rclone filter syntax.

    Lines already in filter syntax (starting with "+" or "-") are kept as-is,
    so the function is safe to re-run on a list.txt that was already converted.
    A trailing "- *" catch-all (excluding everything else) is (re-)appended once.
    """
    lines = []
    seen = set()
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        if raw.startswith("+") or raw.startswith("-"):
            if raw.replace(" ", "") == "-*":
                continue
            line = raw
        else:
            line = _raw_path_to_filter_line(raw)
        if line not in seen:
            seen.add(line)
            lines.append(line)

    if not lines:
        raise ValueError("No product paths found to build the rclone filter from.")

    lines.append("- *")
    return lines


def build_filter_file(list_path: Path) -> Path:
    """Read list_path and write the corresponding filter.txt next to it."""
    with open(list_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    filter_lines = build_filter_lines(raw_lines)

    filter_path = list_path.parent / "filter.txt"
    with open(filter_path, "w", encoding="utf-8") as f:
        f.write("\n".join(filter_lines) + "\n")

    return filter_path


def run_rclone_copy(
    filter_path: Path,
    dest_dir: Path,
    transfers: int = DEFAULT_TRANSFERS,
    multi_thread_streams: int = DEFAULT_MULTI_THREAD_STREAMS,
    remote: str = DEFAULT_REMOTE,
) -> None:
    """Run a single parallelized rclone copy of every product matched by filter_path into dest_dir."""
    cmd = [
        "rclone", "copy", remote, ".",
        "--filter-from", str(filter_path),
        "--transfers", str(transfers),
        "--multi-thread-streams", str(multi_thread_streams),
        "--progress",
    ]
    subprocess.run(cmd, cwd=dest_dir, check=True)


def remove_empty_dirs(root: Path) -> None:
    """Remove every directory under root left empty after flattening (bottom-up, non-destructive)."""
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        p = Path(dirpath)
        if p == root:
            continue
        try:
            p.rmdir()
        except OSError:
            pass  # not empty (or in use) — leave it alone


def flatten_safe_dirs(dest_dir: Path) -> list:
    """Move every *.SAFE directory found anywhere under dest_dir up to its root, then prune the empty tree."""
    safe_dirs = sorted(p for p in dest_dir.rglob("*.SAFE") if p.is_dir())

    moved = []
    for safe_dir in safe_dirs:
        target = dest_dir / safe_dir.name
        if safe_dir == target:
            moved.append(target)
            continue
        if target.exists():
            print(f"[WARN] {target} already exists, skipping {safe_dir}")
            continue
        safe_dir.rename(target)
        moved.append(target)

    remove_empty_dirs(dest_dir)
    return moved


def parallel_download(
    list_path: str = None,
    folder: str = DEFAULT_FOLDER,
    transfers: int = DEFAULT_TRANSFERS,
    multi_thread_streams: int = DEFAULT_MULTI_THREAD_STREAMS,
    remote: str = DEFAULT_REMOTE,
) -> Path:
    """
    Download every Sentinel-1 product listed in list_path into data/data_raw/<folder>/.

    Args:
        list_path: Path to the list of remote product paths (default: data/data_raw/list.txt).
        folder: Sub-folder of data/data_raw to download into (created if missing, reused if present).
        transfers: rclone --transfers value.
        multi_thread_streams: rclone --multi-thread-streams value.
        remote: rclone remote:bucket to copy from.

    Returns:
        Path to data/data_raw/<folder>, containing the flattened .SAFE products.
    """
    list_path = Path(list_path) if list_path else _DATA_RAW_DIR / "list.txt"
    dest_dir = _DATA_RAW_DIR / folder
    dest_dir.mkdir(parents=True, exist_ok=True)

    filter_path = build_filter_file(list_path)
    print(f"Filter file written to {filter_path}")

    run_rclone_copy(filter_path, dest_dir, transfers, multi_thread_streams, remote)

    moved = flatten_safe_dirs(dest_dir)
    print(f"{len(moved)} .SAFE product(s) available in {dest_dir}")

    return dest_dir


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parallel download of Sentinel-1 .SAFE products from CDSE via rclone.\n\n"
            "Reads a list of remote product paths, builds an rclone filter file, runs a\n"
            "single parallelized `rclone copy` for every product at once, then flattens\n"
            "the resulting mission/date tree so every .SAFE ends up directly under\n"
            "data/data_raw/<folder>/.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", default=str(_DATA_RAW_DIR / "list.txt"), metavar="PATH",
                        help=f"[optional] Path to the product list (default: {_DATA_RAW_DIR / 'list.txt'!s})")
    parser.add_argument("--folder", default=DEFAULT_FOLDER, metavar="NAME",
                        help=f"[optional] Sub-folder of data/data_raw to download into (default: {DEFAULT_FOLDER!r})")
    parser.add_argument("--transfers", type=int, default=DEFAULT_TRANSFERS, metavar="N",
                        help=f"[optional] rclone --transfers value (default: {DEFAULT_TRANSFERS})")
    parser.add_argument("--multi-thread-streams", type=int, default=DEFAULT_MULTI_THREAD_STREAMS, metavar="N",
                        help=f"[optional] rclone --multi-thread-streams value (default: {DEFAULT_MULTI_THREAD_STREAMS})")
    parser.add_argument("--remote", default=DEFAULT_REMOTE, metavar="REMOTE:BUCKET",
                        help=f"[optional] rclone remote to copy from (default: {DEFAULT_REMOTE!r})")

    args = parser.parse_args()
    parallel_download(
        list_path=args.list,
        folder=args.folder,
        transfers=args.transfers,
        multi_thread_streams=args.multi_thread_streams,
        remote=args.remote,
    )


if __name__ == "__main__":
    main()
