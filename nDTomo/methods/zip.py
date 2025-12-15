# -*- coding: utf-8 -*-
"""
Methods for zipping files and folders.

@author: Antony Vamvakeros

Example usage:

# (1) one zip per tif (recursive)
zips = zip_each_file_with_extension(r"D:\data", "tif", recursive=True)

# (2) one zip containing all png + tif under folder
z = zip_all_files_with_extension(r"D:\data", "png,tif", "all_images", recursive=True, keep_paths=True)

# (3) zip an explicit list of files (store basenames; rename if collisions)
z = zip_file_list(
    [r"D:\a\1.tif", r"D:\b\1.tif", r"D:\b\2.tif"],
    r"D:\out\selected.zip",
    keep_paths=False,
    on_collision="rename",
)

"""

from __future__ import annotations
import pathlib
import re
import shutil
import subprocess
import tempfile
import zipfile
from typing import Iterable, List, Optional, Sequence, Union

Path = pathlib.Path

# -----------------------------
# Helpers
# -----------------------------

def _normalize_exts(ext: str) -> List[str]:
    """
    Parse one or more filename extensions into canonical, lowercase suffixes.

    Parameters
    ----------
    ext
        Extension specification(s). Accepts common forms such as ``"tif"``,
        ``".tif"``, ``"*.tif"``, whitespace-separated strings (e.g. ``"tif png"``),
        or comma-separated strings (e.g. ``"tif,png"``). Matching is performed
        case-insensitively elsewhere.

    Returns
    -------
    list of str
        Normalized extensions as lowercase suffixes including the leading dot,
        e.g. ``[".png", ".tif"]``. Order is preserved and duplicates are removed.

    Raises
    ------
    ValueError
        If `ext` is empty or cannot be parsed into at least one extension.
    """
    if not isinstance(ext, str) or not ext.strip():
        raise ValueError("Provide at least one extension, e.g. 'png' or '.png'.")

    parts = ext.replace(",", " ").split()
    exts: List[str] = []
    for e in parts:
        e = e.strip().lower()
        if not e:
            continue
        if e.startswith("*."):
            e = e[1:]          # '*.png' -> '.png'
        if not e.startswith("."):
            e = "." + e
        exts.append(e)
    if not exts:
        raise ValueError("Provide at least one extension, e.g. 'png' or '.png'.")
    # dedupe, preserve order
    seen = set()
    out = []
    for e in exts:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def _iter_files(folder: Path, recursive: bool) -> Iterable[Path]:
    """
    Iterate over files within a directory.

    Parameters
    ----------
    folder
        Directory to search.
    recursive
        If ``True``, traverse all subdirectories using ``rglob("*")``.
        If ``False``, only consider direct children using ``glob("*")``.

    Yields
    ------
    pathlib.Path
        Paths to filesystem entries that are regular files.

    Notes
    -----
    This function does not filter by extension; callers should apply any
    suffix/pattern filtering.
    """    
    it = folder.rglob("*") if recursive else folder.glob("*")
    for p in it:
        if p.is_file():
            yield p


def _resolve_7z(seven_zip: Optional[Union[str, Path]] = None) -> str:
    """
    Resolve a 7-Zip executable for use with subprocess.

    Parameters
    ----------
    seven_zip
        Optional explicit path to the 7-Zip executable (``7z``/``7z.exe``).
        If provided, it must exist.

    Returns
    -------
    str
        Path to a usable 7-Zip executable.

    Raises
    ------
    FileNotFoundError
        If `seven_zip` is provided but does not exist, or if no executable can be
        found on PATH and no supported fallback location exists.

    Notes
    -----
    Resolution order:
    1) explicit `seven_zip` argument
    2) ``shutil.which("7z")`` / ``shutil.which("7z.exe")``
    3) a common Windows install path (currently ``D:\\Program Files\\7-Zip\\7z.exe``)
    """
    if seven_zip is not None:
        p = Path(seven_zip)
        if not p.exists():
            raise FileNotFoundError(f"7-Zip not found at: {p}")
        return str(p)

    from_path = shutil.which("7z") or shutil.which("7z.exe")
    if from_path:
        return from_path

    common = Path(r"D:\Program Files\7-Zip\7z.exe")
    if common.exists():
        return str(common)

    raise FileNotFoundError(
        "7-Zip executable not found. Provide seven_zip=... or add 7z to PATH."
    )


def _run_7z(cmd: Sequence[str], *, cwd: Optional[Union[str, Path]] = None) -> None:
    """
    Execute a 7-Zip command and raise a rich error message on failure.

    Parameters
    ----------
    cmd
        Argument vector passed to ``subprocess.run`` (e.g. ``["7z", "a", ...]``).
    cwd
        Optional working directory for command execution.

    Raises
    ------
    RuntimeError
        If 7-Zip returns a non-zero exit status. The raised error includes the
        command, return code, stdout and stderr to aid debugging.
    """
    try:
        subprocess.run(
            list(cmd),
            check=True,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = [
            "7-Zip failed.",
            f"Return code: {e.returncode}",
            f"Command: {' '.join(cmd)}",
            f"CWD: {cwd if cwd is not None else ''}",
            "--- STDOUT ---",
            e.stdout or "",
            "--- STDERR ---",
            e.stderr or "",
        ]
        raise RuntimeError("\n".join(msg)) from e


def _format_volume_size(volume: Union[str, int]) -> str:
    """
    Format a volume size specification for 7-Zip split archives (``-v``).

    Parameters
    ----------
    volume
        Split size. If a string, accepted examples include ``"10g"``, ``"500m"``,
        ``"100k"``, ``"123b"``, or plain digits like ``"1048576"`` (interpreted
        as bytes and converted to ``"1048576b"``). If an integer, it is treated
        as a byte count and converted to ``"<N>b"``.

    Returns
    -------
    str
        A 7-Zip-compatible volume string, typically ending in ``b`` for bytes or
        using a unit suffix.

    Raises
    ------
    TypeError
        If `volume` is neither `str` nor `int`.
    ValueError
        If `volume` is invalid or non-positive.
    """
    if isinstance(volume, str):
        v = volume.strip().lower()
        if not re.fullmatch(r"\d+(\.\d+)?[kmg]?", v) and not v.endswith("b"):
            # allow '10g', '500m', '100k', '123b', '10'
            raise ValueError("Volume must be like '10g', '500m', '100k', '123b', or '10'.")
        # If user wrote plain digits, default to bytes
        if re.fullmatch(r"\d+(\.\d+)?", v):
            # 7z expects an integer for bytes; reject floats for bytes.
            if "." in v:
                raise ValueError("If passing bytes as a string, use an integer like '123b'.")
            return v + "b"
        return v

    if not isinstance(volume, int):
        raise TypeError("volume must be a str like '10g' or an int number of bytes.")
    if volume <= 0:
        raise ValueError("Volume size in bytes must be > 0.")
    return f"{volume}b"


# -----------------------------
# ZIP (Python stdlib)
# -----------------------------

def zip_by_extension(
    folder: Union[str, Path],
    ext: str,
    zip_basename: str,
    *,
    recursive: bool = False,
    fast: bool = True,
    keep_paths: bool = True,
    on_collision: str = "raise",  # 'raise' | 'rename'
) -> Path:
    """
    Create a ZIP archive containing all files with the requested extension(s).

    Parameters
    ----------
    folder
        Directory to scan.
    ext
        Extension specification(s) to include (see `_normalize_exts`), e.g.
        ``"tif"``, ``".tif"``, ``"tif,png"``. Matching is case-insensitive.
    zip_basename
        Output ZIP name without the ``.zip`` suffix. The archive is created as
        ``<folder>/<zip_basename>.zip``.
    recursive
        If ``True``, include files from all subdirectories under `folder`.
    fast
        If ``True`` (default), uses DEFLATE with ``compresslevel=1`` (fast,
        lossless). If ``False``, uses STORED (no compression).
    keep_paths
        If ``True`` (default), archive members store paths relative to `folder`
        (preserves folder structure). If ``False``, only basenames are stored,
        which can cause name collisions.
    on_collision
        Collision policy when two input files map to the same archive member
        name (most commonly when `keep_paths=False`):
        - ``"raise"``: raise `FileExistsError`
        - ``"rename"``: append ``__<n>`` before the suffix

    Returns
    -------
    pathlib.Path
        Path to the created ZIP archive.

    Raises
    ------
    FileNotFoundError
        If `folder` does not exist.
    NotADirectoryError
        If `folder` is not a directory.
    FileExistsError
        If a name collision occurs and `on_collision="raise"`.
    ValueError
        If `on_collision` is not one of ``{"raise", "rename"}``.

    Notes
    -----
    Files are added in deterministic order (case-insensitive path sort) to make
    archive contents reproducible.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    exts = set(_normalize_exts(ext))
    out_zip = folder / f"{zip_basename}.zip"

    compression = zipfile.ZIP_DEFLATED if fast else zipfile.ZIP_STORED
    kwargs = {"compresslevel": 1} if compression == zipfile.ZIP_DEFLATED else {}

    # deterministic ordering
    files = sorted(
        (p for p in _iter_files(folder, recursive) if p.suffix.lower() in exts),
        key=lambda p: str(p).lower(),
    )

    used = set()
    added = 0

    with zipfile.ZipFile(out_zip, mode="w", compression=compression, **kwargs) as zf:
        for p in files:
            if keep_paths:
                arc = str(p.relative_to(folder)).replace("\\", "/")
            else:
                arc = p.name

            if arc in used:
                if on_collision == "raise":
                    raise FileExistsError(
                        f"Archive name collision for '{arc}'. "
                        f"Set keep_paths=True or on_collision='rename'."
                    )
                elif on_collision == "rename":
                    stem, suffix = p.stem, p.suffix
                    n = 1
                    while True:
                        cand = f"{stem}__{n}{suffix}"
                        if cand not in used:
                            arc = cand
                            break
                        n += 1
                else:
                    raise ValueError("on_collision must be 'raise' or 'rename'.")

            zf.write(p, arcname=arc)
            used.add(arc)
            added += 1

    if added == 0:
        print(f"No files with extensions {sorted(exts)} found in {folder}")
    else:
        print(f"Added {added} file(s) to {out_zip}")
    return out_zip


# -----------------------------
# ZIP (7-Zip)
# -----------------------------

def zip_with_7z(
    folder: Union[str, Path],
    ext: str,
    zip_basename: str,
    *,
    recursive: bool = False,
    level: int = 1,  # 0..9
    seven_zip: Optional[Union[str, Path]] = None,
    keep_paths: bool = True,
) -> Path:
    """
    Create a ZIP archive using 7-Zip (Deflate, multi-threaded).

    This is typically faster than Python's `zipfile` for large file sets.

    Parameters
    ----------
    folder
        Directory to scan.
    ext
        Extension specification(s) to include (see `_normalize_exts`).
    zip_basename
        Output ZIP name without the ``.zip`` suffix. The archive is created as
        ``<folder>/<zip_basename>.zip``.
    recursive
        If ``True``, include files from all subdirectories under `folder`.
    level
        7-Zip compression level (0..9). ``1`` is fast; higher values trade speed
        for smaller archives.
    seven_zip
        Optional explicit path to the 7-Zip executable. If not provided, the
        executable is resolved via `_resolve_7z`.
    keep_paths
        If ``True`` (default), store paths relative to `folder` in the archive.

    Returns
    -------
    pathlib.Path
        Path to the created ZIP archive.

    Raises
    ------
    ValueError
        If ``recursive=True`` and ``keep_paths=False`` (7-Zip cannot reliably
        locate subfolder files when only basenames are provided).
    FileNotFoundError
        If `folder` does not exist or 7-Zip cannot be resolved.
    NotADirectoryError
        If `folder` is not a directory.
    RuntimeError
        If the underlying 7-Zip process fails.

    Notes
    -----
    A temporary listfile is used to avoid Windows command-length limits.
    """

    if recursive and not keep_paths:
        raise ValueError("zip_with_7z: recursive=True requires keep_paths=True (otherwise 7z can't find subfolder files).")

    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    seven_zip = _resolve_7z(seven_zip)
    out_zip = folder / f"{zip_basename}.zip"

    exts = set(_normalize_exts(ext))

    files = [p for p in _iter_files(folder, recursive) if p.suffix.lower() in exts]
    files.sort(key=lambda p: str(p).lower())
    if not files:
        print("No matching files.")
        return out_zip

    # Use a listfile to avoid Windows command-length limits.
    # Prefer relative paths from cwd=folder to ensure stored paths are relative.
    with tempfile.TemporaryDirectory() as td:
        lst = Path(td) / "files.txt"
        if keep_paths:
            rels = [str(p.relative_to(folder)).replace("\\", "/") for p in files]
        else:
            rels = [p.name for p in files]  # caller accepts collision risk
        lst.write_text("\n".join(rels), encoding="utf-8")

        cmd = [
            seven_zip, "a",
            "-tzip",
            "-mmt=on",
            f"-mx={int(level)}",
            "-mm=Deflate",
            "-y",
            str(out_zip),
            f"@{lst}",
        ]
        _run_7z(cmd, cwd=folder)

    return out_zip


# -----------------------------
# 7z (single file / folder)
# -----------------------------

def seven_zip_file(
    file_path: Union[str, Path],
    *,
    seven_zip: Optional[Union[str, Path]] = None,
    level: int = 5,
    out_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Create a ``.7z`` archive from a single file using 7-Zip.

    Parameters
    ----------
    file_path
        Path to the input file.
    seven_zip
        Optional explicit path to the 7-Zip executable. If not provided, resolved
        via `_resolve_7z`.
    level
        7-Zip compression level (0..9). Default is a balanced value (5).
    out_path
        Optional output archive path. If omitted, creates
        ``<file_path>.7z`` alongside the input file.

    Returns
    -------
    pathlib.Path
        Path to the created ``.7z`` archive.

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist or is not a file, or if 7-Zip cannot be
        resolved.
    RuntimeError
        If the underlying 7-Zip process fails.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    seven_zip = _resolve_7z(seven_zip)

    out = Path(out_path) if out_path is not None else Path(str(file_path) + ".7z")
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        seven_zip, "a",
        "-t7z",
        f"-mx={int(level)}",
        "-mmt=on",
        "-y",
        str(out),
        str(file_path),
    ]
    _run_7z(cmd)
    print(f"Created {out}")
    return out


def seven_zip_folder(
    folder_path: Union[str, Path],
    *,
    seven_zip: Optional[Union[str, Path]] = None,
    level: int = 5,
    out_dir: Optional[Union[str, Path]] = None,
    include_folder: bool = True,
) -> Path:
    """
    Create a ``.7z`` archive from a directory using 7-Zip.

    Parameters
    ----------
    folder_path
        Directory to archive (contents are included recursively).
    seven_zip
        Optional explicit path to the 7-Zip executable. If not provided, resolved
        via `_resolve_7z`.
    level
        7-Zip compression level (0..9). Default is a balanced value (5).
    out_dir
        Optional output directory. If omitted, writes ``<folder_path>.7z`` next
        to the source folder.
    include_folder
        Controls archive layout:
        - ``True``: archive root contains the folder itself
        - ``False``: archive root contains the folder contents

    Returns
    -------
    pathlib.Path
        Path to the created ``.7z`` archive.

    Raises
    ------
    NotADirectoryError
        If `folder_path` is not a directory.
    FileNotFoundError
        If 7-Zip cannot be resolved.
    RuntimeError
        If the underlying 7-Zip process fails.
    """
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise NotADirectoryError(folder_path)

    seven_zip = _resolve_7z(seven_zip)

    if out_dir is None:
        out = Path(str(folder_path) + ".7z")
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / (folder_path.name + ".7z")

    if include_folder:
        cmd = [
            seven_zip, "a",
            "-t7z",
            f"-mx={int(level)}",
            "-mmt=on",
            "-y",
            str(out),
            folder_path.name,  # relative from parent cwd
        ]
        _run_7z(cmd, cwd=folder_path.parent)
    else:
        cmd = [
            seven_zip, "a",
            "-t7z",
            f"-mx={int(level)}",
            "-mmt=on",
            "-y",
            str(out),
            "*",
        ]
        _run_7z(cmd, cwd=folder_path)

    print(f"Created {out}")
    return out


def seven_zip_folder_split(
    folder_path: Union[str, Path],
    *,
    seven_zip: Optional[Union[str, Path]] = None,
    level: int = 1,
    volume: Union[str, int] = "10g",
    include_folder: bool = True,
    out_dir: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """
    Create a split-volume ``.7z`` archive from a directory using 7-Zip.

    Output parts are named ``<name>.7z.001``, ``<name>.7z.002``, ... and can be
    reassembled/extracted by 7-Zip when all parts are present.

    Parameters
    ----------
    folder_path
        Directory to archive (contents are included recursively).
    seven_zip
        Optional explicit path to the 7-Zip executable. If not provided, resolved
        via `_resolve_7z`.
    level
        7-Zip compression level (0..9). ``1`` is fast and is often suitable for
        already-compressed scientific data formats.
    volume
        Target part size passed to 7-Zip via ``-v``. Examples: ``"10g"``,
        ``"500m"``, ``"100k"``, ``"123b"``, or an integer byte count.
    include_folder
        Controls archive layout:
        - ``True``: archive root contains the folder itself
        - ``False``: archive root contains the folder contents
    out_dir
        Optional output directory. If omitted, parts are written next to the
        source folder.

    Returns
    -------
    list of pathlib.Path
        Paths to the created part files (sorted by filename).

    Raises
    ------
    NotADirectoryError
        If `folder_path` is not a directory.
    FileNotFoundError
        If 7-Zip cannot be resolved.
    RuntimeError
        If the underlying 7-Zip process fails, or if no part files are detected
        after the command completes.
    ValueError, TypeError
        If `volume` is invalid.

    Notes
    -----
    Part files are detected using the pattern ``<archive>.7z.<NNN...>`` in the
    output directory.
    """
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise NotADirectoryError(folder_path)

    seven_zip = _resolve_7z(seven_zip)
    vol_arg = "-v" + _format_volume_size(volume)

    if out_dir is None:
        out_base = Path(str(folder_path) + ".7z")
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_base = out_dir / (folder_path.name + ".7z")

    if include_folder:
        cmd = [
            seven_zip, "a",
            "-t7z",
            f"-mx={int(level)}",
            "-mmt=on",
            "-y",
            vol_arg,
            str(out_base),
            folder_path.name,
        ]
        _run_7z(cmd, cwd=folder_path.parent)
    else:
        cmd = [
            seven_zip, "a",
            "-t7z",
            f"-mx={int(level)}",
            "-mmt=on",
            "-y",
            vol_arg,
            str(out_base),
            "*",
        ]
        _run_7z(cmd, cwd=folder_path)

    # Collect numeric parts only: <name>.7z.001, .002, ...
    part_re = re.compile(re.escape(out_base.name) + r"\.\d{3,}$")
    parts = sorted(
        [p for p in out_base.parent.iterdir() if p.is_file() and part_re.fullmatch(p.name)],
        key=lambda p: p.name,
    )
    if not parts:
        raise RuntimeError(
            f"No split parts created at {out_base.parent} "
            f"(expected {out_base.name}.001, .002, ...)"
        )
    return parts


def zip_each_file_with_extension(
    folder: Union[str, Path],
    ext: str,
    *,
    recursive: bool = False,
    fast: bool = True,
    out_dir: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """
    Create one ZIP archive per matched file.

    Each output ZIP contains exactly one member (the input file), stored using
    the file basename.

    Parameters
    ----------
    folder
        Directory to scan.
    ext
        Extension specification(s) to include (see `_normalize_exts`).
    recursive
        If ``True``, include files from all subdirectories under `folder`.
    fast
        If ``True`` (default), uses DEFLATE with ``compresslevel=1`` (fast,
        lossless). If ``False``, uses STORED (no compression).
    out_dir
        Optional output directory. If omitted, each ZIP is created next to its
        corresponding input file.

    Returns
    -------
    list of pathlib.Path
        Paths to the created ZIP archives (sorted by input file path).

    Raises
    ------
    NotADirectoryError
        If `folder` is not a directory.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    exts = set(_normalize_exts(ext))
    files = sorted(
        (p for p in _iter_files(folder, recursive) if p.suffix.lower() in exts),
        key=lambda p: str(p).lower(),
    )
    if not files:
        return []

    compression = zipfile.ZIP_DEFLATED if fast else zipfile.ZIP_STORED
    kwargs = {"compresslevel": 1} if compression == zipfile.ZIP_DEFLATED else {}

    out_dir_p = Path(out_dir) if out_dir is not None else None
    if out_dir_p is not None:
        out_dir_p.mkdir(parents=True, exist_ok=True)

    out_zips: List[Path] = []
    for f in files:
        zip_name = f.stem + ".zip"
        zpath = (out_dir_p / zip_name) if out_dir_p else (f.parent / zip_name)

        with zipfile.ZipFile(zpath, "w", compression=compression, **kwargs) as zf:
            zf.write(f, arcname=f.name)

        out_zips.append(zpath)

    return out_zips


def zip_all_files_with_extension(
    folder: Union[str, Path],
    ext: str,
    zip_basename: str,
    *,
    recursive: bool = False,
    fast: bool = True,
    keep_paths: bool = True,
    on_collision: str = "raise",
) -> Path:
    """
    Convenience wrapper for creating a single ZIP from all matched files.

    This is a thin wrapper around `zip_by_extension` and shares the same
    semantics and collision handling.

    Parameters
    ----------
    folder, ext, zip_basename, recursive, fast, keep_paths, on_collision
        See `zip_by_extension`.

    Returns
    -------
    pathlib.Path
        Path to the created ZIP archive.
    """
    return zip_by_extension(
        folder, ext, zip_basename,
        recursive=recursive,
        fast=fast,
        keep_paths=keep_paths,
        on_collision=on_collision,
    )


def zip_file_list(
    files: Sequence[Union[str, Path]],
    out_zip: Union[str, Path],
    *,
    base_dir: Optional[Union[str, Path]] = None,
    fast: bool = True,
    keep_paths: bool = False,
    on_collision: str = "raise",  # 'raise' | 'rename'
) -> Path:
    """
    Create a ZIP archive from an explicit list of files.

    Parameters
    ----------
    files
        Sequence of file paths (absolute or relative). All paths must resolve to
        existing files.
    out_zip
        Output ZIP path. Parent directories are created if required.
    base_dir
        Base directory used to compute archive member paths when `keep_paths=True`.
        Must be provided in that case, and all files must be within `base_dir`.
    fast
        If ``True`` (default), uses DEFLATE with ``compresslevel=1`` (fast,
        lossless). If ``False``, uses STORED (no compression).
    keep_paths
        If ``True``, store paths relative to `base_dir` (preserves structure).
        If ``False`` (default), store basenames only.
    on_collision
        Collision policy when multiple input files map to the same archive member
        name (primarily when `keep_paths=False`):
        - ``"raise"``: raise `FileExistsError`
        - ``"rename"``: append ``__<n>`` before the suffix

    Returns
    -------
    pathlib.Path
        Path to the created ZIP archive.

    Raises
    ------
    FileNotFoundError
        If any input file does not exist.
    ValueError
        If `keep_paths=True` and `base_dir` is not provided, or if `on_collision`
        is invalid.
    FileExistsError
        If a name collision occurs and `on_collision="raise"`.

    Notes
    -----
    Inputs are archived in deterministic order (case-insensitive path sort).
    Archive member paths use forward slashes to be platform-independent.
    """
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    resolved = []
    for f in files:
        p = Path(f)
        if not p.is_file():
            raise FileNotFoundError(p)
        resolved.append(p)

    # deterministic order
    resolved.sort(key=lambda p: str(p).lower())

    compression = zipfile.ZIP_DEFLATED if fast else zipfile.ZIP_STORED
    kwargs = {"compresslevel": 1} if compression == zipfile.ZIP_DEFLATED else {}

    base_dir_p = Path(base_dir).resolve() if base_dir is not None else None
    used = set()

    with zipfile.ZipFile(out_zip, "w", compression=compression, **kwargs) as zf:
        for p in resolved:
            if keep_paths:
                if base_dir_p is None:
                    raise ValueError("base_dir must be provided when keep_paths=True.")
                arc = str(p.resolve().relative_to(base_dir_p)).replace("\\", "/")
            else:
                arc = p.name

            if arc in used:
                if on_collision == "raise":
                    raise FileExistsError(
                        f"Archive name collision for '{arc}'. "
                        f"Use keep_paths=True or on_collision='rename'."
                    )
                elif on_collision == "rename":
                    stem, suffix = p.stem, p.suffix
                    n = 1
                    while True:
                        cand = f"{stem}__{n}{suffix}"
                        if cand not in used:
                            arc = cand
                            break
                        n += 1
                else:
                    raise ValueError("on_collision must be 'raise' or 'rename'.")

            zf.write(p, arcname=arc)
            used.add(arc)

    return out_zip
