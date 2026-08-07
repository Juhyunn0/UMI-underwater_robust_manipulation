#!/usr/bin/env python3
"""
profile.py — YAML operating profiles for `c3_collect.py`.

A dive is a dozen flags long, and the same dozen every time. A profile is that
dozen in a file:

    ./c3 collect --profile research_near              # instead of the flag wall
    ./c3 collect --profile research_near --fps 30     # the flag still wins

WHERE A PROFILE SITS. Lowest to highest:

    StreamConfig defaults      (config.py:175)
      < mode defaults          (c3_collect.py RESEARCH_DEFAULTS / REVIEW_DEFAULTS)
      < HOST_DEPTH_DEFAULTS    (only with --depth-source host)
      < profile YAML           (parents first, then children, via `extends:`)
      < flags typed on the command line

The last step is why `c3_collect.explicit_dests` exists: half the flags have a
real default rather than None, so "was it typed" cannot be answered by comparing
values. See that function.

WHAT A PROFILE CAN SET. Everything the command line can, *plus* the fifteen
StreamConfig fields it never exposed — depth_preset, median, confidence,
subpixel, lr_check, pair_mode, mjpeg_quality and friends. The SCHEMA table below
is the whole contract: section -> key -> where the value lands. Nothing is
inferred from the key's name, which is what makes a typo produce a sentence
instead of a silent no-op.

WHY THIS DUPLICATES `umi_handheld/record.py:65-105`. That module has the same
`extends:` loader, and it is not imported here: it pulls in cv2, numpy, yaml and
the whole handheld camera-model stack to lend twelve lines of dict merging. The
two implementations agree on semantics deliberately (child-file-relative
`extends:`, lists replaced wholesale, schema checked after the merge), and they
differ in two places on purpose — this one validates each file's own keys
*before* merging, so an error names the file the typo is in, and it keeps
`_extends_chain` for provenance rather than computing it and dropping it.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Callable, NamedTuple

from c3_camera import config as C

SCHEMA_VERSION = "c3_collect_profile/1"

REPO = Path(__file__).resolve().parents[1]
PROFILE_DIR = Path(__file__).resolve().parent / "profiles"


class ProfileError(ValueError):
    """A profile file is malformed, and the message says which file and why."""


# =============================================================================
# Coercions — the file and the flag must never disagree about what "1/4" means,
# so the three interesting ones call the same parsers argparse does.
# =============================================================================
def _reject_bool(v: Any, what: str) -> None:
    """YAML's `off`/`on`/`no`/`yes` are booleans, and several of our string
    values look exactly like them. `median: off` reads as False, and
    MEDIAN_FILTERS has an "off" *key*, so the mistake is silent without this."""
    if isinstance(v, bool):
        raise ProfileError(
            f"{what}: YAML read {str(v).lower()!r} as the boolean {v}, but this "
            f"setting takes a string — quote it, e.g. \"off\"")


def _str(v, what):
    _reject_bool(v, what)
    return str(v)


def _int(v, what):
    _reject_bool(v, what)
    return int(v)


def _float(v, what):
    _reject_bool(v, what)
    return float(v)


def _bool(v, what):
    if not isinstance(v, bool):
        raise ProfileError(f"{what}: expected true or false, got {v!r}")
    return v


def _opt_int(v, what):
    return None if v is None else _int(v, what)


def _opt_float(v, what):
    return None if v is None else _float(v, what)


def _path(v, what):
    return Path(_str(v, what)).expanduser()


def _fraction(v, what):
    if v is None:
        return None
    try:
        return C._parse_fraction(_str(v, what))
    except Exception as e:                                   # noqa: BLE001
        raise ProfileError(f"{what}: {e}") from None


def _size(v, what):
    if v is None:
        return None
    try:
        return C._parse_size(_str(v, what))
    except Exception as e:                                   # noqa: BLE001
        raise ProfileError(f"{what}: {e}") from None


def _streams(v, what):
    items = v if isinstance(v, (list, tuple)) else [v]
    text = ",".join(_str(x, what) for x in items)
    try:
        return C._parse_streams(text)
    except Exception as e:                                   # noqa: BLE001
        raise ProfileError(f"{what}: {e}") from None


# =============================================================================
# The schema
# =============================================================================
class Field(NamedTuple):
    kind: str                    # "dest" -> argparse Namespace | "stream" -> StreamConfig
    target: str
    coerce: Callable[[Any, str], Any]
    invert: bool = False         # YAML says `mp4: true`, the dest is `no_mp4`


def _d(target, coerce, invert=False):
    return Field("dest", target, coerce, invert)


def _s(target, coerce):
    return Field("stream", target, coerce, False)


# Keys are positive concepts. The three inversions (`imu.enable`,
# `recording.mp4`, `display.enable`) are owned here so no file has to spell a
# double negative.
SCHEMA: dict[str, dict[str, Field]] = {
    "connection": {
        "ip":                       _d("ip", _str),
        "mxid":                     _d("mxid", _str),
    },
    "camera": {
        "streams":                  _d("streams", _streams),
        "color_res":                _d("color_res", _str),
        "isp_scale":                _d("isp_scale", _fraction),
        "color_encode":             _d("color_encode", _str),
        "color_wire":               _d("color_wire", _str),
        "video_bitrate_kbps":       _d("video_bitrate_kbps", _int),
        "video_keyframe_frequency": _d("video_keyframe_frequency", _int),
        "fps":                      _d("fps", _float),
        "mjpeg_quality":            _s("mjpeg_quality", _int),
    },
    "stereo": {
        "mono_res":                 _d("mono_res", _str),
        "mono_source":              _d("mono_source", _str),
        "mono_encode":              _d("mono_encode", _str),
        "mono_quality":             _d("mono_quality", _int),
        "depth_size":               _d("depth_size", _size),
        "extended":                 _d("extended", _bool),
        "mono_fps":                 _s("mono_fps", _opt_float),
        "depth_align":              _s("depth_align", _str),
        "depth_match_color":        _s("depth_match_color", _bool),
        "depth_preset":             _s("depth_preset", _str),
        "subpixel":                 _s("subpixel", _bool),
        "lr_check":                 _s("lr_check", _bool),
        "median":                   _s("median", _str),
        "confidence":               _s("confidence", _opt_int),
    },
    "imu": {
        "enable":                   _d("no_camera_imu", _bool, invert=True),
        "rate_hz":                  _d("imu_rate", _int),
        "batch":                    _d("imu_batch", _int),
        "rotation_vector":          _d("imu_rotation_vector", _bool),
        "magnetometer":             _d("imu_magnetometer", _bool),
        "calibrated":               _d("imu_calibrated", _bool),
    },
    "link": {
        "queue_size":               _d("queue_size", _int),
        "queue_blocking":           _s("queue_blocking", _bool),
        "xlink_chunk":              _s("xlink_chunk", _int),
        "device_queue_size":        _s("device_queue_size", _int),
        "small_pools":              _s("small_pools", _bool),
        "pair_mode":                _s("pair_mode", _str),
        "pair_tolerance_ms":        _s("pair_tolerance_ms", _opt_float),
    },
    "depth_source": {
        "where":                    _d("depth_source", _str),
        "host_alpha":               _d("host_alpha", _float),
        "host_num_disparities":     _d("host_num_disparities", _int),
        "host_block_size":          _d("host_block_size", _int),
        "host_downscale":           _d("host_downscale", _int),
        "align_color":              _d("host_align_color", _bool),
        "workers":                  _d("host_depth_workers", _int),
        "queue":                    _d("host_depth_queue", _int),
        "cv_threads":               _d("host_cv_threads", _opt_int),
    },
    "mavlink": {
        "transport":                _d("mavlink_transport", _str),
        "connection":               _d("mavlink", _str),
        "rest_url":                 _d("mavlink_rest_url", _str),
        "rate_hz":                  _d("mavlink_rate", _int),
        "set_rates":                _d("mavlink_set_rates", _bool),
        "wait_s":                   _d("mavlink_wait", _float),
        "setup_endpoint":           _d("setup_endpoint", _bool),
    },
    "recording": {
        "out":                      _d("out", _path),
        "notes":                    _d("notes", _str),
        "record_now":               _d("record_now", _bool),
        "writer_threads":           _d("writer_threads", _int),
        "writer_queue":             _d("writer_queue", _int),
        "png_compression":          _d("png_compression", _int),
        "mp4":                      _d("no_mp4", _bool, invert=True),
        "extract_encoded_rgb":      _d("extract_encoded_rgb", _bool),
        "min_free_gb":              _d("min_free_gb", _float),
    },
    "display": {
        "enable":                   _d("no_display", _bool, invert=True),
        "scale":                    _d("scale", _float),
        "min_mm":                   _d("min_mm", _float),
        "max_mm":                   _d("max_mm", _float),
        "print_every_s":            _d("print_every", _float),
        "duration_s":               _d("duration", _opt_float),
    },
}

# Top-level keys that are not sections.
_META_KEYS = {"schema", "extends", "profile", "mode"}

# Rejected by name rather than by absence, because both would half-work and
# then surprise someone: `profile:` is the file's own label (a second `extends:`
# by the back door if it selected anything), and `dry_run:` is a file that
# quietly turns a recording command into a no-op.
_FORBIDDEN = {
    "dry_run": "--dry-run is a thing you do once, not a property of a dive",
    "profile": "`profile:` at the top level is this file's own name; to build on "
               "another file use `extends:`",
}

_ALL_KEYS = {f"{section}.{key}": field
             for section, keys in SCHEMA.items() for key, field in keys.items()}


# =============================================================================
# Loading
# =============================================================================
def resolve(name: str) -> Path:
    """`--profile` argument -> a file. A bare NAME means profiles/NAME.yaml."""
    p = Path(name).expanduser()
    if "/" in name:
        return p if p.is_absolute() else (REPO / p)
    stem = p.stem if p.suffix in (".yaml", ".yml") else name
    return PROFILE_DIR / f"{stem}.yaml"


def _deep_merge(base: dict, over: dict) -> dict:
    """over wins, but dicts merge key by key so a profile states only its deltas.

    Lists are replaced wholesale, never concatenated: `streams: [color, depth]`
    in a child has to mean exactly those two.
    """
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _validate_file(cfg: dict, path: Path) -> None:
    """Everything checkable about ONE file, before it is merged with anything.

    Before, so that a typo is reported against the file it is actually in rather
    than against whichever child happened to be loaded last.
    """
    where = _rel(path)
    if not isinstance(cfg, dict):
        raise ProfileError(f"{where}: expected a mapping at the top level, "
                           f"got {type(cfg).__name__}")
    # Checked before the keys, so pointing --profile at a config belonging to
    # another tool (configs/pipeline.yaml is umi_handheld_pipeline/1) gets one
    # sentence rather than a walk through its unknown keys. A file that declares
    # no schema at all is fine here and is caught after the merge, so a child can
    # inherit its parent's.
    declared = cfg.get("schema")
    if declared is not None and declared != SCHEMA_VERSION:
        raise ProfileError(
            f"{where}: this is not a c3_collect profile — expected "
            f"`schema: {SCHEMA_VERSION}`, found {declared!r}")
    for key in cfg:
        if key in _META_KEYS or key in SCHEMA:
            continue
        if key in _FORBIDDEN:
            raise ProfileError(f"{where}: `{key}:` is not settable — "
                               f"{_FORBIDDEN[key]}")
        near = difflib.get_close_matches(key, sorted(_META_KEYS | set(SCHEMA)), 1)
        hint = f"; did you mean `{near[0]}:`?" if near else ""
        raise ProfileError(f"{where}: unknown top-level key `{key}:`. Sections "
                           f"are {', '.join(sorted(SCHEMA))}{hint}")

    if "mode" in cfg and cfg["mode"] not in ("research", "review"):
        raise ProfileError(f"{where}: mode must be research or review, "
                           f"got {cfg['mode']!r}")

    for section, body in cfg.items():
        if section not in SCHEMA:
            continue
        if not isinstance(body, dict):
            raise ProfileError(f"{where}: section `{section}:` must be a mapping, "
                               f"got {type(body).__name__}")
        for key, value in body.items():
            if key in SCHEMA[section]:
                # Coerce now, so a bad value also names its own file.
                SCHEMA[section][key].coerce(value, f"{where}: {section}.{key}")
                continue
            elsewhere = [s for s in SCHEMA if key in SCHEMA[s]]
            if elsewhere:
                raise ProfileError(
                    f"{where}: `{key}` belongs in section `{elsewhere[0]}:`, "
                    f"not `{section}:`")
            near = difflib.get_close_matches(key, sorted(SCHEMA[section]), 1)
            hint = f"; did you mean `{near[0]}`?" if near else ""
            raise ProfileError(f"{where}: unknown key `{key}` in section "
                               f"`{section}:`{hint}")


def load(name_or_path, _seen=None) -> dict:
    """Read a profile, resolving `extends:` parents first. Returns the merged
    mapping plus three bookkeeping keys: `_path`, `_extends_chain`, `_origin`.

    `_origin` maps "section.key" -> the file that last set it, which is what
    lets --dry-run print where every value came from.
    """
    try:
        import yaml
    except ImportError:                                      # pragma: no cover
        raise ProfileError(
            "--profile needs PyYAML, which is not installed in this "
            "interpreter.\n  pip install 'PyYAML>=6'") from None

    path = name_or_path if isinstance(name_or_path, Path) else resolve(str(name_or_path))
    path = path.resolve()
    _seen = _seen or []
    if path in _seen:
        chain = " -> ".join(p.name for p in _seen + [path])
        raise ProfileError(f"extends: cycle {chain}")
    if not path.exists():
        raise ProfileError(
            f"no such profile: {path}\n  available: "
            f"{', '.join(sorted(p.stem for p in PROFILE_DIR.glob('*.yaml'))) or '(none)'}")

    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    parent_name = cfg.pop("extends", None)
    _validate_file(cfg, path)

    origin = {f"{s}.{k}": _rel(path)
              for s, body in cfg.items() if s in SCHEMA for k in body}
    if "mode" in cfg:
        origin["mode"] = _rel(path)
    chain: list[str] = []

    if parent_name:
        ppath = Path(parent_name).expanduser()
        if not ppath.is_absolute():
            ppath = path.parent / ppath
        base = load(ppath, _seen + [path])
        chain = base.pop("_extends_chain", []) + [base.pop("_path")]
        origin = {**base.pop("_origin", {}), **origin}
        cfg = _deep_merge(base, cfg)

    if cfg.get("schema") != SCHEMA_VERSION:
        raise ProfileError(
            f"{_rel(path)}: this is not a c3_collect profile — expected "
            f"`schema: {SCHEMA_VERSION}`, found {cfg.get('schema')!r}")

    cfg["_path"] = _rel(path)
    cfg["_extends_chain"] = chain
    cfg["_origin"] = origin
    return cfg


# =============================================================================
# Applying
# =============================================================================
def apply(ns, cfg: dict, explicit: set[str]) -> dict:
    """Push a loaded profile onto an argparse Namespace, minus what was typed.

    Mutates `ns` and returns the StreamConfig fields that have no flag to land
    on, for the caller to fold into its config dict. `mode` is deliberately NOT
    handled here — the caller has to resolve it *before* choosing its defaults
    dict, or a profile's `mode:` would only change the banner.
    """
    stream_extra: dict[str, Any] = {}
    for section, body in cfg.items():
        if section not in SCHEMA:
            continue
        for key, value in body.items():
            field = SCHEMA[section][key]
            coerced = field.coerce(value, f"{cfg.get('_path', '?')}: {section}.{key}")
            if field.invert:
                coerced = not coerced
            if field.kind == "stream":
                stream_extra[field.target] = coerced
            elif field.target not in explicit:
                setattr(ns, field.target, coerced)
    return stream_extra


def mode_of(cfg: dict | None, ns, explicit: set[str]) -> str:
    """`--mode` under the same three-level rule, resolved early enough to matter."""
    if "mode" in explicit or cfg is None:
        return ns.mode
    return cfg.get("mode", ns.mode)


def origin_of(dotted: str, cfg: dict | None, explicit: set[str],
              mode_keys: set[str], mode: str) -> str:
    """Where a setting's value came from, for the --dry-run table."""
    field = _ALL_KEYS.get(dotted)
    if field is not None and field.kind == "dest" and field.target in explicit:
        return "cli"
    if cfg is not None and dotted in cfg.get("_origin", {}):
        return f"profile:{cfg['_origin'][dotted]}"
    if field is not None and field.target in mode_keys:
        return f"mode:{mode}"
    return "default"
