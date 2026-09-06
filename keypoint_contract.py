"""Keypoint-JSON export contract — the cross-pack data interchange format.

Pure stdlib + numpy. Deliberately importable with NO ComfyUI present, because
the whole point is that a CONSUMER can read this format without importing this
pack. That is what makes it a contract rather than an API: the future
animal-retarget consumer lives in a different repo under a different licence
(this pack is Apache-2.0), and R10 forbids cross-repo runtime imports.

Format (one object, whole clip):

    {
      "schema_version": 1,
      "pack_commit": "<sha or 'unknown'>",
      "image_w": 1920,
      "image_h": 1080,
      "source": "vitpose",
      "frames": [
        {"frame_index": 0,
         "keypoints": {"nose": [960.0, 412.0, 0.93],
                       "left_eye": null}},
        ...
      ]
    }

Rules that make it safe to consume:

* A keypoint is ``[x, y, confidence]`` or ``null``. **Missing keypoints are
  null, never omitted** — so a consumer can index by name without a KeyError,
  and "absent" is distinguishable from "present at 0,0", which is a real
  detection value and not a sentinel.
* ``confidence`` is always in [0, 1]. Detectors that emit logits or percentages
  must normalise before calling; :func:`build_keypoint_export` validates and
  raises rather than writing an out-of-range value a consumer would misread.
* Coordinates are in PIXELS of the stated ``image_w``/``image_h``, never
  normalised, so a consumer never has to guess a convention.
* ``schema_version`` is an integer that increments on any breaking change.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

__all__ = [
    "SCHEMA_VERSION",
    "VALID_SOURCES",
    "KeypointContractError",
    "build_keypoint_export",
    "validate_keypoint_export",
    "pack_commit",
]

SCHEMA_VERSION = 1

#: Which detector produced the points. A consumer needs this because the name
#: sets differ — a vitpose "nose" and a gaze "nose" are not interchangeable.
VALID_SOURCES = ("vitpose", "pose3d_nlf", "gaze", "expression")


class KeypointContractError(ValueError):
    """Raised when an export would violate the contract.

    Hard error by design: writing a malformed file is worse than not writing
    one, because the consumer discovers it later and in a different repo.
    """


def pack_commit(default: str = "unknown") -> str:
    """Best-effort git SHA of this pack, for provenance in the export."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = (out.stdout or "").strip()
        return sha or default
    except Exception:
        return default


def _coerce_point(name, value, frame_index):
    """Return [x, y, conf] floats, or None. Raises on anything unreadable."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise KeypointContractError(
            f"frame {frame_index} keypoint {name!r}: need at least [x, y]; got {value!r}"
        )
    x, y = float(arr[0]), float(arr[1])
    conf = float(arr[2]) if arr.size >= 3 else 1.0
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(conf)):
        # A NaN coordinate means "not detected". Express that as null rather
        # than writing NaN, which is not valid JSON and which json.loads on the
        # consumer side would either reject or silently turn into a float.
        return None
    if not (0.0 <= conf <= 1.0):
        raise KeypointContractError(
            f"frame {frame_index} keypoint {name!r}: confidence {conf} outside [0,1]. "
            "Normalise logits/percentages before export — a consumer in another "
            "repo cannot know your scale."
        )
    return [x, y, conf]


def build_keypoint_export(
    frames,
    *,
    image_w,
    image_h,
    source,
    commit=None,
):
    """Build the export dict.

    Args:
        frames: sequence of per-frame ``{name: point_or_None}`` mappings, in
            frame order. ``point`` is ``[x, y]`` or ``[x, y, confidence]``.
        image_w/image_h: pixel dimensions the coordinates refer to.
        source: one of :data:`VALID_SOURCES`.
        commit: override the recorded pack commit (tests pin it).

    Returns:
        A JSON-serialisable dict.

    Raises:
        KeypointContractError: on any contract violation.
    """
    if source not in VALID_SOURCES:
        raise KeypointContractError(
            f"source {source!r} is not one of {list(VALID_SOURCES)}"
        )
    w, h = int(image_w), int(image_h)
    if w <= 0 or h <= 0:
        raise KeypointContractError(f"image dimensions must be positive; got {w}x{h}")

    frames = list(frames)
    if not frames:
        raise KeypointContractError("no frames to export")

    # Every frame must advertise the SAME name set — a consumer indexes by name
    # and a frame that quietly drops a key turns into a KeyError far away.
    names = None
    out_frames = []
    for i, mapping in enumerate(frames):
        if not isinstance(mapping, dict):
            raise KeypointContractError(f"frame {i}: expected a dict, got {type(mapping).__name__}")
        keys = set(mapping)
        if names is None:
            names = keys
        elif keys != names:
            missing = sorted(names - keys)
            extra = sorted(keys - names)
            raise KeypointContractError(
                f"frame {i}: keypoint names differ from frame 0 "
                f"(missing {missing}, unexpected {extra}). Emit null for an "
                "undetected point instead of omitting it."
            )
        out_frames.append(
            {
                "frame_index": i,
                "keypoints": {
                    n: _coerce_point(n, mapping[n], i) for n in sorted(mapping)
                },
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "pack_commit": pack_commit() if commit is None else str(commit),
        "image_w": w,
        "image_h": h,
        "source": str(source),
        "frames": out_frames,
    }


def validate_keypoint_export(obj):
    """Validate a decoded export. Returns it; raises on violation.

    Written so a CONSUMER can call it too — it imports nothing from this pack.
    """
    if not isinstance(obj, dict):
        raise KeypointContractError(f"export must be an object, got {type(obj).__name__}")
    for key in ("schema_version", "pack_commit", "image_w", "image_h", "source", "frames"):
        if key not in obj:
            raise KeypointContractError(f"export missing required field {key!r}")
    if obj["schema_version"] != SCHEMA_VERSION:
        raise KeypointContractError(
            f"schema_version {obj['schema_version']} != supported {SCHEMA_VERSION}"
        )
    if obj["source"] not in VALID_SOURCES:
        raise KeypointContractError(f"unknown source {obj['source']!r}")

    names = None
    for fr in obj["frames"]:
        for key in ("frame_index", "keypoints"):
            if key not in fr:
                raise KeypointContractError(f"frame missing {key!r}")
        keys = set(fr["keypoints"])
        if names is None:
            names = keys
        elif keys != names:
            raise KeypointContractError(
                f"frame {fr['frame_index']} keypoint names differ from frame 0"
            )
        for n, pt in fr["keypoints"].items():
            if pt is None:
                continue
            if not isinstance(pt, list) or len(pt) != 3:
                raise KeypointContractError(
                    f"frame {fr['frame_index']} {n!r}: expected [x, y, confidence] or null"
                )
            if not 0.0 <= float(pt[2]) <= 1.0:
                raise KeypointContractError(
                    f"frame {fr['frame_index']} {n!r}: confidence {pt[2]} outside [0,1]"
                )
    return obj


def dumps_keypoint_export(*args, **kwargs) -> str:
    """Convenience: build and serialise in one call."""
    return json.dumps(build_keypoint_export(*args, **kwargs), separators=(",", ":"))
