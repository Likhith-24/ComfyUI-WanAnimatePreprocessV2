# Keypoint-JSON contract (schema_version 1)

The interchange format this pack writes so **another repo can consume its
keypoints without importing it**. That constraint is the whole design driver:
`ComfyUI-WanAnimatePreprocessV2` is Apache-2.0, consumers may be under other
licences, and cross-repo runtime imports are forbidden (R10). A data contract is
the only coupling allowed.

Reference implementation: [`keypoint_contract.py`](../keypoint_contract.py).
A consumer needs **plain `json.loads` and nothing else** — but may copy
`validate_keypoint_export` if it wants the checks.

## Shape

```json
{
  "schema_version": 1,
  "pack_commit": "a1b2c3d",
  "image_w": 1920,
  "image_h": 1080,
  "source": "vitpose",
  "frames": [
    {
      "frame_index": 0,
      "keypoints": {
        "nose":      [960.0, 412.0, 0.93],
        "left_eye":  [938.0, 400.5, 0.88],
        "right_eye": null
      }
    }
  ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | Increments on any breaking change. A consumer must refuse a version it does not know. |
| `pack_commit` | string | Short git SHA of the producing pack, or `"unknown"`. Provenance for a bug report. |
| `image_w` / `image_h` | int > 0 | Pixel dimensions the coordinates refer to. |
| `source` | string | `vitpose` \| `pose3d_nlf` \| `gaze` \| `expression`. |
| `frames` | array | One entry per frame, in order. |
| `frames[].frame_index` | int | 0-based, contiguous. |
| `frames[].keypoints` | object | `name → [x, y, confidence]` or `null`. |

## The four rules that make it safe to consume

**1. Missing keypoints are `null`, never omitted.** Every frame carries the same
name set, so a consumer indexes by name without a `KeyError` and without
per-frame guards. `null` also keeps "not detected" distinguishable from
"detected at the origin" — `[0, 0]` is a real coordinate, not a sentinel.

**2. `confidence` is always in `[0, 1]`.** Detectors emitting logits or
percentages must normalise before export. The builder **raises** rather than
writing an out-of-range value, because a consumer in another repo cannot know
your scale and would silently misread a `95.0` as saturated confidence.

**3. Coordinates are in pixels** of the stated `image_w`/`image_h`, never
normalised. A normalised export looks entirely plausible and misplaces every
point by a factor of the image size.

**4. `NaN` never appears.** It is not valid JSON; a detector using `NaN` for
"no detection" is converted to `null` at export.

## Consuming it

```python
import json

with open("keypoints.json", encoding="utf-8") as fh:
    data = json.load(fh)

if data["schema_version"] != 1:
    raise SystemExit(f"unsupported schema_version {data['schema_version']}")

for frame in data["frames"]:
    nose = frame["keypoints"]["nose"]
    if nose is None:          # rule 1: always present, may be null
        continue
    x, y, conf = nose         # pixels, conf in [0,1]
```

## Producing it

```python
from keypoint_contract import build_keypoint_export, dumps_keypoint_export

payload = dumps_keypoint_export(
    frames=[{"nose": [960.0, 412.0, 0.93], "left_eye": None}, ...],
    image_w=1920, image_h=1080, source="vitpose",
)
```

`build_keypoint_export` validates as it builds and raises
`KeypointContractError` on any violation. Writing a malformed file is worse than
writing none — the consumer discovers it later, in a different repo.

## Versioning

Adding an **optional** top-level field is non-breaking and does not bump the
version. Renaming or removing a field, changing a coordinate convention, or
changing the meaning of `null` **is** breaking and must bump `schema_version`,
with consumers refusing unknown versions.

Covered by [`tests/test_keypoint_contract.py`](../tests/test_keypoint_contract.py),
which exercises the format through `json.loads` rather than through pack
objects — the same path a real consumer takes.
