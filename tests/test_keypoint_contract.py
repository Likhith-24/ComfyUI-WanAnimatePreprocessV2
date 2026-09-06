"""Keypoint-JSON contract: round trip, null handling, confidence range.

Every test carries an INVARIANT line. The point of this format is that a
consumer in ANOTHER repo (different licence, no cross-repo imports allowed)
can read it with plain json.loads, so the tests exercise it that way.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from keypoint_contract import (  # noqa: E402
    SCHEMA_VERSION,
    VALID_SOURCES,
    KeypointContractError,
    build_keypoint_export,
    dumps_keypoint_export,
    validate_keypoint_export,
)

NAMES = ("nose", "left_eye", "right_eye")


def _frames(n=3, drop_left_eye_on=()):
    out = []
    for i in range(n):
        out.append(
            {
                "nose": [100.0 + i, 200.0, 0.9],
                "left_eye": None if i in drop_left_eye_on else [90.0, 190.0, 0.8],
                "right_eye": [110.0, 190.0, 0.75],
            }
        )
    return out


def test_round_trip_through_plain_json():
    # INVARIANT: the export survives json.dumps -> json.loads unchanged. This is
    # the whole contract — a consumer uses stdlib json, not this pack.
    obj = build_keypoint_export(_frames(), image_w=640, image_h=480,
                                source="vitpose", commit="abc1234")
    rt = json.loads(json.dumps(obj))
    assert rt == obj
    assert validate_keypoint_export(rt) is rt


def test_required_fields_all_present():
    # INVARIANT: the brief's field list, exactly.
    obj = build_keypoint_export(_frames(), image_w=640, image_h=480,
                                source="vitpose", commit="abc1234")
    for key in ("schema_version", "pack_commit", "image_w", "image_h", "source", "frames"):
        assert key in obj, f"missing {key}"
    assert obj["schema_version"] == SCHEMA_VERSION
    assert obj["image_w"] == 640 and obj["image_h"] == 480
    assert obj["frames"][0]["frame_index"] == 0
    assert obj["frames"][-1]["frame_index"] == len(obj["frames"]) - 1


def test_missing_keypoint_is_null_never_omitted():
    # INVARIANT: the rule that makes the format indexable. An undetected point
    # must be present as null; omitting it would make a consumer KeyError, and
    # substituting [0,0] would be indistinguishable from a real corner detection.
    obj = build_keypoint_export(_frames(drop_left_eye_on=(1,)), image_w=640,
                                image_h=480, source="vitpose", commit="x")
    for fr in obj["frames"]:
        assert set(fr["keypoints"]) == set(NAMES), "a frame dropped a keypoint name"
    assert obj["frames"][1]["keypoints"]["left_eye"] is None
    assert obj["frames"][0]["keypoints"]["left_eye"] is not None


def test_nan_coordinate_becomes_null_not_nan():
    # INVARIANT: NaN is not valid JSON. A detector emitting NaN for "no
    # detection" must serialise as null, or json.dumps writes bare NaN and a
    # strict consumer parser rejects the whole file.
    frames = _frames(1)
    frames[0]["nose"] = [float("nan"), float("nan"), 0.0]
    obj = build_keypoint_export(frames, image_w=64, image_h=64,
                                source="gaze", commit="x")
    assert obj["frames"][0]["keypoints"]["nose"] is None
    assert "NaN" not in json.dumps(obj)


def test_confidence_out_of_range_is_rejected():
    # INVARIANT: confidence in [0,1]. A detector emitting percentages or logits
    # must normalise; a consumer in another repo cannot know the scale.
    frames = _frames(1)
    frames[0]["nose"] = [10.0, 10.0, 95.0]       # percentage, not a probability
    with pytest.raises(KeypointContractError) as exc:
        build_keypoint_export(frames, image_w=64, image_h=64,
                              source="vitpose", commit="x")
    assert "confidence" in str(exc.value) and "0,1" in str(exc.value).replace(" ", "")


def test_inconsistent_names_between_frames_rejected():
    # INVARIANT: every frame advertises the same name set, so a consumer can
    # index by name across the clip without per-frame guards.
    frames = _frames(2)
    del frames[1]["left_eye"]
    with pytest.raises(KeypointContractError) as exc:
        build_keypoint_export(frames, image_w=64, image_h=64,
                              source="vitpose", commit="x")
    assert "left_eye" in str(exc.value)
    assert "null" in str(exc.value), "error should name the correct fix"


@pytest.mark.parametrize("source", VALID_SOURCES)
def test_every_advertised_source_is_accepted(source):
    # INVARIANT: the documented source list is the accepted one — no entry is
    # advertised but rejected.
    obj = build_keypoint_export(_frames(1), image_w=64, image_h=64,
                                source=source, commit="x")
    assert obj["source"] == source


def test_unknown_source_rejected():
    # INVARIANT: an unlisted source is a finding, not a silent pass-through.
    with pytest.raises(KeypointContractError):
        build_keypoint_export(_frames(1), image_w=64, image_h=64,
                              source="not_a_detector", commit="x")


def test_coordinates_are_pixels_not_normalised():
    # INVARIANT: coordinates are in pixels of the stated dimensions. Pins the
    # convention — a normalised export would look plausible and misplace every
    # point by a factor of the image size.
    obj = build_keypoint_export(_frames(1), image_w=640, image_h=480,
                                source="vitpose", commit="x")
    x, y, _c = obj["frames"][0]["keypoints"]["nose"]
    assert x > 1.0 and y > 1.0, "coordinates look normalised, not pixels"


def test_two_element_point_defaults_to_full_confidence():
    # INVARIANT: [x, y] without a confidence is accepted and recorded as 1.0,
    # so a detector with no score does not have to invent one.
    frames = [{"nose": [5.0, 6.0]}]
    obj = build_keypoint_export(frames, image_w=64, image_h=64,
                                source="expression", commit="x")
    assert obj["frames"][0]["keypoints"]["nose"] == [5.0, 6.0, 1.0]


def test_hostile_inputs_rejected():
    # INVARIANT: malformed input is a finding, never a malformed file on disk.
    with pytest.raises(KeypointContractError):
        build_keypoint_export([], image_w=64, image_h=64, source="vitpose")
    with pytest.raises(KeypointContractError):
        build_keypoint_export([{"nose": [1.0]}], image_w=64, image_h=64, source="vitpose")
    with pytest.raises(KeypointContractError):
        build_keypoint_export(_frames(1), image_w=0, image_h=64, source="vitpose")
    with pytest.raises(KeypointContractError):
        build_keypoint_export(["not a dict"], image_w=64, image_h=64, source="vitpose")


def test_validator_catches_a_hand_edited_file():
    # INVARIANT: validate_keypoint_export is usable by a CONSUMER on a file it
    # did not produce — it must reject corruption, not assume good faith.
    obj = build_keypoint_export(_frames(2), image_w=64, image_h=64,
                                source="vitpose", commit="x")
    obj["frames"][1]["keypoints"]["nose"] = [1.0, 2.0, 7.0]   # bad confidence
    with pytest.raises(KeypointContractError):
        validate_keypoint_export(obj)

    obj2 = build_keypoint_export(_frames(1), image_w=64, image_h=64,
                                 source="vitpose", commit="x")
    del obj2["image_w"]
    with pytest.raises(KeypointContractError):
        validate_keypoint_export(obj2)


def test_consumer_needs_no_pack_import():
    # INVARIANT: the format is readable with stdlib json alone. Parsing the
    # serialised string here without touching any pack symbol is the proof.
    payload = dumps_keypoint_export(_frames(2), image_w=64, image_h=64,
                                    source="pose3d_nlf", commit="x")
    data = json.loads(payload)
    assert data["source"] == "pose3d_nlf"
    nose = data["frames"][0]["keypoints"]["nose"]
    assert isinstance(nose, list) and len(nose) == 3
    assert 0.0 <= nose[2] <= 1.0


def test_numpy_inputs_accepted():
    # INVARIANT: detectors hand back numpy rows; those must not need converting
    # at every call site, and must serialise as plain floats not numpy scalars.
    frames = [{"nose": np.array([1.5, 2.5, 0.5], dtype=np.float32)}]
    payload = dumps_keypoint_export(frames, image_w=64, image_h=64,
                                    source="vitpose", commit="x")
    assert json.loads(payload)["frames"][0]["keypoints"]["nose"][2] == pytest.approx(0.5)
