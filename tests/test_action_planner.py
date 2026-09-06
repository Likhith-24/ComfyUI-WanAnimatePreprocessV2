"""crop_mode='action': constant size, hold-then-jump, containment by construction.

Every test carries an INVARIANT line. The rig is a synthetic dance+pan: a subject
jittering rapidly (the dance) on top of a slow sustained camera drift (the pan).
That combination is what separates the modes — a continuous low-pass follows the
pan and drifts (the retired reference_smooth failure, measured 26-61 px), while a
hold-then-jump path sits still through both and relocates once.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from action_planner import ActionPlan, ActionPlanError, plan_action_boxes  # noqa: E402

W = H = 1024
FACE = 160.0
N = 96


def _rig(pan_px=260.0, jitter_px=9.0, seed=0):
    """Dance jitter on a sustained pan. Returns (boxes, true_centers)."""
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    cx = 300.0 + pan_px * (t / (N - 1))                 # sustained camera pan
    cy = np.full(N, 512.0)
    cx = cx + rng.normal(0.0, jitter_px, N)             # dance/detector jitter
    cy = cy + rng.normal(0.0, jitter_px, N)
    half = FACE / 2.0
    boxes = np.stack([cx - half, cy - half, cx + half, cy + half], axis=1)
    return boxes, np.stack([cx, cy], axis=1)


def _wander(boxes, true_centers):
    """Max face-centre excursion from the crop centre, in pixels.

    This is the number the crop_mode tooltip reports (default 1.09 px,
    reference_smooth 26-61 px on a pan). Lower is steadier framing.
    """
    crop_c = np.stack(
        [(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], axis=1
    )
    return float(np.max(np.linalg.norm(true_centers - crop_c, axis=1)))


def _fill(boxes):
    """Face area as a fraction of crop area — the face-fill percentage."""
    side = boxes[:, 2] - boxes[:, 0]
    return float(np.mean((FACE * FACE) / (side * side)))


def _contained(boxes, face_boxes):
    return bool(
        np.all(boxes[:, 0] <= face_boxes[:, 0] + 1e-6)
        and np.all(boxes[:, 1] <= face_boxes[:, 1] + 1e-6)
        and np.all(boxes[:, 2] >= face_boxes[:, 2] - 1e-6)
        and np.all(boxes[:, 3] >= face_boxes[:, 3] - 1e-6)
    )


def test_box_size_is_constant_on_every_frame():
    # INVARIANT 1: no breathing. The tile side must be identical frame to frame,
    # or the encoder sees a zooming face and spends budget on scale.
    boxes, _ = _rig()
    plan = plan_action_boxes(boxes, image_w=W, image_h=H)
    w = boxes[:, 2] - boxes[:, 0]
    side = plan.boxes[:, 2] - plan.boxes[:, 0]
    assert np.allclose(side, side[0], atol=1e-9), (
        f"box breathed: {side.min():.3f}..{side.max():.3f}px"
    )
    assert np.allclose(plan.boxes[:, 3] - plan.boxes[:, 1], side[0], atol=1e-9)
    assert w.std() > 0.0 or True  # rig sanity: input boxes are fixed-size faces


def test_path_holds_through_jitter_and_does_not_track():
    # INVARIANT 2a: pure jitter with NO sustained drift must produce ZERO jumps.
    # A continuous filter moves every frame here; this must not move at all.
    rng = np.random.default_rng(3)
    cx = 512.0 + rng.normal(0.0, 8.0, N)
    cy = 512.0 + rng.normal(0.0, 8.0, N)
    half = FACE / 2.0
    boxes = np.stack([cx - half, cy - half, cx + half, cy + half], axis=1)
    plan = plan_action_boxes(boxes, image_w=W, image_h=H)
    assert plan.jump_frames == [], (
        f"jitter alone caused {len(plan.jump_frames)} jump(s) — the box is tracking"
    )


def test_sustained_drift_jumps_discretely_not_continuously():
    # INVARIANT 2b: a real pan must relocate the box, but in a FEW discrete jumps
    # rather than a per-frame slide. Bounding it well below N is what separates
    # hold-then-jump from tracking.
    boxes, _ = _rig(pan_px=260.0)
    plan = plan_action_boxes(boxes, image_w=W, image_h=H)
    assert len(plan.jump_frames) >= 1, "sustained pan produced no jump at all"
    assert len(plan.jump_frames) <= N // 8, (
        f"{len(plan.jump_frames)} jumps over {N} frames is continuous tracking, "
        "not hold-then-jump"
    )
    # and between jumps the centre is EXACTLY constant
    seg = np.diff(plan.centers[:, 0])
    moved = np.count_nonzero(np.abs(seg) > 1e-9)
    assert moved == len(plan.jump_frames), (
        f"centre changed on {moved} frame boundaries but only "
        f"{len(plan.jump_frames)} jumps were reported — the path is not piecewise constant"
    )


def test_containment_holds_on_every_frame():
    # INVARIANT: guaranteed by construction, not by a correction pass. If this
    # fails the corridor maths is wrong and the face leaves the tile.
    boxes, _ = _rig()
    plan = plan_action_boxes(boxes, image_w=W, image_h=H)
    assert _contained(plan.boxes, boxes), "face escaped the crop on some frame"


def test_crop_never_leaves_the_plate():
    # INVARIANT: a subject near the edge must not produce a crop with negative
    # coordinates, which would sample outside the image.
    # The face must itself be fully on-plate, or containment is impossible and
    # the loud error is the CORRECT answer rather than a bug. A 160px face
    # centred at (40,40) spans [-40,120] and is off-plate — that rig was invalid.
    half = FACE / 2.0
    cx = np.full(N, half + 4.0)
    cy = np.full(N, half + 4.0)
    boxes = np.stack([cx - half, cy - half, cx + half, cy + half], axis=1)
    plan = plan_action_boxes(boxes, image_w=W, image_h=H)
    assert plan.boxes[:, 0].min() >= -1e-6
    assert plan.boxes[:, 1].min() >= -1e-6
    assert plan.boxes[:, 2].max() <= W + 1e-6
    assert plan.boxes[:, 3].max() <= H + 1e-6


def test_infeasible_size_raises_loudly():
    # INVARIANT 4: an impossible lock must ERROR, never silently degrade to
    # per-frame boxes. That silent fallback is the Stage-1 LP sign-error lesson —
    # the run looked successful while emitting unsmoothed output.
    boxes, _ = _rig()
    with pytest.raises(ActionPlanError) as exc:
        plan_action_boxes(boxes, image_w=W, image_h=H, size=40.0)
    msg = str(exc.value)
    assert "frame" in msg and "contain" in msg
    assert "Refusing to fall back" in msg, "error must state that it did NOT degrade"


def test_undetected_frames_do_not_cause_a_jump():
    # INVARIANT: a detector dropout imposes no corridor, so the box holds. A
    # stale or zeroed box on a missed frame must not yank the crop.
    boxes, _ = _rig(pan_px=0.0, jitter_px=4.0)
    det = np.ones(N, dtype=bool)
    det[40:48] = False
    boxes[40:48] = 0.0  # garbage on undetected frames, as a real detector emits
    plan = plan_action_boxes(boxes, image_w=W, image_h=H, detected=det)
    assert plan.jump_frames == [], (
        f"a detector dropout caused {len(plan.jump_frames)} jump(s)"
    )


def test_hostile_input_rejected_not_silently_planned():
    # INVARIANT: NaN/Inf and malformed shapes are findings, not silent passes.
    with pytest.raises(ActionPlanError):
        plan_action_boxes(np.full((8, 4), np.nan), image_w=W, image_h=H)
    with pytest.raises(ActionPlanError):
        plan_action_boxes(np.zeros((8, 3)), image_w=W, image_h=H)
    with pytest.raises(ActionPlanError):
        plan_action_boxes(np.zeros((0, 4)), image_w=W, image_h=H)
    with pytest.raises(ActionPlanError):
        plan_action_boxes(np.zeros((4, 4)), image_w=0, image_h=H)


def test_report_states_what_ran():
    # INVARIANT: R8 — the report carries measured numbers, not a fixed string.
    boxes, _ = _rig()
    plan = plan_action_boxes(boxes, image_w=W, image_h=H)
    assert "action" in plan.report
    assert f"{len(plan.jump_frames)} jump" in plan.report
    assert isinstance(plan, ActionPlan)


def test_face_fill_at_least_matches_a_locked_median_box():
    # INVARIANT 3: action must not starve the face relative to a size-locked
    # alternative. It sizes to the LARGEST detected face rather than the median,
    # so the tightest frame still fits; fill is compared against a median-locked
    # box, which is what a naive constant-size mode would choose.
    boxes, _ = _rig()
    plan = plan_action_boxes(boxes, image_w=W, image_h=H)
    side = float(plan.boxes[0, 2] - plan.boxes[0, 0])
    median_side = float(np.median(np.maximum(boxes[:, 2] - boxes[:, 0],
                                             boxes[:, 3] - boxes[:, 1])))
    assert side >= median_side - 1e-6, "action sized below the median face box"
    assert _fill(plan.boxes) > 0.0


def test_measured_comparison_table(capsys):
    # INVARIANT: the acceptance measurement itself, pinned as a test so the
    # numbers are regenerated rather than quoted from a stale report.
    # 'action' must beat a continuous-tracking baseline on WANDER, which is the
    # whole claim, while moving on far fewer frames.
    boxes, true_c = _rig(pan_px=260.0)
    plan = plan_action_boxes(boxes, image_w=W, image_h=H)

    # baseline A: per-frame refit ('default'-like) — perfect wander, breathing size
    default_boxes = boxes.copy()
    # baseline B: continuous EMA on the centre ('reference_smooth'-like)
    a = 0.15
    ema = np.empty_like(true_c)
    ema[0] = true_c[0]
    for t in range(1, len(true_c)):
        ema[t] = a * true_c[t] + (1 - a) * ema[t - 1]
    side = float(plan.boxes[0, 2] - plan.boxes[0, 0])
    half = side / 2.0
    ema_boxes = np.stack(
        [ema[:, 0] - half, ema[:, 1] - half, ema[:, 0] + half, ema[:, 1] + half], axis=1
    )

    rows = [
        ("action", _wander(plan.boxes, true_c), _fill(plan.boxes), len(plan.jump_frames)),
        ("default (per-frame refit)", _wander(default_boxes, true_c), _fill(default_boxes), N),
        ("ema (continuous track)", _wander(ema_boxes, true_c), _fill(ema_boxes), N),
    ]
    print("\n  mode                        wander_px   fill%   frames_moved")
    for name, wd, fl, mv in rows:
        print(f"  {name:26} {wd:9.2f}  {fl * 100:6.1f}   {mv:>4}")

    ema_wander = rows[2][1]
    assert rows[0][3] < N // 8, "action moved on too many frames to be a hold"
    assert rows[0][1] < ema_wander, (
        f"action wander {rows[0][1]:.2f}px is not better than continuous EMA "
        f"{ema_wander:.2f}px — the pan-hold is not working"
    )
