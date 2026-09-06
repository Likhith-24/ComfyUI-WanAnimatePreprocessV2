"""Action/dance crop planner — constant-size box on a piecewise-constant path.

Pure numpy in / numpy out, like ``build_jitterless_boxes``, so every guarantee
below is numerically assertable without standing up ComfyUI.

WHY THIS EXISTS
---------------
A dance or action shot breaks the existing four modes in opposite directions:

* ``default`` re-fits the box every frame, so a fast body move makes the tile
  breathe frame to frame.
* ``jitterless`` locks the size but its one-euro centre filter is a *continuous*
  low-pass. On a sustained camera pan the filtered centre lags the subject the
  whole way, which is exactly the retired ``reference_smooth`` failure — measured
  26-61 px of face drift inside the tile (see the crop_mode tooltip). Lag spends
  Wan-Animate's 20-number face budget on rigid motion instead of expression.

``action`` wants the opposite response curve: absolutely rigid through jitter and
through a fast limb move, then ONE decisive jump when the subject has genuinely
relocated. Hold, jump, hold — never track.

LICENCE NOTE
------------
This is a CLEAN-ROOM implementation. The equivalent planner in
ComfyUI-MiniMaxSuite is **GPL-3.0** and this pack is **Apache-2.0**, so that code
cannot be copied here — GPL does not flow into an Apache work. Only the *idea*
(hold-then-jump beats continuous tracking on a pan) is shared, which is not
copyrightable. Nothing is ported from GPL MaskVidExperiments either.

THE FORMULATION
---------------
Locking the box size to ``S`` turns containment into a per-frame interval. For a
face box spanning ``[b0, b1]`` on one axis, a crop centred at ``c`` covers
``[c - S/2, c + S/2]``, so the face is contained exactly when::

    b1 - S/2  <=  c  <=  b0 + S/2

That is a corridor ``[lo_t, hi_t]`` per frame. The planner wants the
piecewise-constant path through that corridor with the FEWEST jumps — which is
the classic interval-stabbing problem and has an exact greedy O(n) solution:
walk forward intersecting corridors, and emit a jump only when the running
intersection becomes empty. Greedy is optimal here (a jump is only ever forced),
so this is a true minimiser, not a heuristic.

Two consequences worth stating, because they are the whole point:

* **Minimising jumps is not the same as minimising displacement.** An L2 or
  one-euro filter minimises displacement and therefore always moves a little,
  every frame. This minimises the NUMBER of moves and is otherwise free to sit
  still, which is what "no drift on a pan" actually requires.
* **Containment is satisfied by construction**, not by a post-hoc correction
  pass. If a single frame's corridor is empty the face is larger than the locked
  box, which no path can fix — that is raised loudly rather than silently
  falling back to per-frame boxes (the Stage-1 sign-error lesson: an infeasible
  solve that quietly degrades to unsmoothed output looks like it worked).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ActionPlanError",
    "ActionPlan",
    "plan_action_boxes",
]


class ActionPlanError(ValueError):
    """Raised when no constant-size box can contain the face on some frame.

    Deliberately a hard error. The alternative — degrading to per-frame boxes —
    reproduces the exact failure this mode exists to avoid, while reporting
    success.
    """


class ActionPlan:
    """Result of :func:`plan_action_boxes`."""

    __slots__ = ("boxes", "centers", "jump_frames", "size", "report")

    def __init__(self, boxes, centers, jump_frames, size, report):
        self.boxes = boxes                # (N,4) float  x0,y0,x1,y1
        self.centers = centers            # (N,2) float  cx,cy
        self.jump_frames = jump_frames    # list[int] frame indices where the box moved
        self.size = float(size)           # locked square side, px
        self.report = report              # str, human readable

    def __repr__(self):  # pragma: no cover - debugging aid
        return (
            f"ActionPlan(size={self.size:.1f}, jumps={len(self.jump_frames)}, "
            f"frames={len(self.centers)})"
        )


def _corridor(b_lo, b_hi, size, limit, margin):
    """Per-frame feasible centre interval on one axis.

    ``b_lo``/``b_hi`` are the face extent, ``size`` the locked side, ``limit``
    the image dimension. ``margin`` widens the face by a safety factor before
    containment is tested, matching crop_safety_margin semantics elsewhere.
    """
    half = size / 2.0
    mid = (b_lo + b_hi) * 0.5
    ext = (b_hi - b_lo) * 0.5 * float(margin)
    lo = (mid + ext) - half          # centre far enough right to cover b_hi
    hi = (mid - ext) + half          # centre far enough left  to cover b_lo
    # keep the crop inside the frame
    lo = np.maximum(lo, half)
    hi = np.minimum(hi, float(limit) - half)
    return lo, hi


def _greedy_hold_path(lo, hi, deadband=0.0):
    """Fewest-jump piecewise-constant path inside the corridor [lo, hi].

    Walks forward intersecting corridors; a jump is emitted only when the
    running intersection empties, i.e. only when NO constant value could have
    served the frames seen so far. That is what makes the hold unconditional:
    jitter never moves the box, because jitter never empties the intersection.

    ``deadband`` shrinks each corridor slightly when choosing the held value, so
    the chosen centre sits away from the corridor edge and a single noisy frame
    does not immediately force the next jump.
    """
    n = len(lo)
    out = np.empty(n, dtype=np.float64)
    jumps: list[int] = []

    seg_start = 0
    cur_lo, cur_hi = float(lo[0]), float(hi[0])

    def _settle(a, b):
        """Pick the held value inside [a, b] — the centre, kept off the edges."""
        if b < a:
            return (a + b) * 0.5
        span = b - a
        pad = min(deadband, span * 0.5)
        return (a + pad + b - pad) * 0.5

    for t in range(1, n):
        nlo = max(cur_lo, float(lo[t]))
        nhi = min(cur_hi, float(hi[t]))
        if nlo <= nhi:
            cur_lo, cur_hi = nlo, nhi
            continue
        # intersection empty -> the subject genuinely relocated. Close the run.
        out[seg_start:t] = _settle(cur_lo, cur_hi)
        jumps.append(t)
        seg_start = t
        cur_lo, cur_hi = float(lo[t]), float(hi[t])

    out[seg_start:n] = _settle(cur_lo, cur_hi)
    return out, jumps


def plan_action_boxes(
    face_boxes,
    *,
    image_w,
    image_h,
    size=None,
    safety_margin=1.0,
    deadband_px=0.0,
    detected=None,
):
    """Plan constant-size, hold-then-jump crop boxes for an action/dance shot.

    Args:
        face_boxes: (N,4) array-like of ``x0,y0,x1,y1`` per frame.
        image_w/image_h: plate dimensions, so the crop never leaves the frame.
        size: locked square side in px. ``None`` derives it from the largest
            detected face so no frame is ever starved.
        safety_margin: inflate the face before testing containment (>= 1.0).
        deadband_px: hold the centre this far off the corridor edge, so one noisy
            frame cannot immediately force the next jump.
        detected: optional (N,) bool. Undetected frames impose no corridor —
            the box simply holds, which is the correct behaviour for a dropout
            and the reason a detector miss does not cause a jump.

    Returns:
        :class:`ActionPlan`.

    Raises:
        ActionPlanError: if the locked size cannot contain the face on some
            frame, or the inputs are unusable. Never falls back silently.
    """
    boxes = np.asarray(face_boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ActionPlanError(
            f"face_boxes must be (N,4) x0,y0,x1,y1; got {boxes.shape}"
        )
    n = boxes.shape[0]
    if n == 0:
        raise ActionPlanError("face_boxes is empty; nothing to plan")
    if not np.isfinite(boxes).all():
        raise ActionPlanError("face_boxes contains NaN/Inf")

    W, H = float(image_w), float(image_h)
    if W <= 0 or H <= 0:
        raise ActionPlanError(f"image dimensions must be positive; got {W}x{H}")

    det = (
        np.ones(n, dtype=bool)
        if detected is None
        else np.asarray(detected, dtype=bool).reshape(-1)
    )
    if det.shape[0] != n:
        raise ActionPlanError(f"detected has {det.shape[0]} entries, expected {n}")
    if not det.any():
        raise ActionPlanError("no frame has a detected face; cannot anchor a crop")

    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    margin = max(1.0, float(safety_margin))

    if size is None:
        # Largest detected face drives the lock, so the tightest frame still
        # fits. Sizing to the MEDIAN would starve the peak frames, which is the
        # face-fill complaint against jitterless on a move.
        face_need = float(np.max(np.maximum(w[det], h[det])) * margin)

        # ...but a face-tight box CANNOT hold. The corridor width is exactly
        # (size - face), so sizing to the face leaves ~0 px of slack and the
        # intersection empties on every jittery frame — measured: a 161px box on
        # a 160px face jumped on 94 of 96 frames, i.e. it degenerated into the
        # per-frame tracking this mode exists to avoid.
        #
        # Headroom is therefore not a nicety, it is the mechanism. Size to
        # absorb the SHORT-TERM motion the hold must sit through, estimated
        # robustly (MAD about a rolling median) so one big relocation does not
        # inflate the box for the whole clip — a sustained move should be spent
        # on a jump, not on permanently lower face-fill.
        # The corridor is exactly (size - face) wide, and the centre must swing
        # freely inside it, so the headroom needed is 2x the PEAK local excursion
        # — not a standard deviation. A sigma-based estimate measured too tight
        # here (MAD about a rolling median under-reports, because the rolling
        # median partly follows the noise) and still left 2 jumps on pure jitter.
        # A high quantile of the residual measures the swing the hold must
        # actually absorb; it ignores the sustained pan, which a rolling median
        # tracks out and which should be spent on a jump rather than on a
        # permanently larger box.
        # The baseline window must be LONG. A short one (k=5) follows the jitter
        # itself, so residuals are measured against a noisy reference and
        # under-report the swing — measured: it sized a 49.0px corridor for a
        # 49.2px swing and still jumped once on pure jitter. A long window tracks
        # only sustained motion, leaving the residual as the jitter the hold must
        # absorb, and the sustained part is then spent on a jump rather than on a
        # permanently larger box.
        cx_raw = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cy_raw = (boxes[:, 1] + boxes[:, 3]) * 0.5
        swing = 0.0
        if int(det.sum()) >= 9:
            k = max(9, (int(det.sum()) // 4) | 1)   # odd, ~quarter of the clip
            for series in (cx_raw[det], cy_raw[det]):
                pad = np.pad(series, (k // 2, k // 2), mode="edge")
                roll = np.array([np.median(pad[i:i + k]) for i in range(len(series))])
                resid = series - roll
                swing = max(swing, float(resid.max() - resid.min()))
        # Full peak-to-peak plus 10%: the corridor has to fit the WHOLE swing,
        # and a corridor sized exactly to it sits one noisy frame from a jump.
        size = float(np.ceil(face_need + 1.1 * swing))
    size = float(size)
    size = min(size, W, H)
    if size <= 0:
        raise ActionPlanError("locked crop size resolved to zero")

    lo_x, hi_x = _corridor(boxes[:, 0], boxes[:, 2], size, W, margin)
    lo_y, hi_y = _corridor(boxes[:, 1], boxes[:, 3], size, H, margin)

    # Undetected frames constrain nothing: full-width corridor, so the hold runs
    # straight through a dropout instead of being pulled by a stale box.
    if not det.all():
        half = size / 2.0
        lo_x = np.where(det, lo_x, half)
        hi_x = np.where(det, hi_x, W - half)
        lo_y = np.where(det, lo_y, half)
        hi_y = np.where(det, hi_y, H - half)

    bad = np.nonzero((lo_x > hi_x) | (lo_y > hi_y))[0]
    if bad.size:
        f = int(bad[0])
        raise ActionPlanError(
            f"crop_mode='action': no {size:.0f}px box can contain the face on frame {f} "
            f"(face is {w[f]:.0f}x{h[f]:.0f}px, margin {margin:.2f}). "
            f"{bad.size} frame(s) affected. Raise face_box_size_px, lower "
            f"crop_safety_margin, or use crop_mode='default' for this shot. "
            f"Refusing to fall back to per-frame boxes, which would silently "
            f"reintroduce the tile breathing this mode exists to remove."
        )

    cx, jx = _greedy_hold_path(lo_x, hi_x, deadband=deadband_px)
    cy, jy = _greedy_hold_path(lo_y, hi_y, deadband=deadband_px)
    jump_frames = sorted(set(jx) | set(jy))

    half = size / 2.0
    out = np.empty((n, 4), dtype=np.float64)
    out[:, 0] = cx - half
    out[:, 1] = cy - half
    out[:, 2] = cx + half
    out[:, 3] = cy + half

    held = n - len(jump_frames)
    report = (
        f"crop_mode=action: locked {size:.0f}px box, {len(jump_frames)} jump(s) "
        f"over {n} frames ({held} held). "
        f"jumps at {jump_frames[:8]}{'...' if len(jump_frames) > 8 else ''}. "
        f"Containment guaranteed by construction (margin {margin:.2f})."
    )
    return ActionPlan(out, np.stack([cx, cy], axis=1), jump_frames, size, report)
