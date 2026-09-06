# Copyright 2025 kijai (Jukka Seppänen) — original ComfyUI-WanAnimatePreprocess
#               https://github.com/kijai/ComfyUI-WanAnimatePreprocess
#               Apache License 2.0
#
# Copyright 2025 steven850 — improved pose/face pipeline (CLAHE, temporal
#               smoothing, constant-size face box, blur preprocessing)
#               Contributed in issue #10 of ComfyUI-WanAnimatePreprocess:
#               https://github.com/kijai/ComfyUI-WanAnimatePreprocess/issues/10
#               Apache License 2.0 (contributed to an Apache-2.0 repo)
#
# Copyright 2025-2026 Code2Collapse (https://github.com/Code2Collapse)
#               Additional work: iris/pupil detection (gradient voting, Timm-Barth
#               inspired multi-strategy), MediaPipe FaceMesh integration,
#               protobuf-5.x compatibility fix, V2 extensions and enhancements
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---- Modifications by Code2Collapse (2025-2026) relative to steven850/kijai base ----
# - Added MediaPipe FaceMesh 478-point landmark pipeline with iris/gaze tracking
# - Added protobuf >=5.x compatibility fix for mediapipe <=0.10.x
# - Added gradient-based pupil centre detection (Timm-Barth 2011 inspired)
# - Added multi-strategy iris fallback (contour moments + weighted centroid)
# - Added iris/gaze overlay to debug visualisation
# - Added lip openness ratio output
# - Renamed nodes to V2 namespace; added RETURN_TYPES for iris/gaze/lip outputs

import os
import torch
from tqdm import tqdm
import numpy as np
import folder_paths
import cv2
import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from . import _interrupt_check as _IC
from ._is_changed_util import hash_args_and_kwargs
script_directory = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------
# Optional MediaPipe Face Mesh (graceful fallback)
# ---------------------------------------------------
# MediaPipe provides 478 facial landmarks (468 mesh + 10 iris when
# refine_landmarks=True). It dramatically improves iris/lip tracking
# fidelity compared to the 68-point ViTPose face output and the custom
# OpenCV pupil voter. If `mediapipe` is not installed, the pipeline
# transparently falls back to the legacy ViTPose + `_find_pupil_center`
# code path and keeps working.
try:
    import mediapipe as _mp
    _MP_AVAILABLE = True
except Exception as _mp_err:  # ImportError or runtime DLL issues
    _mp = None
    _MP_AVAILABLE = False
    logging.getLogger(__name__).info(
        "MediaPipe not available, falling back to ViTPose-only face pipeline (%s)",
        _mp_err,
    )

# mediapipe 1.0 REMOVED `mp.solutions.face_mesh` (the whole legacy solutions
# graph), so there is no module-level FaceMesh handle any more and no
# `_get_mp_face_mesh()`. All face-mesh work goes through the FaceLandmarker
# Tasks API in gaze_blendshape.py, which loads `face_landmarker.task` from a
# real ComfyUI models path (see _resolve_model_dir there) rather than bundling
# a graph. The `_mp` import above is kept only as an availability probe.

# ---------------------------------------------------
# MediaPipe -> dlib 68 landmark mapping
# ---------------------------------------------------
# Wan 2.x face conditioning consumes the standard 68-point dlib layout
# (slotted into face_kps[1:69]; face_kps[0] is the body-anchored face
# centre coming from ViTPose). We slice the 478 MediaPipe FaceMesh
# vertices to reconstruct that exact ordering, so existing limbSeq /
# `draw_aapose_by_meta_new` visualisation and the Wan pose encoder keep
# working without modification.
#
# Layout (68 = 17+5+5+4+5+6+6+12+8):
#   0-16  jawline (right ear -> chin -> left ear)
#   17-21 right eyebrow
#   22-26 left eyebrow
#   27-30 nose bridge (top -> tip)
#   31-35 nose bottom (right nostril -> tip -> left nostril)
#   36-41 right eye  (outer, upper-outer, upper-inner, inner, lower-inner, lower-outer)
#   42-47 left eye
#   48-59 outer lip (12 pts, clockwise from right corner)
#   60-67 inner lip (8 pts, clockwise from right corner)
MP_TO_DLIB68 = [
    # Jaw 0-16
    127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
    377, 400, 378, 379, 365,
    # Right eyebrow 17-21
    70, 63, 105, 66, 107,
    # Left eyebrow 22-26
    336, 296, 334, 293, 300,
    # Nose bridge 27-30
    168, 6, 197, 195,
    # Nose bottom 31-35
    115, 220, 4, 440, 344,
    # Right eye 36-41
    33, 160, 158, 133, 153, 144,
    # Left eye 42-47
    362, 385, 387, 263, 373, 380,
    # Outer lip 48-59
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    # Inner lip 60-67
    78, 81, 13, 311, 308, 402, 14, 178,
]
assert len(MP_TO_DLIB68) == 68, "MP -> dlib mapping must define exactly 68 indices"

# Iris landmarks (only present when refine_landmarks=True).
MP_RIGHT_IRIS_CENTER = 468
MP_LEFT_IRIS_CENTER = 473
MP_RIGHT_IRIS_RING = [469, 470, 471, 472]
MP_LEFT_IRIS_RING = [474, 475, 476, 477]

# Eye corner landmarks (same MediaPipe model as the iris — use these as
# the gaze reference frame instead of dlib eye-contour centroid. Reason:
# the dlib contour averages upper+lower eyelid points whose vertical
# spread is asymmetric (lower lid extends further than upper), biasing
# the centroid DOWN by ~2 px relative to the iris and producing a
# spurious "iris-is-up" signal even when the subject looks straight
# ahead. Eye-corner midpoint has no eyelid component, so vertical
# offset = true vertical gaze and horizontal offset = true horizontal
# gaze. Subject's right eye (viewer-left): outer=33 inner=133. Subject's
# left eye (viewer-right): inner=362 outer=263.
MP_RIGHT_EYE_OUTER = 33
MP_RIGHT_EYE_INNER = 133
MP_LEFT_EYE_INNER  = 362
MP_LEFT_EYE_OUTER  = 263

# Inner-lip indices used for "openness" (mouth aspect ratio).
# Vertical opening: top-inner (13) <-> bottom-inner (14).
# Horizontal width: right inner corner (78) <-> left inner corner (308).
MP_INNER_LIP_TOP = 13
MP_INNER_LIP_BOTTOM = 14
MP_INNER_LIP_RIGHT = 78
MP_INNER_LIP_LEFT = 308


def mediapipe_to_dlib_68(mp_landmarks_xy):
    """Slice the 478-point MediaPipe array down to the 68-point dlib layout.

    Args:
        mp_landmarks_xy: (478, 2) ndarray of (x, y) coordinates in any
            consistent space (normalised or pixel).

    Returns:
        (68, 2) ndarray in the same coordinate space, ordered exactly as
        dlib's 68-point shape predictor expects.
    """
    return mp_landmarks_xy[MP_TO_DLIB68].copy()


# FaceMesh/FaceLandmarker need the face at a decent pixel size — the attention
# (iris) submesh loses the eye region below roughly ~200px faces. On full/half-
# body shots (the NORMAL Wan Animate input) the padded face crop is often only
# 60–120px → "no face found" / garbage iris = the user's "eye detection not at
# all working". Upscaling the CROP is free w.r.t. coordinate mapping because
# MediaPipe returns NORMALISED [0,1] coords (scale-invariant) that we multiply
# by the ORIGINAL crop_size_wh. Verified live: a 288px face crop failed raw,
# detected perfectly at 3× with iris rings exactly on the irises.
_MP_MIN_CROP_SIDE = 384


def _upscale_face_crop_if_small(face_crop_rgb_uint8):
    """Return the crop upscaled so its long side is >= _MP_MIN_CROP_SIDE."""
    try:
        h, w = face_crop_rgb_uint8.shape[:2]
        side = max(h, w)
        if side >= _MP_MIN_CROP_SIDE or side < 8:
            return face_crop_rgb_uint8
        s = _MP_MIN_CROP_SIDE / float(side)
        return cv2.resize(face_crop_rgb_uint8, (max(8, int(round(w * s))), max(8, int(round(h * s)))),
                          interpolation=cv2.INTER_CUBIC)
    except Exception:
        return face_crop_rgb_uint8


def _run_mediapipe_on_face_crop(face_crop_rgb_uint8, crop_origin_xy, crop_size_wh,
                                  full_w, full_h):
    """Face-mesh landmarks for a single face crop.

    MIGRATED (2026-07-24) from the legacy ``mp.solutions.face_mesh`` graph to
    the FaceLandmarker Tasks API. mediapipe 1.0 REMOVED ``solutions.face_mesh``
    outright — it is not a protobuf-version problem and there is no legacy
    graph left to fall back to — so this is now a thin delegation to
    :func:`_run_face_landmarker_on_face_crop` rather than a second,
    independent implementation.

    Kept as a named function (not deleted) because several call sites and the
    au_amplify path reference it, and its documented return shape is a
    published-enough internal contract to be worth preserving verbatim.

    Verified equivalent before the swap, on a real face, same image, same
    build: both paths return 478 landmarks with the iris at indices 468-477
    in the SAME normalised-[0,1]-relative-to-crop convention; per-landmark
    agreement was mean 1.61px / max 6.48px over the full mesh and mean 1.10px
    over the iris ring specifically. The coordinate mapping below therefore
    did not change at all — it lives in the delegate, byte-identical.

    Args:
        face_crop_rgb_uint8: (h, w, 3) uint8 RGB face crop.
        crop_origin_xy:      (x1, y1) origin of the crop in the full image.
        crop_size_wh:        (cw, ch) pixel size of the crop.
        full_w, full_h:      full image pixel dimensions.

    Returns:
        dict with:
            - 'kps68_norm'  (68, 3) [x/W, y/H, conf=1.0] in *full image* normalised space
            - 'right_iris_px', 'left_iris_px': (x, y) in full image pixel space
            - 'right_iris_radius_px', 'left_iris_radius_px': float
            - 'lip_openness_ratio': float (vertical inner-lip / inner-lip width)
            - 'landmarks_px_full': (478, 2) full mesh in full-frame pixels
        Or None if no face was found / the Tasks model is unavailable.
        (The delegate additionally returns blend-shape gaze + R_head; callers
        that want those already call it directly.)
    """
    return _run_face_landmarker_on_face_crop(
        face_crop_rgb_uint8, crop_origin_xy, crop_size_wh, full_w, full_h,
    )


# ---------------------------------------------------
# Production gaze via FaceLandmarker Tasks API + blend shapes
# ---------------------------------------------------
# The Tasks API replaces the legacy `mp.solutions.face_mesh` glue and
# additionally returns 52 ARKit-compatible blend shapes per face. We use
# the eight `eyeLookIn/Out/Up/Down{Left,Right}` shapes to derive
# head-pose-corrected per-eye yaw/pitch in radians — i.e. real gaze
# angles, not 2D iris offsets. See `gaze_blendshape.py` for the math.
try:
    from . import gaze_blendshape as _gaze_bs  # type: ignore
    _GAZE_BS_IMPORTED = True
except Exception as _exc:  # noqa: BLE001
    _gaze_bs = None
    _GAZE_BS_IMPORTED = False
    logging.getLogger(__name__).info(
        "gaze_blendshape module unavailable (%s); blend-shape gaze disabled.",
        _exc,
    )

# Stage-1 gaze upgrade: solvePnP head pose + Kalman temporal smoother.
# Pure numpy + cv2, no extra downloads. Powers the
# `gaze_engine='blendshape_head_corrected'` widget option.
try:
    from . import gaze_3d as _gaze_3d  # type: ignore
    _GAZE_3D_IMPORTED = True
except Exception as _exc:  # noqa: BLE001
    _gaze_3d = None
    _GAZE_3D_IMPORTED = False
    logging.getLogger(__name__).info(
        "gaze_3d module unavailable (%s); head-pose gaze correction disabled.",
        _exc,
    )

# W7-G1 gaze upgrade: geometric (measured-iris) eye-in-head engine.
# MEASURES the iris position inside the eye aperture instead of estimating
# gaze with a NN; composes with the same solvePnP head pose + Kalman as
# blendshape_head_corrected. Pure math, no downloads.
try:
    from . import gaze_iris_geometric as _gaze_iris  # type: ignore
    _GAZE_IRIS_IMPORTED = True
except Exception as _exc:  # noqa: BLE001
    _gaze_iris = None
    _GAZE_IRIS_IMPORTED = False
    logging.getLogger(__name__).info(
        "gaze_iris_geometric module unavailable (%s); iris_geometric engine disabled.",
        _exc,
    )

# Stage-2 gaze upgrade: L2CS-Net (MIT license, ~3.9 deg MPIIGaze MAE,
# ~10.4 deg Gaze360 MAE — robust to extreme head poses). Optional; only
# imported when the user explicitly selects an `l2cs_*` engine.
try:
    from . import gaze_l2cs as _gaze_l2cs  # type: ignore
    _GAZE_L2CS_IMPORTED = True
except Exception as _exc:  # noqa: BLE001
    _gaze_l2cs = None
    _GAZE_L2CS_IMPORTED = False
    logging.getLogger(__name__).debug(
        "gaze_l2cs module unavailable (%s); L2CS-Net engine disabled.",
        _exc,
    )

# Stage-3 gaze upgrade: pose-normalized data preprocessing (clean-room
# port of the 2018 ETRA paper, Apache-2.0). Pure cv2+numpy; head-pose-
# invariant input warp.
try:
    from . import gaze_pose_norm as _gaze_pose_norm  # type: ignore
    _GAZE_POSE_NORM_IMPORTED = True
except Exception as _exc:  # noqa: BLE001
    _gaze_pose_norm = None
    _GAZE_POSE_NORM_IMPORTED = False
    logging.getLogger(__name__).debug(
        "gaze_pose_norm module unavailable (%s); pose normalization disabled.",
        _exc,
    )

# Pose-normalized ResNet50 gaze estimator (operates on the warped
# canonical face crop from gaze_pose_norm). Wraps a community-released
# checkpoint that the USER places at ComfyUI/models/gaze/ — see the
# license note in gaze_normalized_estimator.py for terms of use.
try:
    from . import gaze_normalized_estimator as _gaze_normalized_estimator  # type: ignore
    _GAZE_NORM_EST_IMPORTED = True
except Exception as _exc:  # noqa: BLE001
    _gaze_normalized_estimator = None
    _GAZE_NORM_EST_IMPORTED = False
    logging.getLogger(__name__).debug(
        "gaze_normalized_estimator module unavailable (%s); "
        "pose_normalized_resnet50 engine disabled.",
        _exc,
    )


def _run_face_landmarker_on_face_crop(
    face_crop_rgb_uint8, crop_origin_xy, crop_size_wh, full_w, full_h,
):
    """Run MediaPipe FaceLandmarker on a face crop and pack into the
    same dict shape as :func:`_run_mediapipe_on_face_crop`, plus an
    extra ``gaze`` entry derived from the eye-look blend shapes.

    Returns ``None`` when the Tasks API is not available or no face was
    found in the crop. Caller may fall back to FaceMesh.
    """
    if not _GAZE_BS_IMPORTED or _gaze_bs is None:
        return None
    if face_crop_rgb_uint8 is None or face_crop_rgb_uint8.size == 0:
        return None
    # Same small-face upscale as the FaceMesh path — normalised coords make it
    # transparent to the crop_size_wh mapping below.
    res = _gaze_bs.run_face_landmarker(_upscale_face_crop_if_small(face_crop_rgb_uint8))
    if res is None:
        return None

    landmarks = res["landmarks_norm"]   # (478, 3) in [0,1] crop-space
    if landmarks.shape[0] < 478:
        return None

    cx0, cy0 = crop_origin_xy
    cw, ch = crop_size_wh

    pts_px = np.empty((landmarks.shape[0], 2), dtype=np.float32)
    pts_px[:, 0] = landmarks[:, 0] * cw + cx0
    pts_px[:, 1] = landmarks[:, 1] * ch + cy0

    kps68_px = pts_px[MP_TO_DLIB68]
    kps68_norm = np.zeros((68, 3), dtype=np.float32)
    kps68_norm[:, 0] = kps68_px[:, 0] / max(full_w, 1)
    kps68_norm[:, 1] = kps68_px[:, 1] / max(full_h, 1)
    kps68_norm[:, 2] = 1.0

    r_iris = pts_px[MP_RIGHT_IRIS_CENTER]
    l_iris = pts_px[MP_LEFT_IRIS_CENTER]
    r_ring = pts_px[MP_RIGHT_IRIS_RING]
    l_ring = pts_px[MP_LEFT_IRIS_RING]
    r_radius = float(np.mean(np.linalg.norm(r_ring - r_iris[None, :], axis=1)))
    l_radius = float(np.mean(np.linalg.norm(l_ring - l_iris[None, :], axis=1)))

    r_outer = pts_px[MP_RIGHT_EYE_OUTER]
    r_inner = pts_px[MP_RIGHT_EYE_INNER]
    l_inner = pts_px[MP_LEFT_EYE_INNER]
    l_outer = pts_px[MP_LEFT_EYE_OUTER]

    top = pts_px[MP_INNER_LIP_TOP]
    bot = pts_px[MP_INNER_LIP_BOTTOM]
    rgt = pts_px[MP_INNER_LIP_RIGHT]
    lft = pts_px[MP_INNER_LIP_LEFT]
    v = float(np.linalg.norm(top - bot))
    h = float(np.linalg.norm(rgt - lft))
    lip_ratio = float(v / h) if h > 1e-6 else 0.0

    gaze = _gaze_bs.blendshapes_to_gaze(res.get("blendshapes") or {})

    # Stage-1: solvePnP head rotation from full-frame MediaPipe pixel
    # landmarks. Used downstream to compose eye-in-head gaze with the
    # head pose so the rendered arrow tracks rotated heads correctly.
    R_head = None
    if _GAZE_3D_IMPORTED and _gaze_3d is not None:
        try:
            hp = _gaze_3d.estimate_head_pose_from_pixels(
                pts_px, (int(full_w), int(full_h)),
            )
            if hp is not None:
                R_head = hp[0]
        except Exception:  # noqa: BLE001
            R_head = None

    return {
        'kps68_norm': kps68_norm,
        'right_iris_px': (float(r_iris[0]), float(r_iris[1])),
        'left_iris_px': (float(l_iris[0]), float(l_iris[1])),
        'right_iris_radius_px': r_radius,
        'left_iris_radius_px': l_radius,
        'right_eye_outer_px': (float(r_outer[0]), float(r_outer[1])),
        'right_eye_inner_px': (float(r_inner[0]), float(r_inner[1])),
        'left_eye_inner_px':  (float(l_inner[0]), float(l_inner[1])),
        'left_eye_outer_px':  (float(l_outer[0]), float(l_outer[1])),
        'lip_openness_ratio': lip_ratio,
        # NEW: production gaze from blend shapes — head-pose corrected,
        # in radians per eye, plus a 2D dx/dy for legacy debug overlay.
        'gaze_blendshape': gaze,
        'blendshapes': res.get("blendshapes") or {},
        'face_transform': res.get("transform"),
        'R_head': R_head,
        'source': 'face_landmarker',
        # Full 478-point MediaPipe mesh in FULL-FRAME pixel coords.
        # Consumed by gaze_pose_norm for solvePnP head-pose estimation.
        'landmarks_px_full': pts_px,
    }


from comfy import model_management as mm
from comfy.utils import ProgressBar
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))


def _ensure_onnx_detection_support():
    """Make our nodes detect `.onnx` even when ComfyUI's folder extension
    filter omits it. ComfyUI's `get_filename_list` only returns files whose
    extension is registered for the folder; `.onnx` is not in the default set,
    so detection models silently fail to list. We force-register `.onnx`
    (plus the usual weight extensions) for the `detection` folder, creating the
    entry and the directory if needed, and bust the filename cache.
    """
    exts = {".onnx", ".pt", ".pth", ".safetensors", ".bin"}
    extra_dirs = []
    try:
        det = os.path.join(folder_paths.models_dir, "detection")
        os.makedirs(det, exist_ok=True)
        extra_dirs.append(det)
        # also surface a couple of common alt locations users drop ONNX into
        for alt in ("onnx", "ultralytics", "vitpose", "yolo"):
            p = os.path.join(folder_paths.models_dir, alt)
            if os.path.isdir(p):
                extra_dirs.append(p)
    except Exception:
        pass
    try:
        entry = folder_paths.folder_names_and_paths.get("detection")
        if entry is None:
            folder_paths.folder_names_and_paths["detection"] = (list(extra_dirs), set(exts))
        else:
            paths, e = entry[0], entry[1]
            for d in extra_dirs:
                if d not in paths:
                    paths.append(d)
            try:
                e.update(exts)                      # mutate existing set
            except (AttributeError, TypeError):
                folder_paths.folder_names_and_paths["detection"] = (paths, set(e) | exts)
    except Exception as _e:
        print(f"[WanAnimateV2] .onnx detection registration warning: {_e}")
    # bust caches so the new extension takes effect immediately
    for attr in ("filename_list_cache", "cache_helper"):
        try:
            c = getattr(folder_paths, attr, None)
            if isinstance(c, dict):
                c.pop("detection", None)
        except Exception:
            pass


_ensure_onnx_detection_support()


def list_onnx_detection_models():
    """Robust model list for the detection dropdowns: prefer ComfyUI's
    `get_filename_list("detection")`, but ALSO scan the detection folder(s) on
    disk for `*.onnx` directly — so files show up even if folder_paths refuses
    to (the user's explicit requirement: ours must detect `.onnx` regardless).
    """
    names = []
    seen = set()

    def _add(n):
        if not n:
            return
        n = n.replace("\\", "/")          # normalise so disk-scan + folder_paths dedupe
        if n not in seen:
            seen.add(n)
            names.append(n)

    try:
        for n in folder_paths.get_filename_list("detection"):
            _add(n)
    except Exception:
        pass
    try:
        for base in folder_paths.get_folder_paths("detection"):
            if not base or not os.path.isdir(base):
                continue
            for root, _dirs, files in os.walk(base):
                for f in files:
                    if f.lower().endswith((".onnx", ".pt", ".pth", ".safetensors")):
                        rel = os.path.relpath(os.path.join(root, f), base)
                        _add(rel.replace("\\", "/"))
    except Exception:
        pass
    return names if names else ["(place .onnx models in ComfyUI/models/detection)"]


def _resolve_detection_path(name):
    """Path resolver that tolerates names discovered by the disk scan even if
    folder_paths can't map them (mirrors list_onnx_detection_models)."""
    try:
        p = folder_paths.get_full_path("detection", name)
        if p and os.path.exists(p):
            return p
    except Exception:
        pass
    try:
        for base in folder_paths.get_folder_paths("detection"):
            cand = os.path.join(base, name.replace("/", os.sep))
            if os.path.exists(cand):
                return cand
    except Exception:
        pass
    raise FileNotFoundError(
        f"Detection model {name!r} not found under ComfyUI/models/detection/. "
        f"Place the .onnx file there (or check the name)."
    )


from .models.onnx_models import ViTPose, Yolo
from .pose_utils.pose2d_utils import load_pose_metas_from_kp2ds_seq, crop, bbox_from_detector
from .utils import (
    get_face_bboxes,
    _SOURCE_FACE_MIN_PX,
    padding_resize,
    adjust_bbox_eye_upper_third,
    apply_eye_offset_to_center,
    compute_eye_midpoint_from_face_kps,
    compute_frame_blur_score,
    compute_eye_region_brightness,
    resize_face_crop,
    amplify_landmarks_from_neutral,
)
from .pose_utils.human_visualization import AAPoseMeta, draw_aapose_by_meta_new
from .retarget_pose import get_retarget_pose
from .flux_retarget import (
    edit_with_flux, free_flux_cache, get_editing_prompts, load_flux_kontext,
)


# ---------------------------------------------------
# Image enhancement utilities
# ---------------------------------------------------
def preprocess_for_pose(img, use_clahe=True, clahe_clip=2.0, clahe_grid=8,
                        gamma=1.0, white_balance=False, denoise=0.0,
                        sharpen=0.0, saturation=1.0):
    """Detection-side image conditioning for ViTPose / YOLO / MediaPipe.

    Applied ONLY to the detector's input — never to the pixels that ship as
    face_images or pose_data. Detection quality is what limits landmark
    accuracy on bad footage, so cleaning the detector's view is free accuracy;
    doing it to the OUTPUT would alter what the Wan-Animate face encoder sees
    and is a different (and much riskier) decision.

    Order matters and is deliberate:
      white balance -> gamma -> CLAHE -> denoise -> sharpen -> saturation
    Grey-world balance first (so CLAHE isn't amplifying a colour cast),
    gamma before CLAHE (lift shadows into the range CLAHE can equalise),
    denoise before sharpen (never sharpen noise), saturation last (it only
    affects the chroma planes the luma work above ignored).

    All steps are no-ops at their defaults, so the default path is
    byte-identical to the original CLAHE-only behaviour.
    """
    if img is None:
        return img

    # NO AUTO COLOUR (reverted 2026-07-31). A previous version derived
    # white-balance and gamma from the frame automatically. That is wrong and
    # it broke real footage: grey-world balance measures the spread of the
    # channel means and calls anything lopsided a "colour cast" to neutralise
    # — but it cannot tell a cast from a subject that is GENUINELY strongly
    # coloured. On a gold-painted character it neutralised the actual paint,
    # handing the detector an image nothing like the shot and moving every
    # landmark. Exposure auto-gamma had the same problem on deliberately dark
    # or bright grades.
    #
    # These are DETECTOR-side controls and they stay MANUAL, at no-op
    # defaults. The operator can see the footage; this function cannot.
    do_wb = bool(white_balance)
    do_gamma = abs(float(gamma) - 1.0) > 1e-3
    do_clahe = bool(use_clahe)
    do_dn = float(denoise) > 1e-3
    do_sh = float(sharpen) > 1e-3
    do_sat = abs(float(saturation) - 1.0) > 1e-3
    if not (do_wb or do_gamma or do_clahe or do_dn or do_sh or do_sat):
        return img

    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    if do_wb:
        # Grey-world: equalise per-channel means. Cheap, no model, and it
        # rescues tungsten/underwater casts that otherwise skew skin tone and
        # cost the face detector confidence.
        f = u8.astype(np.float32)
        means = f.reshape(-1, f.shape[-1]).mean(axis=0)
        grey = float(means.mean())
        with np.errstate(divide="ignore", invalid="ignore"):
            gains = np.where(means > 1e-3, grey / means, 1.0)
        gains = np.clip(gains, 0.5, 2.0)
        u8 = np.clip(f * gains, 0, 255).astype(np.uint8)

    if do_gamma:
        g = float(np.clip(gamma, 0.1, 5.0))
        lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / g)) * 255.0, 0, 255).astype(np.uint8)
        u8 = cv2.LUT(u8, lut)

    if do_clahe or do_sat:
        lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        if do_clahe:
            grid = max(1, int(clahe_grid))
            clahe = cv2.createCLAHE(clipLimit=max(0.1, float(clahe_clip)),
                                    tileGridSize=(grid, grid))
            l = clahe.apply(l)
        if do_sat:
            s = float(np.clip(saturation, 0.0, 3.0))
            a = np.clip((a.astype(np.float32) - 128.0) * s + 128.0, 0, 255).astype(np.uint8)
            b = np.clip((b.astype(np.float32) - 128.0) * s + 128.0, 0, 255).astype(np.uint8)
        u8 = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    if do_dn:
        # Edge-preserving: a plain blur would cost ViTPose the very edges it
        # localises keypoints from.
        strength = float(np.clip(denoise, 0.0, 1.0))
        u8 = cv2.bilateralFilter(u8, d=5, sigmaColor=25 + 75 * strength,
                                 sigmaSpace=25 + 75 * strength)

    if do_sh:
        amt = float(np.clip(sharpen, 0.0, 2.0))
        blur = cv2.GaussianBlur(u8, (0, 0), 1.2)
        u8 = np.clip(u8.astype(np.float32) * (1.0 + amt)
                     - blur.astype(np.float32) * amt, 0, 255).astype(np.uint8)

    return u8.astype(np.float32) / 255.0


# ---------------------------------------------------
# Iris / pupil estimation (image-based)
# ---------------------------------------------------
# Eye contour landmark indices within the 69-point face array.
# face array = kp2ds[22:91]; index 0 is body, indices 1-68 are
# standard 68-face landmarks (standard N -> face[N+1]).
#
# Dlib-68 eye contour indices for the EXISTING ``face_kps`` array layout.
#
# CRITICAL INDEXING NOTE (verified May 2026 with live _gaze_debug.log):
# face_kps[0] is the body-anchored face anchor; face_kps[1:69] holds the
# 68 dlib landmarks (assigned via `face_kps[1:69, :] = mp_result['kps68_norm']`
# where kps68_norm is 0-indexed dlib order). So to access dlib's right-eye
# contour (dlib indices 36-41) we read array slots 37-42; for the left eye
# (dlib 42-47) we read array slots 43-48.
#
# History: A prior attempted fix changed these to [36..41]/[42..47]
# (commit cc608dd) on the assumption the array was 0-indexed dlib. Live
# diagnostics on a real frame proved otherwise: the LEFT-eye centroid was
# being contaminated by the lower-outer RIGHT-eye corner, dragging the
# centroid leftward and flipping (iris - centroid) to positive when the
# subject was actually looking left — i.e. arrow pointed RIGHT. Reverted.
_RIGHT_EYE_IDX = [37, 38, 39, 40, 41, 42]
_LEFT_EYE_IDX  = [43, 44, 45, 46, 47, 48]
_EYE_CONTOUR_INDICES = [_RIGHT_EYE_IDX, _LEFT_EYE_IDX]

# Eye-Aspect-Ratio of a comfortably open eye. Used as the target when forcing
# eyes open; a natural open eye sits around 0.28-0.35, a blink near 0.05-0.15.
_EYE_OPEN_TARGET_EAR = 0.30
# Never scale a lid open by more than this — past it the eyelid geometry stops
# being a plausible deformation of the source face and the downstream warp
# starts smearing rather than opening.
_EYE_OPEN_MAX_SCALE = 3.0


def force_eye_open_landmarks(kps_norm, W, H, amount,
                             target_ear=_EYE_OPEN_TARGET_EAR,
                             max_scale=_EYE_OPEN_MAX_SCALE):
    """Open the eyelids of one frame's 69-row face-landmark array in place-safe
    fashion (returns a NEW array), and report what it did.

    Why this lives at the LANDMARK level even though Wan-Animate is 100%
    pixel-driven: DrawViTPoseV2.apply_pose_edits_to_face already turns a
    landmark delta (pose_metas vs pose_metas_original) into a real Delaunay
    piecewise-affine warp of the face crop. So writing opened-eye landmarks
    here IS the pixel edit — the existing warp executes it, using the frame's
    own crop as the source, which is exactly the "preserve identity / pose /
    mouth / lighting, move only the eye region" property that matters.

    Geometry: the two eye CORNERS (p1, p4) are held fixed and define the eye's
    axis; the four lid points are scaled along the perpendicular to that axis
    about the corner midline. That opens the aperture without translating,
    rotating or resizing the eye, so the warp stays local.

    ``kps_norm`` is (>=69, 2+) normalised to the full frame. Only rows in
    _RIGHT_EYE_IDX / _LEFT_EYE_IDX are touched. Scaling is clamped to
    [1.0, max_scale] — this only ever OPENS, never closes.

    Returns ``(new_kps, info)`` where info is
    ``{"right_ear": float, "left_ear": float, "right_scale": float,
       "left_scale": float, "changed": bool}``. NaN EARs mean the eye was
    unusable (missing/zeroed landmarks) and it was left untouched.
    """
    out = np.array(kps_norm, dtype=np.float32, copy=True)
    info = {"right_ear": float("nan"), "left_ear": float("nan"),
            "right_scale": 1.0, "left_scale": 1.0, "changed": False}
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0 or out.shape[0] <= max(_LEFT_EYE_IDX):
        return out, info

    for side, idx in (("right", _RIGHT_EYE_IDX), ("left", _LEFT_EYE_IDX)):
        ear = _eye_aspect_ratio(out, idx, W, H)
        info[f"{side}_ear"] = float(ear)
        if not np.isfinite(ear) or ear <= 1e-6:
            continue                      # unusable eye — leave it alone
        if ear >= target_ear:
            continue                      # already open enough

        # Work in pixels so the aspect ratio of the frame can't skew the
        # perpendicular direction.
        px = out[idx, :2].astype(np.float64) * np.array([W, H], dtype=np.float64)
        p1, p2, p3, p4, p5, p6 = px       # p1/p4 = outer/inner corner
        axis = p4 - p1
        axis_len = float(np.hypot(axis[0], axis[1]))
        if axis_len < 1e-6:
            continue
        u = axis / axis_len
        n = np.array([-u[1], u[0]], dtype=np.float64)   # perpendicular
        centre = 0.5 * (p1 + p4)

        k_full = float(np.clip(target_ear / ear, 1.0, max_scale))
        k = 1.0 + (k_full - 1.0) * amount               # blend by `amount`
        info[f"{side}_scale"] = k
        if k <= 1.0 + 1e-6:
            continue

        for slot in (1, 2, 4, 5):                       # p2, p3, p5, p6
            d = px[slot] - centre
            along = float(np.dot(d, u))
            perp = float(np.dot(d, n))
            px[slot] = centre + along * u + (perp * k) * n

        out[idx, 0] = (px[:, 0] / max(W, 1)).astype(np.float32)
        out[idx, 1] = (px[:, 1] / max(H, 1)).astype(np.float32)
        info["changed"] = True

    return out, info

# Gaze-arrow rendering tunables.
# DEAD_ZONE_PX: iris-centroid offsets smaller than this are treated as
# noise (MediaPipe landmark precision is roughly 1px on cropped faces).
# Below the dead-zone we suppress the arrow entirely.
_GAZE_DEAD_ZONE_PX = 1.2
# Arrow length is scaled by gaze magnitude (offset / iris_radius) so
# subtle gazes show subtle arrows instead of always-35-pixel arrows. Hard
# floor + ceiling keep arrows in [0, MAX_ARROW_LEN_PX].
_GAZE_MAX_ARROW_LEN_PX = 35
_GAZE_MIN_ARROW_LEN_PX = 6


def _gradient_vote_pupil(roi_gray, mask, gc_local, eye_w, eye_h):
    """Gradient-based pupil centre detection (Timm-Barth 2011 inspired).

    Edge gradients around the circular pupil boundary point radially outward.
    By casting rays in the *negative* gradient direction from every strong-edge
    pixel we accumulate votes at the true centre.

    Returns (local_cx, local_cy, score) or None.
    """
    h, w = roi_gray.shape
    gx = cv2.Sobel(roi_gray.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_gray.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    mag_masked = mag.copy()
    mag_masked[mask == 0] = 0
    vals = mag_masked[mask > 0]
    if len(vals) < 8 or vals.max() < 1:
        return None

    thresh = float(np.percentile(vals, 70))  # top 30 % of gradients
    strong = (mag > thresh) & (mask > 0)
    if np.count_nonzero(strong) < 8:
        return None

    ys, xs = np.where(strong)
    gx_s = gx[ys, xs]
    gy_s = gy[ys, xs]
    mag_s = mag[ys, xs]

    # Normalise & negate  (point *toward* centre)
    gx_n = -gx_s / (mag_s + 1e-10)
    gy_n = -gy_s / (mag_s + 1e-10)

    accumulator = np.zeros((h, w), np.float64)
    max_t = max(int(max(eye_w, eye_h) * 0.5), 5)

    for t in range(1, max_t + 1):
        px = (xs + gx_n * t + 0.5).astype(np.int32)
        py = (ys + gy_n * t + 0.5).astype(np.int32)
        valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        pxv, pyv, magv = px[valid], py[valid], mag_s[valid]
        in_mask = mask[pyv, pxv] > 0
        np.add.at(accumulator, (pyv[in_mask], pxv[in_mask]), magv[in_mask])

    # Weight by darkness (pupil region is darker than sclera)
    dark_w = (255.0 - roi_gray.astype(np.float64)) / 255.0
    accumulator *= (0.5 + 0.5 * dark_w)
    accumulator[mask == 0] = 0

    if accumulator.max() < 1:
        return None

    acc_smooth = cv2.GaussianBlur(accumulator, (5, 5), 1.0)
    _, max_val, _, max_loc = cv2.minMaxLoc(acc_smooth, mask=mask)
    cx, cy = float(max_loc[0]), float(max_loc[1])

    mean_acc = float(np.mean(accumulator[mask > 0]) + 1e-6)
    score = min(1.0, max_val / (mean_acc * 5))
    return cx, cy, score


def _find_pupil_center(eye_pts_px, img_gray, W, H):
    """Locate the pupil/iris centre inside one eye.

    Pipeline
    --------
    1. Build a tight eye-region mask from 6 contour landmarks.
    2. Restrict search to the **upper 65 %** of the lid opening to avoid
       eyelid / eyelash shadow contamination.
    3. Apply CLAHE for better pupil-iris-sclera contrast.
    4. **Primary** – gradient-based centre voting (Timm-Barth inspired):
       robust to lighting, threshold-free.
    5. **Secondary** – multi-threshold contour moments with asymmetric
       vertical scoring.
    6. **Tertiary** – weighted dark-pixel centroid with upper-region bias.
    7. Fallback – geometric centre of the eye contour.
    """
    geo_center = np.mean(eye_pts_px, axis=0)

    # --- Eye Aspect Ratio (EAR) – skip closed eyes ---
    v1 = np.linalg.norm(eye_pts_px[1] - eye_pts_px[5])
    v2 = np.linalg.norm(eye_pts_px[2] - eye_pts_px[4])
    horiz = np.linalg.norm(eye_pts_px[0] - eye_pts_px[3])
    if horiz < 3:
        return float(geo_center[0]), float(geo_center[1]), 0.0
    ear = (v1 + v2) / (2.0 * horiz)
    if ear < 0.12:
        return float(geo_center[0]), float(geo_center[1]), 0.05

    # --- Tight padded ROI ---
    min_xy = np.min(eye_pts_px, axis=0)
    max_xy = np.max(eye_pts_px, axis=0)
    eye_w = max_xy[0] - min_xy[0]
    eye_h = max_xy[1] - min_xy[1]
    pad = max(int(eye_w * 0.15), 2)
    rx1 = max(0, int(min_xy[0]) - pad)
    ry1 = max(0, int(min_xy[1]) - pad)
    rx2 = min(W, int(max_xy[0]) + pad)
    ry2 = min(H, int(max_xy[1]) + pad)
    roi = img_gray[ry1:ry2, rx1:rx2]
    if roi.size < 20:
        return float(geo_center[0]), float(geo_center[1]), 0.1
    h_roi, w_roi = roi.shape

    # --- Eye contour mask ---
    pts_local = eye_pts_px.astype(np.int32).copy()
    pts_local[:, 0] -= rx1
    pts_local[:, 1] -= ry1
    mask_full = np.zeros((h_roi, w_roi), dtype=np.uint8)
    cv2.fillConvexPoly(mask_full, pts_local, 255)

    # --- Search the WHOLE lid opening (fixed 2026-07-31) ---
    # This used to erase the bottom 35 % of the aperture before looking for
    # the pupil:
    #     cutoff = int(eye_top_l + 0.65 * (eye_bot_l - eye_top_l))
    #     mask[cutoff:, :] = 0
    # The stated reason was eyelash/lid-shadow contamination, but the effect is
    # that a pupil in the lower third of the eye — i.e. ANY downward gaze —
    # sits in the deleted region and cannot be found. The search then returns
    # the darkest point still inside the upper 65 %, which is ABOVE the true
    # pupil, so dy = iris_y - eye_centre_y came out NEGATIVE and the arrow
    # pointed UP on a subject looking DOWN. Downward gaze was structurally
    # undetectable, in every engine that falls back to this finder.
    # Lash/shadow rejection is the darkness-weighting's job, not the search
    # region's.
    mask = mask_full

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_inner = cv2.erode(mask, kern, iterations=2)
    if np.count_nonzero(mask_inner) < 5:
        mask_inner = cv2.erode(mask, kern, iterations=1)
    if np.count_nonzero(mask_inner) < 5:
        mask_inner = mask

    # --- CLAHE + gentle blur ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    roi_eq = clahe.apply(roi)
    roi_blur = cv2.GaussianBlur(roi_eq, (3, 3), 0.7)

    masked_pixels = roi_blur[mask_inner > 0]
    if len(masked_pixels) < 5:
        return float(geo_center[0]), float(geo_center[1]), 0.1

    gc_local = geo_center - np.array([rx1, ry1])
    mask_area = float(max(np.count_nonzero(mask_inner), 1))

    # ================================================================
    # Strategy 1 – gradient-based centre voting  (primary)
    # ================================================================
    grad = _gradient_vote_pupil(roi_blur, mask_inner, gc_local, eye_w, eye_h)
    if grad is not None:
        gcx, gcy, gscore = grad
        if gscore > 0.20:
            conf = float(np.clip(ear * 2.5 * gscore, 0.1, 1.0))
            return float(gcx) + rx1, float(gcy) + ry1, conf

    # ================================================================
    # Strategy 2 – multi-threshold contour moments  (secondary)
    # ================================================================
    best_cx, best_cy, best_score = None, None, -1.0
    for pct in (10, 20, 30, 40):
        thresh_val = int(np.percentile(masked_pixels, pct))
        binary = np.zeros_like(roi_blur)
        binary[(roi_blur <= thresh_val) & (mask_inner > 0)] = 255
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kern)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 4:
                continue
            M = cv2.moments(cnt)
            if M["m00"] < 1:
                continue
            cx_l = M["m10"] / M["m00"]
            cy_l = M["m01"] / M["m00"]
            ix, iy = int(cx_l), int(cy_l)
            if not (0 <= ix < w_roi and 0 <= iy < h_roi):
                continue
            if mask_full[iy, ix] == 0:
                continue

            # Circularity
            perim = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * area / (perim ** 2 + 1e-6)
            circ_score = min(circ, 1.0)

            # Proximity – asymmetric vertical penalty
            dx = abs(cx_l - gc_local[0])
            dy = cy_l - gc_local[1]          # positive = below centre
            max_dx = max(eye_w * 0.45, 1)
            h_prox = max(0.0, 1.0 - dx / max_dx)
            max_dy_up = max(eye_h * 0.4, 1)
            max_dy_dn = max(eye_h * 0.25, 1)  # tighter below
            v_prox = max(0.0, 1.0 - abs(dy) / (max_dy_dn if dy > 0 else max_dy_up))
            prox_score = 0.5 * h_prox + 0.5 * v_prox

            # Size
            ratio = area / mask_area
            if ratio < 0.03 or ratio > 0.70:
                size_score = 0.1
            elif 0.08 <= ratio <= 0.45:
                size_score = 1.0
            else:
                size_score = 0.5

            score = circ_score * 0.25 + prox_score * 0.45 + size_score * 0.30
            if score > best_score:
                best_score = score
                best_cx = cx_l + rx1
                best_cy = cy_l + ry1

    if best_cx is not None and best_score > 0.25:
        conf = float(np.clip(ear * 2.5 * best_score, 0.1, 1.0))
        return float(best_cx), float(best_cy), conf

    # ================================================================
    # Strategy 3 – weighted dark-pixel centroid with upper-region bias
    # ================================================================
    thresh_val = int(np.percentile(masked_pixels, 25))
    dark = (roi_blur <= thresh_val) & (mask_inner > 0)
    ys, xs = np.where(dark)
    if len(xs) > 3:
        weights = (255.0 - roi_blur[dark]).astype(np.float64)
        vert_bias = np.clip(
            1.0 - (ys - gc_local[1]) / max(eye_h * 0.3, 1), 0.3, 1.5)
        weights *= vert_bias
        total = weights.sum()
        if total > 0:
            cx = float(np.sum(xs * weights) / total) + rx1
            cy = float(np.sum(ys * weights) / total) + ry1
            return cx, cy, float(np.clip(ear * 1.5, 0.1, 0.7))

    # --- Fallback: geometric centre ---
    return float(geo_center[0]), float(geo_center[1]), 0.1


def estimate_iris_positions(face_kps, image_np, img_width, img_height):
    """Estimate iris centres for both eyes using image-based pupil detection.

    Args:
        face_kps: (69, 3) normalised keypoints [x/W, y/H, conf]
        image_np: (H, W, 3) float32 RGB image [0, 1]
        img_width, img_height: pixel dimensions

    Returns:
        dict with right_iris, left_iris, right_gaze, left_gaze.
    """
    W, H = img_width, img_height
    kps_px = face_kps[:, :2].copy() * np.array([W, H])
    kps_conf = face_kps[:, 2].copy()

    img_u8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    img_gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)

    results = {}
    for eye_name, eye_idx in [('right', _RIGHT_EYE_IDX),
                               ('left', _LEFT_EYE_IDX)]:
        pts = kps_px[eye_idx]      # (6, 2)
        confs = kps_conf[eye_idx]
        mc = float(np.mean(confs))
        geo = np.mean(pts, axis=0)

        if mc < 0.05:
            results[f'{eye_name}_iris'] = {
                'x': float(geo[0]), 'y': float(geo[1]), 'confidence': 0.0}
            results[f'{eye_name}_gaze'] = {'dx': 0.0, 'dy': 0.0}
            continue

        ix, iy, ic = _find_pupil_center(pts, img_gray, W, H)
        results[f'{eye_name}_iris'] = {'x': ix, 'y': iy, 'confidence': ic}

        dx = ix - float(geo[0])
        dy = iy - float(geo[1])
        norm = max(np.hypot(dx, dy), 1e-6)
        # Approximate yaw/pitch from 2D iris-offset (small-angle, scaled by
        # eye span). This lets the downstream gaze-lock + OneEuro paths
        # work even when MediaPipe blendshapes are unavailable.
        eye_span_x = float(np.ptp(pts[:, 0])) or 1.0
        eye_span_y = float(np.ptp(pts[:, 1])) or 1.0
        yaw_rad = np.clip(dx / (eye_span_x * 0.5), -1.2, 1.2) * 0.5  # ~30 deg max
        pitch_rad = np.clip(dy / (eye_span_y * 0.5), -1.2, 1.2) * 0.4  # ~25 deg max
        results[f'{eye_name}_gaze'] = {
            'dx': round(dx / norm, 4), 'dy': round(dy / norm, 4),
            'yaw_rad': float(yaw_rad), 'pitch_rad': float(pitch_rad),
            'magnitude': float(np.hypot(yaw_rad, pitch_rad)),
            'source': 'iris_offset_2d',
        }

    return results


# ---------------------------------------------------
# Debug visualisation overlay
# ---------------------------------------------------
# Colour palette for 68-landmark regions (face-array index ranges)
_LANDMARK_COLORS = [
    (1,  17, (255, 200, 0)),    # jawline
    (18, 22, (200, 255, 0)),    # right eyebrow
    (23, 27, (200, 255, 0)),    # left eyebrow
    (28, 36, (0, 0, 255)),      # nose
    (37, 42, (0, 255, 0)),      # right eye
    (43, 48, (0, 255, 0)),      # left eye
    (49, 60, (0, 255, 255)),    # outer mouth
    (61, 68, (0, 255, 200)),    # inner mouth
]


def draw_debug_overlay(frame_uint8, face_kps_norm, iris_data,
                       face_bbox, body_bbox, W, H):
    """Draw face landmarks, iris positions and bounding boxes for debugging.

    Args:
        frame_uint8:   (H, W, 3) uint8 RGB
        face_kps_norm: (69, 3) normalised keypoints
        iris_data:     dict from estimate_iris_positions
        face_bbox:     (x1, x2, y1, y2) or None
        body_bbox:     [x1, y1, x2, y2, ...] array or None
        W, H:          image pixel dimensions

    Returns:
        vis: (H, W, 3) uint8 RGB image with annotations
    """
    vis = frame_uint8.copy()
    kps_px = face_kps_norm[:, :2] * np.array([W, H])
    kps_conf = face_kps_norm[:, 2]

    # --- Face landmarks ---
    for idx in range(1, min(69, len(kps_px))):
        if kps_conf[idx] < 0.05:
            continue
        x, y = int(kps_px[idx, 0]), int(kps_px[idx, 1])
        if not (0 <= x < W and 0 <= y < H):
            continue
        color = (180, 180, 180)
        for lo, hi, c in _LANDMARK_COLORS:
            if lo <= idx <= hi:
                color = c
                break
        cv2.circle(vis, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, (x, y), 2, color, -1, cv2.LINE_AA)

    # --- Eye contour polylines ---
    for eye_indices in _EYE_CONTOUR_INDICES:
        pts = []
        for i in eye_indices:
            if i < len(kps_px) and kps_conf[i] > 0.05:
                pts.append([int(kps_px[i, 0]), int(kps_px[i, 1])])
        if len(pts) >= 4:
            cv2.polylines(vis, [np.array(pts, np.int32)], True,
                          (0, 255, 0), 1, cv2.LINE_AA)

    # --- Iris markers + gaze arrows ---
    for eye_key, gaze_key in [('right_iris', 'right_gaze'),
                               ('left_iris', 'left_gaze')]:
        iris = iris_data.get(eye_key)
        gaze = iris_data.get(gaze_key)
        if iris is None or iris['confidence'] < 0.05:
            continue
        ix, iy = int(iris['x']), int(iris['y'])
        if 0 <= ix < W and 0 <= iy < H:
            cv2.drawMarker(vis, (ix, iy), (255, 0, 255),
                           cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            cv2.circle(vis, (ix, iy), 5, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"{iris['confidence']:.2f}",
                        (ix + 8, iy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (255, 0, 255), 1, cv2.LINE_AA)
        if gaze and (abs(gaze['dx']) > 1e-4 or abs(gaze['dy']) > 1e-4):
            # Magnitude-aware arrow length: scale by gaze strength so
            # noise-level offsets shrink toward _GAZE_MIN_ARROW_LEN_PX.
            # dx/dy already carry the magnitude, so use a FIXED gain here -
            # scaling by magnitude_norm again would square it.
            arrow_len = int(_GAZE_MAX_ARROW_LEN_PX)
            ex = int(ix + gaze['dx'] * arrow_len)
            ey = int(iy + gaze['dy'] * arrow_len)
            cv2.arrowedLine(vis, (ix, iy), (ex, ey),
                            (0, 200, 255), 2, cv2.LINE_AA, tipLength=0.3)

    # --- Face bounding box (cyan) ---
    if face_bbox is not None:
        x1, x2, y1, y2 = face_bbox
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                      (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, "FACE", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    # --- Body bounding box (green) ---
    if body_bbox is not None:
        bb = np.asarray(body_bbox).flatten()
        if len(bb) >= 4:
            cv2.rectangle(vis, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                          (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, "BODY", (int(bb[0]), int(bb[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    return vis


# ---------------------------------------------------
# ONNX model loader
# ---------------------------------------------------
class OnnxDetectionModelLoaderV2:
    DESCRIPTION = (
        "Load ONNX ViTPose + YOLO detection models for Wan 2.2 Animate "
        "preprocessing. Place model files in `ComfyUI/models/detection/`. "
        "Outputs a `POSEMODEL` bundle that the detection node consumes."
    )

    @classmethod
    def INPUT_TYPES(s):
        files = list_onnx_detection_models()
        lf = lambda x: x.lower()
        onnx = [f for f in files if lf(f).endswith(".onnx")]
        # Smart defaults so the node works out-of-the-box instead of defaulting
        # both slots to the alphabetically-first file (often an animal/apt36k
        # pose model, which makes YOLO error and ViTPose produce empty skeletons).
        vit_default = (next((f for f in onnx if "wholebody" in lf(f)), None)
                       or next((f for f in onnx if "vitpose" in lf(f)
                                and "apt" not in lf(f) and "animal" not in lf(f)), None)
                       or next((f for f in onnx if "vitpose" in lf(f)), None)
                       or (onnx[0] if onnx else (files[0] if files else "")))
        yolo_default = (next((f for f in onnx if "yolov10" in lf(f)), None)
                        or next((f for f in onnx if ("yolov8" in lf(f) or "yolox" in lf(f) or "yolo11" in lf(f))
                                 and "face" not in lf(f) and "pose" not in lf(f)), None)
                        or next((f for f in onnx if "yolo" in lf(f)
                                 and "face" not in lf(f) and "pose" not in lf(f) and "vitpose" not in lf(f)), None)
                        or (files[0] if files else ""))
        return {
            "required": {
                "vitpose_model": (files, {"default": vit_default, "tooltip": "ViTPose ONNX file (human wholebody, e.g. vitpose_h_wholebody_model.onnx). Place in ComfyUI/models/detection/. .onnx is always listed here even if ComfyUI hides it elsewhere."}),
                "yolo_model":    (files, {"default": yolo_default, "tooltip": "YOLO person-detector ONNX file (e.g. yolov10m.onnx — NOT a pose model). Place in ComfyUI/models/detection/. .onnx always listed."}),
                "onnx_device":   (["CUDAExecutionProvider", "CPUExecutionProvider"], {"default": "CUDAExecutionProvider", "tooltip": "Execution provider for ONNX Runtime. CUDA is much faster; CPU is the safe fallback."}),
            },
        }

    RETURN_TYPES = ("POSEMODEL",)
    RETURN_NAMES = ("model", )
    OUTPUT_TOOLTIPS = ("ViTPose+YOLO model bundle. Connect to `model` on Pose and Face Detection (V2).",)
    FUNCTION = "loadmodel"
    CATEGORY = "WanAnimatePreprocess_V2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def loadmodel(self, vitpose_model, yolo_model, onnx_device):
        vitpose_model_path = _resolve_detection_path(vitpose_model)
        yolo_model_path = _resolve_detection_path(yolo_model)

        # ─── Pre-flight: validate that each file has the expected input shape ──
        # The detection pipeline hard-codes input sizes per role:
        #   YOLO  → 640 × 640   (square person detector)
        #   ViTPose → 256 × 192 (portrait pose-keypoint heatmap)
        # When the user accidentally picks a ViTPose checkpoint for the YOLO
        # slot (or vice-versa), onnxruntime raises a cryptic INVALID_ARGUMENT
        # at first inference, several seconds into the workflow. We surface
        # the mismatch up-front with a clear error, and auto-swap the picks
        # if they look transposed.
        import onnxruntime as _ort
        import logging as _logging
        _log = _logging.getLogger("WanAnimateV2.OnnxLoader")

        def _peek_input_hw(path):
            try:
                sess = _ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                shp = sess.get_inputs()[0].shape
                # ONNX shape is [N, C, H, W] — extract last two dims; cast str→None.
                h = shp[-2] if isinstance(shp[-2], int) else None
                w = shp[-1] if isinstance(shp[-1], int) else None
                del sess
                return h, w
            except Exception as e:
                _log.warning("Could not peek %s: %s", path, e)
                return None, None

        vp_h, vp_w = _peek_input_hw(vitpose_model_path)
        yl_h, yl_w = _peek_input_hw(yolo_model_path)

        looks_swapped = (
            yl_h is not None and yl_w is not None
            and (yl_h, yl_w) == (256, 192)
            and (vp_h, vp_w) == (640, 640)
        )
        if looks_swapped:
            _log.warning(
                "Detection model picks look swapped: "
                "vitpose_model=%r is 640x640 (YOLO shape), "
                "yolo_model=%r is 256x192 (ViTPose shape). Auto-swapping.",
                vitpose_model, yolo_model,
            )
            vitpose_model_path, yolo_model_path = yolo_model_path, vitpose_model_path
            vp_h, vp_w, yl_h, yl_w = yl_h, yl_w, vp_h, vp_w

        if yl_h is not None and yl_w is not None and (yl_h, yl_w) != (640, 640):
            raise RuntimeError(
                f"yolo_model {yolo_model!r} has ONNX input shape "
                f"[N,C,{yl_h},{yl_w}], but the YOLO detection path expects "
                f"[N,C,640,640]. Pick a real YOLO person-detector ONNX "
                f"(e.g. yolov8n.onnx / yoloxpose-s.onnx). The file you "
                f"selected looks like a pose model "
                f"({'ViTPose' if (yl_h, yl_w) == (256, 192) else 'unknown'})."
            )
        if vp_h is not None and vp_w is not None and (vp_h, vp_w) != (256, 192):
            raise RuntimeError(
                f"vitpose_model {vitpose_model!r} has ONNX input shape "
                f"[N,C,{vp_h},{vp_w}], but the ViTPose path expects "
                f"[N,C,256,192]. Pick a real ViTPose ONNX "
                f"(e.g. vitpose-h.onnx / vitpose-l.onnx). The file you "
                f"selected looks like a "
                f"{'YOLO detector' if (vp_h, vp_w) == (640, 640) else 'different model'}."
            )

        vitpose = ViTPose(vitpose_model_path, onnx_device)
        yolo = Yolo(yolo_model_path, onnx_device)
        return ({"vitpose": vitpose, "yolo": yolo},)


# ---------------------------------------------------
# Jitterless face-crop helpers
# ---------------------------------------------------
def _parse_keyframes_json(s, B):
    """Return a list of dicts {frame, cx, cy, size?} sorted by frame.

    Tolerates malformed input — bad entries are skipped with a warning.
    """
    if not s or not isinstance(s, str):
        return []
    try:
        raw = json.loads(s)
    except Exception as e:
        print(f"[PoseAndFaceDetectionV2] keyframes_json parse error: {e}; ignoring.")
        return []
    if not isinstance(raw, list):
        return []
    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            f = int(entry.get("frame", -1))
            cx = float(entry.get("cx"))
            cy = float(entry.get("cy"))
        except (TypeError, ValueError):
            continue
        if f < 0 or f >= B:
            continue
        kf = {"frame": f, "cx": cx, "cy": cy}
        if "size" in entry and entry["size"] is not None:
            try:
                kf["size"] = int(entry["size"])
            except (TypeError, ValueError):
                pass
        out.append(kf)
    out.sort(key=lambda e: e["frame"])
    return out


def _interp_keyframes(keyframes, B, default_cx, default_cy, default_size):
    """Densify sorted keyframes into per-frame (cx, cy, size) arrays.

    Frames before the first keyframe hold the first; after the last hold
    the last; in-between frames are linearly interpolated.
    Returns three np.ndarray of length B (or None if no keyframes).
    """
    if not keyframes:
        return None, None, None
    cx = np.full(B, default_cx, dtype=np.float32)
    cy = np.full(B, default_cy, dtype=np.float32)
    sz = np.full(B, default_size, dtype=np.float32)
    # spread cx/cy across all frames
    frames = np.array([k["frame"] for k in keyframes], dtype=np.int32)
    cxs = np.array([k["cx"] for k in keyframes], dtype=np.float32)
    cys = np.array([k["cy"] for k in keyframes], dtype=np.float32)
    sizes_known = np.array([k.get("size", -1) for k in keyframes], dtype=np.float32)
    xs = np.arange(B, dtype=np.float32)
    cx = np.interp(xs, frames, cxs).astype(np.float32)
    cy = np.interp(xs, frames, cys).astype(np.float32)
    if np.any(sizes_known > 0):
        # only interpolate sizes between keyframes that actually specify one
        mask = sizes_known > 0
        sz_known = sizes_known[mask]
        f_known = frames[mask].astype(np.float32)
        if f_known.size >= 2:
            sz = np.interp(xs, f_known, sz_known).astype(np.float32)
        elif f_known.size == 1:
            sz = np.full(B, float(sz_known[0]), dtype=np.float32)
    return cx, cy, sz


def _gaussian_window(window):
    """1D Gaussian kernel (length=window, odd) suitable for np.convolve."""
    window = max(3, int(window) | 1)  # force odd and >= 3
    sigma = max(0.5, window / 6.0)
    xs = np.arange(window) - (window - 1) / 2.0
    k = np.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def _smooth_centers(centers_xy, method, *, ema_strength=0.6, image_diag=1.0,
                    one_euro_min_cutoff=1.0, one_euro_beta=0.05,
                    gaussian_window=7, zero_phase=True):
    """Apply a temporal filter to a (B, 2) array of (cx, cy)."""
    centers_xy = np.asarray(centers_xy, dtype=np.float32)
    if centers_xy.ndim != 2 or centers_xy.shape[0] < 2 or method == "none":
        return centers_xy.copy()

    # ZERO-PHASE (2026-07-29). ema and one_euro are CAUSAL: each output only
    # sees the past, so on any sustained motion the filtered centre trails the
    # real one by a roughly constant amount. That lag is not jitter — it is a
    # steady positional bias, and it lands in face_images as the face sitting
    # consistently to one side of the tile for the whole move. Measured on a
    # left-to-right pan: 21px mean horizontal offset in a 512 tile for the EMA
    # path, 18px for one_euro, in the SAME direction on every frame.
    #
    # This node is not a streaming filter — the whole clip is already in
    # memory — so there is no reason to accept causal lag. Running the filter
    # forwards and backwards and averaging cancels the phase error exactly
    # (the standard filtfilt trick) while keeping the jitter rejection: the
    # backward pass lags by the same amount in the opposite direction.
    # 'gaussian' is already symmetric, hence zero-phase, so it is excluded.
    if zero_phase and method in ("ema", "one_euro"):
        _kw = dict(ema_strength=ema_strength, image_diag=image_diag,
                   one_euro_min_cutoff=one_euro_min_cutoff,
                   one_euro_beta=one_euro_beta,
                   gaussian_window=gaussian_window, zero_phase=False)
        fwd = _smooth_centers(centers_xy, method, **_kw)
        bwd = _smooth_centers(centers_xy[::-1], method, **_kw)[::-1]
        return (0.5 * (fwd + bwd)).astype(np.float32)

    if method == "ema":
        out = np.empty_like(centers_xy)
        out[0] = centers_xy[0]
        norm = max(1.0, image_diag)
        base = float(np.clip(ema_strength, 0.0, 1.0))
        for i in range(1, len(centers_xy)):
            curr = centers_xy[i]
            prev = out[i - 1]
            motion = float(np.mean(np.abs(curr - prev)) / norm)
            dyn = base * np.exp(-motion * 5.0)
            alpha = 1.0 - dyn
            out[i] = alpha * curr + (1.0 - alpha) * prev
        return out

    if method == "gaussian":
        k = _gaussian_window(gaussian_window)
        # reflect-pad so the ends don't darken
        pad = len(k) // 2
        padded = np.pad(centers_xy, ((pad, pad), (0, 0)), mode="edge")
        out = np.empty_like(centers_xy)
        out[:, 0] = np.convolve(padded[:, 0], k, mode="valid")
        out[:, 1] = np.convolve(padded[:, 1], k, mode="valid")
        return out

    # default: one_euro
    _OEF = None
    if _GAZE_BS_IMPORTED and _gaze_bs is not None:
        _OEF = getattr(_gaze_bs, "OneEuroFilter", None)
    if _OEF is None:
        # Graceful fallback. Must forward zero_phase: without it the fallback
        # silently re-enabled zero-phase on a caller that explicitly asked for
        # the causal filter, so the flag did nothing whenever OneEuroFilter was
        # unavailable.
        return _smooth_centers(centers_xy, "ema",
                               ema_strength=ema_strength,
                               image_diag=image_diag,
                               gaussian_window=gaussian_window,
                               zero_phase=zero_phase)
    fx = _OEF(freq=30.0, min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
    fy = _OEF(freq=30.0, min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
    out = np.empty_like(centers_xy)
    for i in range(len(centers_xy)):
        out[i, 0] = fx(float(centers_xy[i, 0]))
        out[i, 1] = fy(float(centers_xy[i, 1]))
    return out


def _smooth_1d(values, method, *, ema_strength=0.6, scale_norm=1.0,
               one_euro_min_cutoff=1.0, one_euro_beta=0.05,
               gaussian_window=7, zero_phase=True):
    """Temporal filter for a 1-D scalar series (e.g. per-frame crop sizes).

    Same methods as _smooth_centers: one_euro (default), ema, gaussian, none.
    ``scale_norm`` normalises the motion magnitude for the adaptive EMA.
    """
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 2 or method == "none":
        return values.copy()

    # Zero-phase, for the same reason as _smooth_centers: a causal filter on
    # the crop SIZE makes the tile trail the subject's real scale change, so
    # the face keeps drifting toward/away from filling the tile during a
    # dolly or an approach. See the long note there.
    if zero_phase and method in ("ema", "one_euro"):
        _kw = dict(ema_strength=ema_strength, scale_norm=scale_norm,
                   one_euro_min_cutoff=one_euro_min_cutoff,
                   one_euro_beta=one_euro_beta,
                   gaussian_window=gaussian_window, zero_phase=False)
        fwd = _smooth_1d(values, method, **_kw)
        bwd = _smooth_1d(values[::-1], method, **_kw)[::-1]
        return (0.5 * (fwd + bwd)).astype(np.float32)

    if method == "gaussian":
        k = _gaussian_window(gaussian_window)
        pad = len(k) // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        return np.convolve(padded, k, mode="valid").astype(np.float32)

    if method == "ema":
        out = np.empty_like(values)
        out[0] = values[0]
        norm = max(1.0, float(scale_norm))
        base = float(np.clip(ema_strength, 0.0, 1.0))
        for i in range(1, len(values)):
            motion = abs(float(values[i]) - float(out[i - 1])) / norm
            dyn = base * np.exp(-motion * 5.0)
            out[i] = (1.0 - dyn) * float(values[i]) + dyn * float(out[i - 1])
        return out

    # default: one_euro
    _OEF = None
    if _GAZE_BS_IMPORTED and _gaze_bs is not None:
        _OEF = getattr(_gaze_bs, "OneEuroFilter", None)
    if _OEF is None:
        # forward zero_phase for the same reason as _smooth_centers
        return _smooth_1d(values, "ema", ema_strength=ema_strength,
                          scale_norm=scale_norm,
                          gaussian_window=gaussian_window,
                          zero_phase=zero_phase)
    f = _OEF(freq=30.0, min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
    out = np.empty_like(values)
    for i in range(len(values)):
        out[i] = f(float(values[i]))
    return out


# Jitterless: how far (as a fraction of the locked crop size) the smoothed
# centre is allowed to lag behind the true target centre. A temporal filter
# ALWAYS lags on fast motion; without a bound that lag is unbounded and the
# face slides toward — eventually out of — the crop edge. 0.12 keeps the face
# centre inside the middle ~24% of the tile, which is imperceptible as
# "off-centre" but still lets the filter absorb per-frame detector jitter.
_JITTERLESS_MAX_CENTER_DRIFT_FRAC = 0.12


def _locked_crop_side(face_box_size_px, raw_face_bboxes, W, H, mode_label):
    """Side length for the constant-size crop modes (auto / jitterless).

    ``face_box_size_px`` is an ABSOLUTE pixel size, and that is the problem it
    exists to solve here. Its old default of 512 has nothing to do with the
    footage: on a 832x480 clip it clamps to min(W,H)=480, so the node cut a
    480px window around a face that the detector measured at ~125px. The face
    then occupied ~26% of the tile and the remaining 74% was background — and
    after the stretch to 512x512 the face carried roughly a QUARTER of the
    pixels it should. That is fatal for micro-expressions, which are a few
    pixels of eyelid and mouth-corner movement, and it also reads as "the face
    is off to one side" because a small subject inside a large window drifts
    visibly while a face-tight crop cannot.

    The reference pipeline never does this: it crops face-TIGHT per frame
    (get_face_bboxes -> that frame's own box), so the face fills the tile by
    construction. The constant-size modes exist to stop the box breathing, not
    to detach it from the subject's actual size — so derive the locked side
    from the DETECTED face and then hold it fixed for the whole clip.

    Returns the median detected face side (robust to a few bad frames), which
    keeps the reference's face-fill while still being one constant number.
    ``face_box_size_px > 0`` remains an explicit override.
    """
    if face_box_size_px and int(face_box_size_px) > 0:
        return float(face_box_size_px)
    sides = [max(float(b[1] - b[0]), float(b[3] - b[2]))
             for b in (raw_face_bboxes or []) if b is not None]
    if not sides:
        return float(min(W, H)) * 0.5
    side = float(np.median(sides))
    side = float(np.clip(side, 64.0, float(min(W, H))))
    logging.getLogger(__name__).info(
        "PoseAndFaceDetectionV2 [%s]: face_box_size_px=0 (auto) -> locked crop "
        "side %.0fpx, derived from the median DETECTED face box over %d frames. "
        "The face fills ~%.0f%% of the tile (a fixed 512 would have given ~%.0f%%).",
        mode_label, side, len(sides), 100.0,
        100.0 * side / max(float(min(W, H)), 1.0),
    )
    return side


# Minimum ViTPose confidence for a face landmark to be trusted when MEASURING
# the face box. Deliberately低 — 0.20 — because we only need to exclude the
# genuinely wild points, not the merely soft ones: over-filtering shrinks the
# box and crops the jaw. pose_threshold (a user widget) zeroes confidence but
# NOT the coordinate, so without this the zeroed points still vote in min/max.
_FACE_KP_MIN_CONF = 0.20


def _face_pin_ring(pts, w, h, n=24, pad=1.18):
    """A closed ring of anchor points just outside the face landmarks.

    Any piecewise-affine face warp has to answer: what happens to the pixels
    that are NOT part of the face? The 68-point iBUG contour cannot answer it,
    because points 0-16 trace an open ARC along the jaw and there are no
    forehead points at all — the face is unbounded at the top. Triangulate that
    against a pinned image border and the triangles covering forehead, hair,
    ears and background have moving vertices, so amplifying an expression drags
    the whole picture. That is what makes a warped tile read as "the entire
    image is warped" rather than "the mouth moved".

    Emitting this ring identically in src and dst seals the face inside a
    closed, fixed boundary: every triangle that touches anything beyond the
    face has all three vertices pinned and therefore cannot move at all.
    """
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    cx, cy = float(p[:, 0].mean()), float(p[:, 1].mean())
    rx = max(float(np.abs(p[:, 0] - cx).max()) * pad, 8.0)
    ry = max(float(np.abs(p[:, 1] - cy).max()) * pad, 8.0)
    a = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    ring = np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], axis=1)
    ring[:, 0] = np.clip(ring[:, 0], 0, w - 1)
    ring[:, 1] = np.clip(ring[:, 1], 0, h - 1)
    return ring.astype(np.float32)


def _crop_with_padding(frame, x1, x2, y1, y2):
    """Crop ``frame`` to the (possibly out-of-frame) box, edge-padding as needed.

    Returns exactly (y2-y1, x2-x1) pixels ALWAYS, so a crop box that hangs off
    the frame still yields a full-size tile with the requested centre at the
    tile's centre. Clamping the box back inside the frame instead — the old
    behaviour — silently moves the subject off-centre, which is the failure
    this exists to prevent.

    Edge-replicate (not black) padding: the face encoder was trained with
    scale/colour/noise augmentation on real face crops, so a hard synthetic
    border is further out of distribution than smeared edge pixels.
    """
    h, w = frame.shape[:2]
    x1i, x2i, y1i, y2i = int(x1), int(x2), int(y1), int(y2)
    tw, th = max(0, x2i - x1i), max(0, y2i - y1i)
    if tw == 0 or th == 0:
        return frame[0:0, 0:0]
    # Fast path: fully inside.
    if x1i >= 0 and y1i >= 0 and x2i <= w and y2i <= h:
        return frame[y1i:y2i, x1i:x2i]
    sx1, sy1 = max(0, x1i), max(0, y1i)
    sx2, sy2 = min(w, x2i), min(h, y2i)
    if sx2 <= sx1 or sy2 <= sy1:
        # Box lies entirely outside the frame — replicate the nearest pixel.
        cy = min(max(0, (y1i + y2i) // 2), h - 1)
        cx = min(max(0, (x1i + x2i) // 2), w - 1)
        return np.repeat(np.repeat(frame[cy:cy + 1, cx:cx + 1], th, axis=0), tw, axis=1)
    inner = frame[sy1:sy2, sx1:sx2]
    pad_top, pad_left = sy1 - y1i, sx1 - x1i
    pad_bot, pad_right = y2i - sy2, x2i - sx2
    pads = [(pad_top, pad_bot), (pad_left, pad_right)]
    if inner.ndim == 3:
        pads.append((0, 0))
    return np.pad(inner, pads, mode="edge")


def _floor_face_boxes(bboxes, min_side: int, W: int, H: int):
    """Expand any crop whose longer side is under ``min_side`` source pixels.

    WHY: a 46px face Lanczos'd to 512 invents 99% of the encoder input
    (eyeballs ~4px). Wan-Animate trained the Face Adapter with *scale*
    augmentation, so a slightly looser crop is in-distribution; an 11×
    upsample of 46px is not. Expand around the same centre, keep aspect,
    do not clamp into the frame (``_crop_with_padding`` edge-pads). Close-ups
    already larger than ``min_side`` are unchanged (face still fills the tile).
    ``min_side`` is clamped to the shorter frame axis so a 480-tall plate
    cannot request a 512 crop.
    """
    cap = float(min(int(min_side), int(min(W, H))))
    if cap < 8 or not bboxes:
        return list(bboxes), 0
    out = []
    raised = 0
    for x1, x2, y1, y2 in bboxes:
        w = float(x2 - x1)
        h = float(y2 - y1)
        side = max(w, h, 1.0)
        if side >= cap:
            out.append((int(x1), int(x2), int(y1), int(y2)))
            continue
        raised += 1
        s = cap / side
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        nw, nh = w * s, h * s
        nx1 = int(round(cx - nw / 2.0))
        ny1 = int(round(cy - nh / 2.0))
        out.append((nx1, nx1 + int(round(nw)), ny1, ny1 + int(round(nh))))
    return out, raised


def build_jitterless_boxes(
    *,
    target_centers,
    target_sizes_raw,
    anchor_size,
    W, H,
    smoothing_method="one_euro",
    face_smoothing_strength=0.6,
    one_euro_min_cutoff=1.0,
    one_euro_beta=0.05,
    size_one_euro_beta=None,
    gaussian_window=7,
    lock_size=True,
    max_center_drift_frac=_JITTERLESS_MAX_CENTER_DRIFT_FRAC,
    hold_mask=None,
    safety_margin=1.0,
    containment_boxes=None,
    containment_tolerance=0.0,
    aspect_ratios=None,
):
    """Build the jitterless per-frame square crop boxes.

    Pure function (numpy in, numpy out) so the guarantees below can be
    asserted numerically without standing up ComfyUI.

    Guarantees when ``lock_size`` is True:
      * every returned box is EXACTLY ``round(anchor_size)`` px on both axes
        (subject only to the frame being at least that large), and
      * ``|smoothed_centre - target_centre| <= max_center_drift_frac * size``
        on every frame where the crop is not pinned against a frame edge.

    ``lock_size=False`` reproduces the legacy face-scale-preserving behaviour
    (crop size follows the detected face size), which is what ``auto`` wants.

    Returns ``(boxes, sizes, centers)`` where boxes are ``(x1, x2, y1, y2)``.
    """
    target_centers = np.asarray(target_centers, dtype=np.float32).reshape(-1, 2).copy()
    B = int(target_centers.shape[0])
    if B == 0:
        return [], np.zeros((0,), np.float32), np.zeros((0, 2), np.float32)

    max_side = float(min(W, H))
    margin = float(safety_margin) if safety_margin and safety_margin > 0 else 1.0
    if lock_size:
        # THE lock. One value, computed once, used for every frame. Rounding
        # happens ONCE here rather than per frame, so integer box widths cannot
        # wobble by +/-1px between frames.
        # safety_margin inflates it so filter lag / yaw-foreshortened
        # detections / expression-driven bbox growth cannot clip the face.
        size_const = float(np.clip(float(anchor_size) * margin, 8.0, max_side))
        sizes = np.full((B,), size_const, dtype=np.float32)
    else:
        sizes = np.clip(
            np.asarray(target_sizes_raw, dtype=np.float32).reshape(-1) * margin,
            8.0, max_side)
        # The SIZE trajectory is a different signal from the CENTRE trajectory:
        # position wants heavy damping (kill detector jitter), scale wants to
        # follow real zoom/approach or it under-sizes mid-move. Give size its
        # own beta instead of inheriting the centre's.
        _size_beta = one_euro_beta if size_one_euro_beta is None else float(size_one_euro_beta)
        sizes = _smooth_1d(
            sizes, method=str(smoothing_method),
            ema_strength=face_smoothing_strength, scale_norm=max(anchor_size, 1.0),
            one_euro_min_cutoff=one_euro_min_cutoff, one_euro_beta=_size_beta,
            gaussian_window=int(gaussian_window),
        )
        sizes = np.clip(sizes, 8.0, max_side)

    image_diag = float((W * W + H * H) ** 0.5)
    centers = _smooth_centers(
        target_centers, method=str(smoothing_method),
        ema_strength=face_smoothing_strength, image_diag=image_diag,
        one_euro_min_cutoff=one_euro_min_cutoff, one_euro_beta=one_euro_beta,
        gaussian_window=int(gaussian_window),
    )
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)

    # Bound the filter lag. Without this the face drifts off-centre for as
    # long as the motion lasts, which is exactly the "face slides to the edge
    # during fast motion" failure.
    if max_center_drift_frac is not None and max_center_drift_frac > 0:
        for i in range(B):
            limit = float(max_center_drift_frac) * float(sizes[i])
            if limit <= 0:
                continue
            dx = float(centers[i, 0] - target_centers[i, 0])
            dy = float(centers[i, 1] - target_centers[i, 1])
            dist = math.hypot(dx, dy)
            if dist > limit:
                k = limit / dist
                centers[i, 0] = target_centers[i, 0] + dx * k
                centers[i, 1] = target_centers[i, 1] + dy * k

    # Hold-last-known across frames with no detection.
    if hold_mask is not None:
        for i in range(1, B):
            if hold_mask[i]:
                centers[i] = centers[i - 1]
                if not lock_size:
                    sizes[i] = sizes[i - 1]

    # Containment: a HARD per-frame guarantee that the real detected face box
    # is inside the crop, not a heuristic.
    #
    # DESIGN NOTE — this deliberately does NOT grow the crop when the size is
    # locked. Growing one frame would silently break the exact-size guarantee
    # that is the entire point of jitterless. Instead:
    #   1. SHIFT the crop to contain the face (free — size is untouched), then
    #   2. only if the face is genuinely LARGER than the locked crop, report
    #      it. That is a "your locked size is too small for this shot"
    #      condition the user must fix with face_box_size_px /
    #      crop_safety_margin — silently resizing would hide it.
    # When the size is not locked (auto / key-framed), growing is fine and is
    # what happens.
    n_shifted = 0
    n_too_small = 0
    if containment_boxes is not None:
        tol = float(containment_tolerance)
        for i in range(min(B, len(containment_boxes))):
            fb = containment_boxes[i]
            if fb is None:
                continue
            fx1, fx2, fy1, fy2 = (float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3]))
            need_w = (fx2 - fx1) + 2.0 * tol
            need_h = (fy2 - fy1) + 2.0 * tol
            need = max(need_w, need_h)
            if need > float(sizes[i]):
                if lock_size:
                    n_too_small += 1
                else:
                    sizes[i] = float(np.clip(need, 8.0, max_side))
            # Re-centre so the face box sits inside the crop.
            half_i = float(sizes[i]) / 2.0
            cx_i, cy_i = float(centers[i, 0]), float(centers[i, 1])
            lo_x, hi_x = (fx2 + tol) - half_i, (fx1 - tol) + half_i
            lo_y, hi_y = (fy2 + tol) - half_i, (fy1 - tol) + half_i
            new_cx = min(max(cx_i, lo_x), hi_x) if lo_x <= hi_x else 0.5 * (fx1 + fx2)
            new_cy = min(max(cy_i, lo_y), hi_y) if lo_y <= hi_y else 0.5 * (fy1 + fy2)
            if abs(new_cx - cx_i) > 1e-3 or abs(new_cy - cy_i) > 1e-3:
                n_shifted += 1
            centers[i, 0], centers[i, 1] = new_cx, new_cy

    # Aspect (height/width). Wan-Animate's own crop is NOT square: its
    # get_face_bboxes expands the face box by an AREA factor and biases the
    # top edge 3x harder than the bottom (forehead/hair in, chin tight), so
    # a typical box is ~1.25 taller than wide, and that non-square tile is
    # what gets resized to 512x512 for the encoder. Forcing a square crop
    # here fed the Face Adapter a framing it was never trained on. Smoothing
    # the aspect (rather than taking it raw per frame) keeps it jitter-free.
    if aspect_ratios is None:
        aspects = np.ones((B,), dtype=np.float32)
    else:
        aspects = np.clip(
            np.asarray(aspect_ratios, dtype=np.float32).reshape(-1), 0.25, 4.0)
        if aspects.shape[0] != B:
            aspects = np.resize(aspects, (B,))
        aspects = _smooth_1d(
            aspects, method=str(smoothing_method),
            ema_strength=face_smoothing_strength, scale_norm=1.0,
            one_euro_min_cutoff=one_euro_min_cutoff,
            one_euro_beta=(one_euro_beta if size_one_euro_beta is None
                           else float(size_one_euro_beta)),
            gaussian_window=int(gaussian_window),
        )
        aspects = np.clip(aspects, 0.25, 4.0)
        if lock_size:
            # A locked crop must also hold a locked SHAPE, or the tile's
            # framing breathes frame to frame even though its width does not.
            aspects = np.full((B,), float(np.median(aspects)), dtype=np.float32)

    boxes = []
    for i in range(B):
        size_i_int = int(round(float(sizes[i])))
        size_i_int = max(8, min(size_i_int, int(max_side)))
        half = size_i_int / 2.0
        # NOT clamped into the frame (2026-07-24 bug fix). Clamping is what
        # broke centring: with a locked size close to a frame dimension the
        # box has zero room to move (H - side == 0) and gets pinned to the
        # edge, leaving the face permanently off-centre. The box is emitted
        # centred on the tracked point even when it hangs off the frame;
        # _crop_with_padding() edge-pads at extraction so the tile is still
        # full-size and the face is still dead-centre.
        # Aspect-preserving fit. If the aspect-derived height does not fit the
        # frame, scale BOTH sides down together — clamping the height alone
        # silently rewrites the aspect (measured: a 512 box at the paper's
        # 1.25 on 832x480 footage became 512x480 = 0.94, which is neither the
        # paper framing nor the old square).
        a_i = float(aspects[i])
        w_i = float(size_i_int)
        h_i = w_i * a_i
        if h_i > H:
            h_i = float(H)
            w_i = h_i / max(a_i, 1e-6)
        if w_i > W:
            w_i = float(W)
            h_i = w_i * a_i
        size_i_int = max(8, int(round(w_i)))
        h_i_int = max(8, int(round(h_i)))
        half = size_i_int / 2.0
        x1 = int(round(float(centers[i, 0] - half)))
        y1 = int(round(float(centers[i, 1] - h_i_int / 2.0)))
        boxes.append((x1, x1 + size_i_int, y1, y1 + h_i_int))
    build_jitterless_boxes.last_stats = {
        "shifted": n_shifted, "too_small": n_too_small, "frames": B,
    }
    return boxes, sizes, centers


# ---------------------------------------------------
# Pose and Face Detection
# ---------------------------------------------------
class PoseAndFaceDetectionV2:
    DESCRIPTION = (
        "Run YOLO person detection + ViTPose 2D keypoints + (optional) MediaPipe "
        "FaceMesh on a video tensor. Produces the full pose/face/iris bundle "
        "required by Wan 2.2 Animate Character Replacement workflows.\n\n"
        "Wan-Animate fidelity notes (spec 2.5-2.7, workflow-level — not "
        "something this node can enforce for you):\n"
        "2.5 Wan-Animate splices long generations in ~78-frame segments with "
        "1-5 frame temporal handoffs (WanVideoAnimateEmbeds.frame_window_size, "
        "default 77); a brief microexpression (often only 2-4 frames) landing "
        "exactly on a segment boundary risks being smoothed by the "
        "discard/resume splice. If a specific expression must survive, check "
        "where it falls relative to your frame_window_size and shift the cut "
        "(or duration) if needed.\n"
        "2.6 Feed this node NATIVE framerate footage. Don't downsample fps "
        "upstream (e.g. a LoadVideo 'force_rate' below the source fps) — the "
        "face branch's causal 1D-conv temporal downsampling further "
        "compresses an already-fps-reduced brief expression, and a 2-4 frame "
        "microexpression can be lost entirely before it ever reaches this node.\n"
        "2.7 If your workflow globally autocasts to fp8/fp16, the face "
        "encoder's small-magnitude motion-basis deltas are more vulnerable to "
        "quantization noise than the body/pose branch. As of this writing "
        "Kijai's ComfyUI-WanVideoWrapper has no per-module precision override "
        "for the Wan-Animate face branch specifically — if you need this, "
        "load the full checkpoint in fp16/fp32 rather than an fp8 quant."
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":  ("POSEMODEL", {"tooltip": "From ONNX Detection Model Loader (V2)."}),
                "images": ("IMAGE",     {"tooltip": "Video frames as an IMAGE batch (B,H,W,C float [0,1])."}),
                "width":  ("INT", {"default": 832, "min": 64, "max": 2048, "tooltip": "Target canvas width (px) used for retarget math. Match your Wan 2.2 latent size."}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "tooltip": "Target canvas height (px). Match your Wan 2.2 latent size."}),
                "detection_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "YOLO confidence threshold. Lower = more permissive person detection."}),
                "pose_threshold":      ("FLOAT", {"default": 0.3,  "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Per-keypoint score threshold. Below this a keypoint is treated as missing."}),
                # Enhancement options
                "use_clahe": ("BOOLEAN", {"default": True, "tooltip": "Apply CLAHE contrast enhancement for pose detection."}),
                "clahe_clip_limit": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 8.0, "step": 0.1, "tooltip": "CLAHE contrast-limit. Higher = stronger local contrast, which helps a flat/hazy or backlit shot but starts amplifying grain. 2.0 is the long-standing default; try 3-4 for genuinely flat footage. Only used when use_clahe is on."}),
                "clahe_grid_size": ("INT", {"default": 8, "min": 1, "max": 16, "tooltip": "CLAHE tile grid (NxN). Smaller = more global/gentler; larger = more aggressively local, which can rescue a face lost in shadow but may introduce tile seams. Only used when use_clahe is on."}),
                "detect_gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05, "tooltip": "Gamma applied to the DETECTOR's input only. >1 lifts shadows (a face crushed into darkness becomes detectable), <1 pulls down blown highlights. Applied BEFORE CLAHE so there is signal in range for CLAHE to equalise. 1.0 = off."}),
                "detect_white_balance": ("BOOLEAN", {"default": False, "tooltip": "Grey-world white balance on the DETECTOR's input only. Equalises the per-channel means to remove a colour cast (tungsten, underwater, heavy LUT). Skin tone drifting off-neutral is a common cause of low face-detection confidence. Runs first, so CLAHE is not amplifying a cast."}),
                "detect_denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Edge-preserving (bilateral) denoise on the DETECTOR's input only. For grainy/high-ISO or heavily compressed footage where noise costs keypoint precision. Bilateral rather than blur so ViTPose keeps the edges it localises from. Runs before sharpen so noise is never sharpened. 0 = off."}),
                "detect_sharpen": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Unsharp-mask amount on the DETECTOR's input only. Recovers landmark precision on soft/out-of-focus or upscaled footage. Runs after denoise. Overdoing it creates halos that pull landmarks toward edges — 0.3-0.6 is usually plenty. 0 = off."}),
                "detect_saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Chroma scale on the DETECTOR's input only. Slightly boosting saturation can separate skin from a similarly-lit background; dropping toward 0 makes detection effectively luma-only, which occasionally helps on heavily colour-graded footage. 1.0 = off."}),
                "use_blur_for_pose": ("BOOLEAN", {"default": False, "tooltip": "Apply Gaussian blur internally for YOLO and ViTPose BEFORE detection. Bug-fix (default was True): this softens the exact edges/fine detail ViTPose needs for keypoint precision, producing a visibly blurrier preview and a less accurate skeleton for every user until they discovered and disabled it. Only enable this for genuinely noisy/grainy source footage."}),
                "blur_radius": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1, "tooltip": "Gaussian blur kernel radius applied to the face mask edge to soften the boundary. Higher = wider feather. Kernel size = radius*2+1 px."}),
                "blur_sigma": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "Gaussian blur sigma (standard deviation) for the face mask feather. Higher sigma = softer falloff. Tune together with blur_radius."}),
                # Face smoothing
                "use_face_smoothing": ("BOOLEAN", {"default": True, "tooltip": "Smooth face bounding box center over time."}),
                "face_smoothing_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Higher = more smoothing"}),
                # Constant-size face box
                "use_constant_face_box": ("BOOLEAN", {"default": True, "tooltip": "Keep a constant pixel size face crop; position adapts."}),
                "face_crop_scale": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 3.0, "step": 0.05, "tooltip": "AREA expansion of the face box, passed straight to get_face_bboxes. 1.3 is the value Wan2.2's own process_pipepline.py uses at both call sites, so 1.3 = reference-exact. LOWER (1.1-1.2) crops tighter, which puts MORE pixels on the face after the 512 resize and is the single most effective knob for micro-expression detail; too low and a head turn can clip the jaw/ear. HIGHER gives more headroom and safety at the cost of face resolution. Applies to every crop_mode."}),
                "face_box_size_px": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 4, "tooltip": "Side of the constant-size face crop, in SOURCE pixels, for crop_mode=auto (with use_constant_face_box) and jitterless.\n\n0 = AUTO (recommended): the side is derived from the median DETECTED face box across the clip, then held constant for every frame. You get the reference pipeline's face-tight framing - the face FILLS the tile - while the size still never breathes, which is the whole point of these modes.\n\nA fixed value is an ABSOLUTE pixel size and is almost always wrong unless you know your footage: the old 512 default clamps to min(width,height), so on an 832x480 clip it cut a 480px window around a ~125px face. The face then filled about a quarter of the tile and the rest was background, throwing away roughly 3/4 of the resolution a micro-expression needs and making the subject look off to one side. Set a fixed value only to lock a specific framing across separate renders."}),
                # Iris estimation
                "use_iris_smoothing": ("BOOLEAN", {"default": True, "tooltip": "Temporally smooth iris pixel positions across frames. Reduces per-frame jitter that Wan 2.2 Animate's face encoder picks up and reproduces as wobbly gaze."}),
                "iris_smoothing_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "EMA mix weight when iris_smoothing_method='ema'. Higher = more smoothing, more lag. Ignored for one_euro / none."}),
                "iris_smoothing_method": (["one_euro", "ema", "none"], {"default": "one_euro", "tooltip": "Iris pixel-position smoother. one_euro = adaptive low-pass (Casiez 2012, recommended). ema = legacy first-order; tweak via iris_smoothing_strength. none = raw per-frame positions."}),
                "iris_one_euro_min_cutoff": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 10.0, "step": 0.05, "tooltip": "One-euro min cutoff (Hz) for iris pixel coords. Lower = stronger jitter rejection on near-static eyes (small saccades preserved)."}),
                "iris_one_euro_beta": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "One-euro speed coefficient for iris pixel coords. Higher = filter relaxes faster on quick eye movements; lower = stronger steady-state smoothing."}),
                # Cross-eye coupling (NEW: directly fixes 'eyeballs not in same direction').
                "gaze_lock_eyes": ("BOOLEAN", {"default": True, "tooltip": "Couple left & right eye gaze so they always look in the SAME direction. Both eyes' yaw/pitch are blended toward their per-frame average. Single most effective fix for the 'eyes pointing different directions' artefact in Wan 2.2 Animate output."}),
                "gaze_lock_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "How strongly to pull each eye toward the shared average. 0 = independent (legacy). 1 = perfectly conjugate (both eyes always parallel). 0.7 keeps a touch of natural convergence/divergence."}),
                # MediaPipe face mesh (high-fidelity iris/lip tracking, falls back to ViTPose if unavailable)
                "use_mediapipe_face": ("BOOLEAN", {"default": True, "tooltip": "Use MediaPipe FaceMesh (478 pts incl. iris/lips) to override face landmarks. Falls back to ViTPose pupil voting if MediaPipe is missing or fails on a frame."}),
                # Production gaze (ARKit blend shapes via FaceLandmarker Tasks API)
                "use_blendshape_gaze": ("BOOLEAN", {"default": True, "tooltip": "Use MediaPipe FaceLandmarker (Tasks API) blend shapes for production-grade per-eye yaw/pitch in radians. Head-pose-corrected by training. Auto-downloads face_landmarker.task (~3MB) on first run. Falls back to legacy 2D iris-offset gaze if disabled or unavailable."}),
                "gaze_one_euro_min_cutoff": ("FLOAT", {"default": 1.7, "min": 0.05, "max": 10.0, "step": 0.05, "tooltip": "One-euro filter base cutoff frequency (Hz). Lower = more aggressive jitter rejection at the cost of slight lag. 1.7 is a good default for 24-30 fps gaze."}),
                "gaze_one_euro_beta": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": "One-euro filter speed coefficient. Higher = filter relaxes faster on quick saccades, preserving responsiveness; lower = stronger smoothing during fast moves."}),
                "gaze_max_yaw_deg": ("FLOAT", {"default": 30.0, "min": 5.0, "max": 60.0, "step": 1.0, "tooltip": "Saturation yaw angle in degrees that corresponds to blend shape value 1.0. 30\u00b0 covers the comfortable physiological range; raise for more dramatic eye motion."}),
                "gaze_max_pitch_deg": ("FLOAT", {"default": 25.0, "min": 5.0, "max": 60.0, "step": 1.0, "tooltip": "Saturation pitch angle in degrees that corresponds to blend shape value 1.0. 25\u00b0 covers the comfortable physiological range."}),
                # ---- Jitterless face crop (manual frame-0 anchor + keyframes) ----
                "crop_mode": (["default", "expression_lock", "jitterless", "auto", "action"], {"default": "default", "tooltip": "How the face crop box is built. Five modes - the ones that measured well and the ones you asked to keep.\n\ndefault = the REFERENCE behaviour, byte for byte with Wan2.2's own process_pipepline.py: per-frame face-tight box, no smoothing of anything. Measured 1.09px of face wander inside the 512 tile and 100% face-fill. This is the safe choice and the one to A/B against.\n\nexpression_lock = the reference box with the centre taken RAW per frame and only the box SIZE stabilised, so the tile stops breathing. Measured 1.18px wander, 100% fill - statistically the same as default, with a steadier tile size.\n\njitterless = locked constant-size crop, Mocha-style planar hold. Steadiest tile SIZE of all, at the cost of face-fill when the subject moves toward or away from camera.\n\nauto = legacy motion-adaptive smoothing with an optional constant-size box. The most forgiving on very jittery handheld, the least faithful on subtle expression.action = for DANCE / fast body action with a moving camera. Constant-size box on a HOLD-THEN-JUMP path: the crop sits perfectly still through jitter and through a fast limb move, then relocates in one discrete jump once the subject has genuinely travelled. It never tracks continuously, which is what makes it different from jitterless - jitterless one-euro-filters the CENTRE, and on a sustained pan that filter lags the whole way (the same mechanism that got reference_smooth retired at 26-61px of drift). Cost: the box must be LARGER than the face to have room to hold, so face-fill is lower than default by construction - measured 57% vs default 100% on a synthetic dance+pan rig, with 6 jumps over 96 frames instead of 96. Use it when a steady tile matters more than maximum face pixels; use default when the camera is locked off.\n\nRETIRED: central_face and reference_smooth. Both still RUN if a saved workflow selects them, but they are no longer offered - central_face crops eyebrow-to-mouth only, which starves an already small face, and reference_smooth filters the crop CENTRE, which lets the face drift inside the tile (measured 26-61px on a pan) and spends Wan-Animate's 20-number face budget on rigid motion instead of expression.\n\nNOTE: the mode matters far less than the face RESOLUTION. If your face box is under ~160px the tile is mostly invented pixels and no mode fixes that - wire a full-res plate to hires_images."}),
                "frame0_cx": ("INT", {"default": -1, "min": -1, "max": 8192, "tooltip": "Frame 0 anchor center X in pixels. -1 = use detected face center on frame 0. Used only when crop_mode=jitterless."}),
                "frame0_cy": ("INT", {"default": -1, "min": -1, "max": 8192, "tooltip": "Frame 0 anchor center Y in pixels. -1 = use detected face center on frame 0."}),
                "frame0_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 4, "tooltip": "Locked square crop size in pixels (used for the entire clip). 0 = fall back to face_box_size_px."}),
                "keyframes_json": ("STRING", {"default": "[]", "multiline": True, "tooltip": "JSON list of per-frame overrides: [{\"frame\":N, \"cx\":X, \"cy\":Y, \"size\":S?}, ...]. Frames between key-frames are linearly interpolated. size is optional; if omitted the locked size is kept."}),
                "smoothing_method": (["one_euro", "ema", "gaussian", "none"], {"default": "one_euro", "tooltip": "Center-trajectory filter. one_euro = jitterless adaptive low-pass (recommended). ema = legacy motion-adaptive EMA. gaussian = fixed-window 1D blur. none = raw."}),
                "crop_one_euro_min_cutoff": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 10.0, "step": 0.05, "tooltip": "One-euro min cutoff (Hz) for crop center. Lower = stronger jitter rejection."}),
                "crop_one_euro_beta": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "One-euro speed coefficient for crop center. Higher = filter relaxes faster on quick motion."}),
                "crop_gaussian_window": ("INT", {"default": 7, "min": 3, "max": 51, "step": 2, "tooltip": "Window size (odd) for the Gaussian temporal blur of the crop center."}),
                "crop_safety_margin": ("FLOAT", {"default": 1.12, "min": 1.0, "max": 2.0, "step": 0.01, "tooltip": "Inflate the crop by this factor before smoothing so filter lag, yaw-foreshortened detections and expression-driven bbox growth cannot clip the face. 1.0 = no margin (old behaviour). Applies to both 'auto' and 'jitterless'. If crop_containment_check reports corrections on more than a handful of frames, raise this toward 1.15-1.20 rather than fighting it downstream."}),
                "crop_size_one_euro_beta": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "One-euro beta for the crop SIZE trajectory, separate from crop_one_euro_beta (which is the CENTER's). Position wants heavy damping to kill detector jitter; scale wants to follow real zoom/approach or the crop under-sizes mid-move. Only used when the size is allowed to vary (crop_mode='auto', or jitterless with explicit key-frame sizes) — a locked jitterless size ignores it by definition."}),
                "crop_containment_check": ("BOOLEAN", {"default": True, "tooltip": "HARD per-frame guarantee that the actual detected face bbox ends up inside the final crop. After smoothing, any frame whose face escapes the crop is corrected. In 'jitterless' the correction SHIFTS the crop (the exact-size lock is preserved); growing would silently break the lock, so a face genuinely larger than the locked size is reported in the log instead — that means face_box_size_px / crop_safety_margin is too small for the shot. In 'auto' the crop may grow. Correction counts are logged."}),
                "crop_containment_tolerance": ("INT", {"default": 4, "min": 0, "max": 128, "tooltip": "Extra pixels of slack required around the detected face bbox when crop_containment_check tests containment."}),
                "auto_smoothing_method": (["legacy_ema", "one_euro", "ema", "gaussian", "none"], {"default": "legacy_ema", "tooltip": "Which filter crop_mode='auto' uses. 'legacy_ema' keeps auto's original bespoke EMA byte-for-byte (the default, so existing workflows are untouched); the others route auto through the same shared filters jitterless uses, honouring crop_one_euro_* / crop_gaussian_window. Ignored unless crop_mode='auto'."}),
                "preserve_face_aspect": ("BOOLEAN", {"default": False, "tooltip": "OFF (default) = square crop, which is what Kijai's WanVideoWrapper needs: WanVideoAnimateEmbeds re-resizes anything that is not already 512x512 with common_upscale(..., 'center'), and that CENTER-CROPS to square first, so a non-square tile would have its top and bottom cut off. This node always emits 512x512 so the wrapper passes it straight through. ON = build the crop at the per-frame face-box aspect before the 512 resize. EXPERIMENTAL and NOT what the reference does: Wan2.2's process_pipepline.py simply does get_face_bboxes -> frames[y1:y2, x1:x2] -> cv2.resize(512,512) per frame, with no aspect tracking, no locked size and no smoothing. For the reference behaviour exactly, use crop_mode='default'."}),
                "force_eyes_open": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Force closed/blinking eyes open. 0 = off (default). 1 = fully open to a natural EAR of ~0.30; intermediate values blend.\n\nThis REACHES THE MODEL because it pairs with DrawViTPoseV2's apply_pose_edits_to_face warp: Wan-Animate's face conditioning is 100%% pixel-driven (landmarks only place the crop, the LIA motion encoder reads raw crop pixels), so this node writes opened-eye LANDMARKS and DrawViTPoseV2 warps the actual crop PIXELS to match — using each frame's own crop as the source, so identity, head pose, mouth and lighting are preserved and only the eye aperture changes. Wire face_images/face_images_512 -> DrawViTPoseV2.face_images and leave apply_pose_edits_to_face='warp' (the default) or this does nothing visible.\n\nOnly ever opens, never closes; eye corners stay fixed so the warp stays local."}),
                "eye_open_mode": (["blinks_only", "all_frames"], {"default": "blinks_only", "tooltip": "Which frames force_eyes_open targets. 'blinks_only' = only frames whose measured Eye-Aspect-Ratio falls below eye_open_blink_ear (keeps natural performance, removes blinks). 'all_frames' = whole-shot override, for when the subject squints throughout."}),
                "eye_open_blink_ear": ("FLOAT", {"default": 0.18, "min": 0.01, "max": 0.40, "step": 0.01, "tooltip": "Eye-Aspect-Ratio below which a frame counts as a blink for eye_open_mode='blinks_only'. A natural open eye is ~0.28-0.35, a full blink ~0.05-0.15. Raise toward 0.22 to also catch heavy-lidded frames."}),
                # ---- Wan-Animate paper-driven gaze fixes (arXiv:2509.14055) ----
                "eye_align_mode": (["default", "eye_upper_third"], {"default": "default", "tooltip": "Wan-Animate paper recommendation #1: 'eye_upper_third' vertically shifts the face crop so eyes land at the upper third of the 512x512 face encoder input. The encoder reads holistic face appearance, so consistent eye placement directly improves gaze fidelity. 'default' keeps legacy bbox center."}),
                "eye_y_fraction": ("FLOAT", {"default": 0.30, "min": 0.10, "max": 0.60, "step": 0.01, "tooltip": "Target eye row as a fraction of crop height (0.30 = upper third). Only used when eye_align_mode = 'eye_upper_third'."}),
                "face_cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1, "tooltip": "Wan-Animate paper Sec. 4.3 names CFG on the face-conditioning branch as one lever for finer expression control, BUT Kijai's ComfyUI-WanVideoWrapper has no separate face-CFG input to wire this into — wiring it nowhere is a dead passthrough. The wrapper instead exposes a STRONGER, more direct lever for exactly this purpose (spec 2.2: 'a raw face-adapter block-scale... changes contribution before guidance math rather than after'): WanVideoAnimateEmbeds.face_strength (default 1.0, try 1.5-2.5 for stronger expression adherence). Use that widget on your WanVideoAnimateEmbeds node instead. This FLOAT output is kept for any sampler that DOES expose a genuine face-CFG input and for forward-compat; 1.0 = no-op."}),
                # ---- Gaze engine selector + Kalman tuning (appended at end for back-compat with saved workflows) ----
                "gaze_engine": (["l2cs_gaze360", "l2cs_mpiigaze", "ethxgaze", "pose_normalized_resnet50", "iris_geometric", "blendshape_head_corrected", "blendshape_only"], {"default": "l2cs_gaze360", "tooltip": "Per-eye gaze yaw/pitch engine. DEFAULT is now l2cs_gaze360 (GPU/CUDA, auto-downloads ~100MB once) so gaze runs on the GPU; blendshape_* are the CPU-only fallbacks.\n\n* iris_geometric (NEW, deterministic): MEASURES the MediaPipe iris centre inside the eye aperture (corner-to-corner, lid-to-lid) instead of estimating gaze with a NN — no per-person appearance bias, per-eye output, blink-gated, composed with the solvePnP head pose + Kalman like blendshape_head_corrected. Best fidelity for animation retargeting (the character's eyeballs copy the performer's iris positions). Pure CPU math, no downloads.\n* blendshape_head_corrected (DEFAULT, recommended): MediaPipe ARKit blend shapes + solvePnP head pose + Kalman temporal smoother. Eye-in-head rotation is composed with the head rotation so the rendered arrow tracks rotated heads. Pure numpy + cv2, no downloads.\n* blendshape_only: legacy May-2026 shipped behavior; eye-in-head only, no head composition.\n* l2cs_gaze360: L2CS-Net (MIT) ResNet50 trained on Gaze360. ~10.4\u00b0 MAE but robust to extreme poses (recommended for Wan-Animate character scenes). One-time ~100MB weight download to ComfyUI/models/gaze/.\n* l2cs_mpiigaze: L2CS-Net MPIIGaze variant. ~3.9\u00b0 MAE but calibrated only for near-portrait subjects.\n* pose_normalized_resnet50: Highest-accuracy path. Pipeline = solvePnP head pose -> analytical pose-normalized 224x224 face warp (head roll removed, camera distance fixed at 600 mm) -> ResNet50+Linear(2048,2) gaze regressor -> de-rotate output back to camera frame. Major accuracy gain on tilted / off-axis heads. The normalization warp is a clean-room implementation of the 2018 ETRA paper's published equations and ships with this pack (Apache-2.0). The ResNet50 checkpoint is NOT bundled \u2014 place a community-released gaze-trained ResNet50 weight file at <ComfyUI>/models/gaze/pose_normalized_resnet50.pth.tar to enable this engine. Note: those community checkpoints are typically released under CC BY-NC-SA 4.0 (non-commercial); you are responsible for confirming the licence of any weights you install matches your use case. If the file is missing the node automatically falls back to l2cs_gaze360.\n* ethxgaze: ETH-XGaze ResNet-50 (ECCV 2020, ~2.5\u00b0 in-the-wild MAE). Post-processes iris_data using pose-normalised 224x224 face crops + the official gaze_network. Requires (a) the third_party/ETH-XGaze/ repo cloned for face_model.txt + model.py and (b) checkpoint `epoch_24_ckpt.pth.tar` placed in `ComfyUI/models/ethxgaze/`. On any missing prerequisite the engine silently keeps the previous engine's output."}),
                "gaze_kalman_meas_std_deg": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 20.0, "step": 0.1, "tooltip": "Kalman measurement noise (degrees). Higher = trust the model less and lean on the velocity model more — smoother. Used by blendshape_head_corrected and l2cs_* engines."}),
                "gaze_kalman_process_std": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 5.0, "step": 0.05, "tooltip": "Kalman process noise (rad/s). Roughly the expected saccade velocity scale. Higher = filter reacts faster to genuine motion but jitters more."}),
                "gaze_fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 1.0, "tooltip": "Video fps used by the Kalman dt. Set to match your source clip; affects velocity coupling, not absolute angles."}),
                "gaze_calibration_frame": ("INT", {"default": -1, "min": -1, "max": 999999, "tooltip": "W7-G2 per-shot gaze calibration (iris_geometric engine only). Set this to a frame index where the subject looks STRAIGHT AT THE CAMERA; the measured eye-in-head angles on that frame become the zero reference for the whole shot, removing per-person eye-shape bias (the last few degrees of error no model can fix). -1 = off."}),
                # C0.1 — per-frame iris repaint at gaze-corrected position.
                "apply_gaze_to_face_image": (["off", "warp", "overlay", "replace"], {"default": "off", "tooltip": "Move real iris pixels in the face crop to match the computed gaze.\n\nDEFAULT IS OFF, deliberately. This is a Delaunay warp of the actual OUTPUT pixels and it is only as good as the gaze estimate driving it. If the estimate is wrong it DAMAGES face_images rather than helping. Turn it on only after checking the gaze arrows are correct on your footage AND you actually want to override the performer's own eyes.\n\nNote gaze reaches Wan-Animate ONLY through these pixels - the pose conditioning image is a body skeleton with five coarse head dots and no iris. With this off, the gaze in face_images is whatever the camera saw, which is normally what you want."}),
                "au_amplify": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 1.5, "step": 0.01, "tooltip": "Wan-Animate spec 2.3: the face encoder compresses to a small fixed-capacity motion-basis vector, so a genuinely subtle real microexpression can sit near the compression noise floor. This pushes each frame's detected face landmarks a bit FURTHER along the direction they already moved from the neutral reference frame (au_amplify_neutral_frame) — amplifying REAL, DETECTED motion so more of it survives compression; it never synthesizes anything that wasn't already measured. 1.0 = off (default). 1.15-1.3 is the range the paper's own architecture analysis suggests; values are capped at 1.5 since the correction is only a 2D (eye-line roll+scale) head-pose approximation, not a full 3D one — the discrepancy grows with head yaw/pitch, so keep this modest for non-frontal shots. Delivered via the same Delaunay real-pixel warp as 'warp' gaze mode; gated by the same blur/quality check; on any per-frame failure that frame is left unamplified."}),
                "au_amplify_neutral_frame": ("INT", {"default": 0, "min": 0, "max": 999999, "tooltip": "Frame index to use as the NEUTRAL reference for au_amplify — pick a frame where the subject's expression is relaxed/neutral (Wan-Animate spec 2.4: an already-tense or asymmetric reference eats into the same motion-basis budget the target microexpression needs). Ignored when au_amplify=1.0."}),
                "export_expression_coeffs": ("BOOLEAN", {"default": False, "tooltip": "Wan-Animate spec 3.1 (closed-loop critic, foundation): export the 'expression_coeffs_json' output — per-frame ARKit-52 blendshapes measured from this run's iris_data. Off by default (no extra cost when unused). Run this node once on the source driving video and once on the Wan-Animate generated output, then wire the source run's expression_coeffs_json into DrawViTPoseV2.reference_expression_coeffs_json for a per-AU fidelity report."}),
            },
            "optional": {
                "bbox_override": ("BBOX", {"tooltip": "Optional external BBOX for the frame-0 anchor. Highest priority; overrides frame0_cx/cy/size widgets."}),
                "landmark_overrides_json": ("STRING", {"default": "{}", "multiline": True, "tooltip": "Manual body-keypoint corrections from the Pose editor. Shape: {\"<frame>\": {\"<jointIdx>\": [x_px, y_px], ...}, ...} in SOURCE pixels. Written by the viewer's Edit mode — drag a joint to fix a mis-detection and the correction flows through retargeting into pose_data AND the rendered pose images (not just the preview). Leave as {} for pure detection."}),
                "face_resize_filter": (["mitchell", "cubic", "keys", "simon", "rifman", "parzen", "notch", "lanczos4", "lanczos6", "sinc4", "impulse"], {"default": "mitchell", "tooltip": "Resampling filter for the face crop -> 512 resize. Nuke's filter set, same names, so a compositor can reason about it with the vocabulary they already use.\n\nMEASURED on the real 46px-face case, resizing 46 -> 512 -> 64 and comparing against 46 -> 64 direct (lower error = less information mangled, overshoot = ringing):\n  cubic     err 0.00297   overshoot 0.0000\n  mitchell  err 0.00311   overshoot 0.0000   <- DEFAULT\n  keys      err 0.00329   overshoot 0.0000\n  lanczos4  err 0.00715   overshoot 0.0001\n  rifman    err 0.01007   overshoot 0.0312\n  sinc4     err 0.02274   overshoot 0.0804\nThe previous hardcoded cv2 LANCZOS4 measured 0.00721 - about 2.4x worse than mitchell.\n\nWHY: a filter rings when its kernel goes NEGATIVE, overshooting at hard edges. That overshoot is invented structure the 20-number face encoder cannot tell from real texture, and it is why a crop taken to 512 and sampled back down does not match the original sampled down directly - the ringing does not cancel.\n\ncubic / parzen / notch / impulse cannot ring (no negative lobes). mitchell has small negative lobes and measured zero overshoot here - Mitchell and Netravali's own paper concludes B=C=1/3 is the best blur-versus-ringing compromise, which is why it is the default. keys / simon / rifman / lanczos / sinc are progressively sharper and ring progressively harder.\n\nDOWNSIZING: prefer mitchell, or parzen if the plate is noisy - ringing survives into the latent as hard fringes, and softness does not."}),
                "face_sr": (["none", "lanczos", "comfy_upscale"], {"default": "none", "tooltip": "Super-resolve the face crop BEFORE it is resized to the 512x512 the encoder needs.\n\nWhy it exists: the tile is always 512, but the region it is cut from is whatever the plate gives. A 46px face box means an ELEVEN-times upscale, so almost everything the model reads is invented by a resize filter. HeadsUp! (arXiv:2510.09924) discards training faces under 64px interocular (~160px face box) as too small to learn from - the node logs where your shot sits.\n\nnone = plain resize (previous behaviour).\nlanczos = deterministic baseline. Invents nothing, so if a real SR model does not beat this on your footage, that model is only adding hallucination and cost.\ncomfy_upscale = any ESRGAN-family model in ComfyUI/models/upscale_models, named in face_sr_model.\n\nSR runs on the NATIVE crop, before the 512 resize. Order matters: Lanczos is a windowed sinc with negative lobes, so it rings on hard edges; running SR after that would just sharpen the ringing. It is also why a crop taken to 512 and sampled back down does not match the original sampled down directly - the ringing does not cancel.\n\nPREFER hires_images if you have a full-res plate. Real pixels always beat invented ones."}),
                "face_sr_model": ("STRING", {"default": "", "tooltip": "Filename of the upscale model in ComfyUI/models/upscale_models, used only when face_sr=comfy_upscale. Errors naming what is installed if it cannot be found."}),
                "face_sr_stabilise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "How hard to stabilise SR detail over time. Per-frame SR hallucinates independently, so texture BOILS - and boiling is exactly the high-frequency per-frame noise Wan-Animate's 20-number motion code cannot tell from real motion. Hallo2 (arXiv:2410.07718) finds SR only helps expression fidelity when paired with temporal alignment.\n\nOnly the DETAIL layer is filtered; the base is left alone so real motion is never smeared.\n\nMeasured on a moving subject: 0.5 -> flicker -37% with 101% of real motion preserved; 0.7 -> -49% / 87%; 1.0 -> -58% / 75%. 0.5 is the last value that costs nothing in motion. Raise it only if a shot still boils."}),
                "hires_images": ("IMAGE", {"tooltip": "OPTIONAL, and the single biggest quality lever in this node.\n\nWire your FULL-RESOLUTION plate here (the EXR/source sequence). Detection still runs at the working resolution because that is what ViTPose wants and it is fast, but the face TILE is cut from this hi-res source instead.\n\nWhy it matters more than any crop_mode: on an 832x480 plate a face box measures about 46px, and that tile is upscaled ELEVEN times to reach the 512x512 the encoder needs - 99.2% of what the model reads is invented, and an eyeball at that scale is roughly 4 pixels across. There is no iris in 4 pixels. Every crop_mode was cropping the same starved image, which is why none of them fixed eye direction or micro-expression detail.\n\nSame framing, real pixels: 46px becomes 106px from a 1080p plate, 212px from 4K.\n\nIgnored (with a warning) if it is not larger than the working plate."}),
                "retarget_image": ("IMAGE", {"tooltip": "Optional reference image of the TARGET character. When connected, the detected driver pose is RETARGETED onto this reference's body proportions and position (the same retarget V1 had): the reference's pose is detected, then get_retarget_pose maps the driver's motion onto it. Leave unconnected for straight detection (no retarget)."}),
                "use_flux": ("BOOLEAN", {"default": False, "tooltip": "Enhanced retargeting via FLUX.1-Kontext-dev (Wan 2.2 Animate's third retarget mode). When ON with retarget_image connected, FLUX normalizes the reference AND the first template frame to a standard front-facing pose BEFORE retargeting, so retargeting starts from a neutral instead of carrying a 3/4-profile or head-tilt into the output. Recommended ONLY when the reference character is NOT front-facing; for a front-facing reference, basic retarget (use_flux off) is enough. Needs the FLUX.1-Kontext-dev model — set flux_kontext_path. The authors' caveat: FLUX.1-Kontext-dev has limited capability, consistency is not guaranteed; check the intermediate edited frames."}),
                "flux_kontext_path": ("STRING", {"default": "", "tooltip": "Path to the FLUX.1-Kontext-dev diffusers checkpoint FOLDER. Download from https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev. Empty = look the model up via ComfyUI's folder_paths ('flux' / 'checkpoints' / 'unet' keys, in that order). Raises a clear error naming what was searched if nothing is found — never silently disables use_flux."}),
            },
        }

    RETURN_TYPES = ("POSEDATA", "IMAGE", "STRING", "BBOX", "BBOX", "STRING", "IMAGE", "STRING", "STRING", "FLOAT", "FACE_RESTORE_INFO", "FLOAT", "IMAGE", "STRING")
    RETURN_NAMES = ("pose_data", "face_images", "key_frame_body_points", "bboxes", "face_bboxes", "iris_data", "debug_image", "right_pupil_xy", "left_pupil_xy", "lip_openness_ratio", "restore_info", "face_cfg_scale", "face_images_512", "expression_coeffs_json")
    OUTPUT_TOOLTIPS = (
        "Per-frame pose+face+iris dict bundle. Feed into Draw ViT Pose (V2).",
        "Cropped face IMAGE batch suitable for face-id encoders.",
        "Key-frame body points as JSON string (debug).",
        "Per-frame body BBOX list.",
        "Per-frame face BBOX list.",
        "Iris/gaze JSON dump (debug).",
        "Annotated debug IMAGE batch (skeleton overlay).",
        "Right pupil pixel xy as JSON (per frame).",
        "Left pupil pixel xy as JSON (per frame).",
        "Mouth-open scalar list (0=closed, 1=wide).",
        "Per-frame {x1,y1,x2,y2,size,frame_shape} dict for paste-back nodes.",
        "CFG scale for the face conditioning input. Wire into the Wan-Animate sampler's face CFG. 1.0 = CFG off (paper default).",
        "Cropped face IMAGE batch force-resized to 512x512 (bilinear). Pre-shaped for the Wan 2.2 Animate face encoder; wire directly without an extra Resize node.",
        "Wan-Animate spec 3.1: per-frame ARKit-52 blendshapes measured from this run's iris_data, as {fps,names,frames:[{frame,blendshapes}]}. Run this node on BOTH the source driving video and the Wan-Animate GENERATED output, then wire this output from the SOURCE run into DrawViTPoseV2.reference_expression_coeffs_json (with the GENERATED run's pose_data wired into DrawViTPoseV2.pose_data as usual) to get a per-AU fidelity report. Empty '{}' when export_expression_coeffs=False (default) or no MediaPipe blendshapes were captured this run.",
    )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess_V2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def process(
        self,
        model,
        images,
        width,
        height,
        detection_threshold,
        pose_threshold,
        use_clahe,
        use_blur_for_pose,
        blur_radius,
        blur_sigma,
        use_face_smoothing,
        face_smoothing_strength,
        use_constant_face_box,
        face_box_size_px,
        use_iris_smoothing,
        iris_smoothing_strength,
        iris_smoothing_method="one_euro",
        iris_one_euro_min_cutoff=1.0,
        iris_one_euro_beta=0.05,
        gaze_lock_eyes=True,
        gaze_lock_strength=0.7,
        use_mediapipe_face=True,
        use_blendshape_gaze=True,
        gaze_engine="l2cs_gaze360",
        gaze_kalman_meas_std_deg=3.0,
        gaze_kalman_process_std=0.8,
        gaze_fps=30.0,
        gaze_calibration_frame=-1,
        gaze_one_euro_min_cutoff=1.7,
        gaze_one_euro_beta=0.3,
        gaze_max_yaw_deg=30.0,
        gaze_max_pitch_deg=25.0,
        crop_mode="default",
        frame0_cx=-1,
        frame0_cy=-1,
        frame0_size=0,
        keyframes_json="[]",
        smoothing_method="one_euro",
        crop_one_euro_min_cutoff=1.0,
        crop_one_euro_beta=0.05,
        crop_gaussian_window=7,
        eye_align_mode="default",
        eye_y_fraction=0.30,
        face_cfg_scale=1.0,
        apply_gaze_to_face_image="off",
        au_amplify=1.0,
        au_amplify_neutral_frame=0,
        export_expression_coeffs=False,
        bbox_override=None,
        landmark_overrides_json="{}",
        retarget_image=None,
        # Appended at the END (never mid-signature): the inner call site
        # passes POSITIONALLY, so inserting a param anywhere above would
        # silently shift every later argument.
        crop_safety_margin=1.12,
        crop_size_one_euro_beta=0.20,
        crop_containment_check=True,
        crop_containment_tolerance=4,
        auto_smoothing_method="legacy_ema",
        force_eyes_open=0.0,
        eye_open_mode="blinks_only",
        eye_open_blink_ear=0.18,
        preserve_face_aspect=True,
        face_crop_scale=1.3,
        clahe_clip_limit=2.0,
        clahe_grid_size=8,
        detect_gamma=1.0,
        detect_white_balance=False,
        detect_denoise=0.0,
        detect_sharpen=0.0,
        detect_saturation=1.0,
        use_flux=False,
        flux_kontext_path="",
        # Appended at the END for the same reason as _process_impl: this is
        # an optional IMAGE input, and ComfyUI passes optional inputs by
        # KEYWORD, so position is free here — but keeping both signatures in
        # the same order makes the positional hand-off below verifiable.
        hires_images=None,
        face_sr="none",
        face_sr_model="",
        face_sr_stabilise=0.5,
        face_resize_filter="mitchell",
    ):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"PoseAndFaceDetectionV2: expected IMAGE (B,H,W,3); got {tuple(getattr(images, 'shape', ()))}"
            )
        with torch.inference_mode():
            return self._process_impl(
                model,
                images,
                width,
                height,
                detection_threshold,
                pose_threshold,
                use_clahe,
                use_blur_for_pose,
                blur_radius,
                blur_sigma,
                use_face_smoothing,
                face_smoothing_strength,
                use_constant_face_box,
                face_box_size_px,
                use_iris_smoothing,
                iris_smoothing_strength,
                iris_smoothing_method,
                iris_one_euro_min_cutoff,
                iris_one_euro_beta,
                gaze_lock_eyes,
                gaze_lock_strength,
                use_mediapipe_face,
                use_blendshape_gaze,
                gaze_engine,
                gaze_kalman_meas_std_deg,
                gaze_kalman_process_std,
                gaze_fps,
                gaze_calibration_frame,
                gaze_one_euro_min_cutoff,
                gaze_one_euro_beta,
                gaze_max_yaw_deg,
                gaze_max_pitch_deg,
                crop_mode,
                frame0_cx,
                frame0_cy,
                frame0_size,
                keyframes_json,
                smoothing_method,
                crop_one_euro_min_cutoff,
                crop_one_euro_beta,
                crop_gaussian_window,
                eye_align_mode,
                eye_y_fraction,
                face_cfg_scale,
                apply_gaze_to_face_image,
                au_amplify,
                au_amplify_neutral_frame,
                export_expression_coeffs,
                bbox_override,
                landmark_overrides_json,
                retarget_image,
                crop_safety_margin,
                crop_size_one_euro_beta,
                crop_containment_check,
                crop_containment_tolerance,
                auto_smoothing_method,
                force_eyes_open,
                eye_open_mode,
                eye_open_blink_ear,
                preserve_face_aspect,
                face_crop_scale,
                clahe_clip_limit,
                clahe_grid_size,
                detect_gamma,
                detect_white_balance,
                detect_denoise,
                detect_sharpen,
                detect_saturation,
                hires_images=hires_images,
                face_resize_filter=face_resize_filter,
                face_sr=face_sr,
                face_sr_model=face_sr_model,
                face_sr_stabilise=face_sr_stabilise,
            )

    def _process_impl(
        self,
        model,
        images,
        width,
        height,
        detection_threshold,
        pose_threshold,
        use_clahe,
        use_blur_for_pose,
        blur_radius,
        blur_sigma,
        use_face_smoothing,
        face_smoothing_strength,
        use_constant_face_box,
        face_box_size_px,
        use_iris_smoothing,
        iris_smoothing_strength,
        iris_smoothing_method="one_euro",
        iris_one_euro_min_cutoff=1.0,
        iris_one_euro_beta=0.05,
        gaze_lock_eyes=True,
        gaze_lock_strength=0.7,
        use_mediapipe_face=True,
        use_blendshape_gaze=True,
        gaze_engine="l2cs_gaze360",
        gaze_kalman_meas_std_deg=3.0,
        gaze_kalman_process_std=0.8,
        gaze_fps=30.0,
        gaze_calibration_frame=-1,
        gaze_one_euro_min_cutoff=1.7,
        gaze_one_euro_beta=0.3,
        gaze_max_yaw_deg=30.0,
        gaze_max_pitch_deg=25.0,
        crop_mode="default",
        frame0_cx=-1,
        frame0_cy=-1,
        frame0_size=0,
        keyframes_json="[]",
        smoothing_method="one_euro",
        crop_one_euro_min_cutoff=1.0,
        crop_one_euro_beta=0.05,
        crop_gaussian_window=7,
        eye_align_mode="default",
        eye_y_fraction=0.30,
        face_cfg_scale=1.0,
        apply_gaze_to_face_image="off",
        au_amplify=1.0,
        au_amplify_neutral_frame=0,
        export_expression_coeffs=False,
        bbox_override=None,
        landmark_overrides_json="{}",
        retarget_image=None,
        # Appended at the END (never mid-signature): the inner call site
        # passes POSITIONALLY, so inserting a param anywhere above would
        # silently shift every later argument.
        crop_safety_margin=1.12,
        crop_size_one_euro_beta=0.20,
        crop_containment_check=True,
        crop_containment_tolerance=4,
        auto_smoothing_method="legacy_ema",
        force_eyes_open=0.0,
        eye_open_mode="blinks_only",
        eye_open_blink_ear=0.18,
        preserve_face_aspect=True,
        face_crop_scale=1.3,
        clahe_clip_limit=2.0,
        clahe_grid_size=8,
        detect_gamma=1.0,
        detect_white_balance=False,
        detect_denoise=0.0,
        detect_sharpen=0.0,
        detect_saturation=1.0,
        use_flux=False,
        flux_kontext_path="",
        # APPENDED AT THE END, never mid-signature: process() calls this
        # POSITIONALLY, so a parameter inserted anywhere above shifts every
        # later argument. Verified by an AST check that each positional arg
        # lands on the identically-named parameter.
        hires_images=None,
        face_sr="none",
        face_sr_model="",
        face_sr_stabilise=0.5,
        face_resize_filter="mitchell",
    ):
        detector = model["yolo"]
        pose_model = model["vitpose"]

        if hasattr(detector, "threshold_conf"):
            detector.threshold_conf = detection_threshold

        B, H, W, C = images.shape
        shape = np.array([H, W])[None]
        images_np = images.detach().cpu().numpy() if hasattr(images, "detach") else images.cpu().numpy()

        # --- Prepare blurred version for detection & pose ---
        if use_blur_for_pose:
            ksize = int(blur_radius) * 2 + 1
            images_blurred = np.stack([
                cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)
                for img in images_np
            ])
        else:
            images_blurred = images_np

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution = (256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()

        # --- Optional retarget reference (V1 parity) ---
        # When a retarget_image is connected, detect ITS pose so the driver's
        # motion can be mapped onto the target character's proportions/position
        # (get_retarget_pose below). Runs here while both detector + pose_model
        # are alive; best-effort — a failed reference just disables retarget.
        refer_pose_meta = None
        refer_img_proc = None
        if retarget_image is not None:
            try:
                _rt = retarget_image[0]
                _rt_np = _rt.detach().cpu().numpy() if hasattr(_rt, "detach") else np.asarray(_rt)
                _rt_np = np.ascontiguousarray(_rt_np[..., :3]).astype(np.float32)
                _ref_shape = np.array([_rt_np.shape[0], _rt_np.shape[1]])[None]
                _ref_dets = detector(
                    cv2.resize(_rt_np, (640, 640)).transpose(2, 0, 1)[None], _ref_shape
                )[0]
                if isinstance(_ref_dets, list) and len(_ref_dets) > 0 and isinstance(_ref_dets[0], dict):
                    _ref_bbox = _ref_dets[0]["bbox"]
                else:
                    _ref_bbox = None
                if (_ref_bbox is None or len(_ref_bbox) < 5 or _ref_bbox[4] <= 0
                        or (_ref_bbox[2] - _ref_bbox[0]) < 10 or (_ref_bbox[3] - _ref_bbox[1]) < 10):
                    _ref_bbox = np.array([0, 0, _rt_np.shape[1], _rt_np.shape[0], 1.0], dtype=np.float32)
                _rc, _rs = bbox_from_detector(_ref_bbox, input_resolution, rescale=rescale)
                _ref_crop = crop(_rt_np, _rc, _rs, (input_resolution[0], input_resolution[1]))[0]
                _ref_crop = preprocess_for_pose(_ref_crop, use_clahe,
                                       clahe_clip=clahe_clip_limit, clahe_grid=clahe_grid_size,
                                       gamma=detect_gamma, white_balance=detect_white_balance,
                                       denoise=detect_denoise, sharpen=detect_sharpen,
                                       saturation=detect_saturation)
                _ref_norm = ((_ref_crop - IMG_NORM_MEAN) / IMG_NORM_STD).transpose(2, 0, 1).astype(np.float32)
                _ref_kp = pose_model(_ref_norm[None], np.array(_rc)[None], np.array(_rs)[None])
                refer_pose_meta = load_pose_metas_from_kp2ds_seq(
                    _ref_kp, width=_rt_np.shape[1], height=_rt_np.shape[0]
                )[0]
                refer_img_proc = _rt_np
                logging.getLogger(__name__).info(
                    "PoseAndFaceDetectionV2: retarget reference detected (%dx%d).",
                    _rt_np.shape[1], _rt_np.shape[0],
                )
            except Exception as _rt_exc:  # noqa: BLE001 — a bad reference just disables retarget
                logging.getLogger(__name__).warning(
                    "PoseAndFaceDetectionV2: retarget_image detection failed (%s); ignoring.", _rt_exc,
                )
                refer_pose_meta = None
                refer_img_proc = None

        comfy_pbar = ProgressBar(B * 2)
        progress = 0
        bboxes = []

        # --- YOLO detection (on blurred) ---
        for img in _IC.track(
            images_blurred, B, "WanAnimateV2: YOLO bbox detect",
        ):
            detections = detector(cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None], shape)[0]
            # IDENTITY TRACKING (fixed 2026-08-01). This used to take
            # detections[0] unconditionally. YOLO does not guarantee a stable
            # order between frames, so with more than one person — or even one
            # person plus a false positive that flickers in and out — the crop
            # could swap subject mid-clip, taking all 133 keypoints with it.
            # Prefer the detection closest to the LAST ACCEPTED box instead, so
            # the same person is followed once chosen.
            if (isinstance(detections, list) and len(detections) > 1
                    and isinstance(detections[0], dict) and bboxes):
                _prev = next((b for b in reversed(bboxes) if b is not None), None)
                if _prev is not None:
                    _pcx = 0.5 * (float(_prev[0]) + float(_prev[2]))
                    _pcy = 0.5 * (float(_prev[1]) + float(_prev[3]))

                    def _near(_d):
                        _b = _d.get("bbox")
                        if _b is None or len(_b) < 4:
                            return 1e9
                        return math.hypot(0.5 * (float(_b[0]) + float(_b[2])) - _pcx,
                                          0.5 * (float(_b[1]) + float(_b[3])) - _pcy)

                    detections = sorted(detections, key=_near)
            if isinstance(detections, list) and len(detections) > 0 and isinstance(detections[0], dict):
                bboxes.append(detections[0]["bbox"])
            else:
                bboxes.append(None)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        detector.cleanup()

        # --- Stabilise the person box before it drives the pose crop ---------
        # (fixed 2026-08-01) This box is re-detected from scratch on every
        # frame and it defines the crop ViTPose sees. ViTPose predicts in CROP
        # space and the result is mapped back, so a few pixels of box jitter
        # moves ALL 133 keypoints together, every frame — the "all points are
        # jumpy as hell" report. The subject's actual motion is low-frequency;
        # the detector's frame-to-frame disagreement is not.
        #
        # Smoothed as centre + size rather than as four independent edges:
        # filtering x1/x2/y1/y2 separately lets the box breathe asymmetrically,
        # which changes the crop's aspect and shears the pose. Zero-phase
        # (forward+backward) so a real move is not delayed — this is offline
        # work, there is no reason to accept a causal filter's lag.
        _bb_idx = [i for i, b in enumerate(bboxes) if b is not None and len(b) >= 4]
        if len(_bb_idx) > 2:
            _cx = np.array([0.5 * (float(bboxes[i][0]) + float(bboxes[i][2])) for i in _bb_idx], np.float32)
            _cy = np.array([0.5 * (float(bboxes[i][1]) + float(bboxes[i][3])) for i in _bb_idx], np.float32)
            _bw = np.array([abs(float(bboxes[i][2]) - float(bboxes[i][0])) for i in _bb_idx], np.float32)
            _bh = np.array([abs(float(bboxes[i][3]) - float(bboxes[i][1])) for i in _bb_idx], np.float32)
            _raw_jit = float(np.abs(np.diff(_cx)).mean() + np.abs(np.diff(_cy)).mean()) if len(_cx) > 1 else 0.0
            _kw = dict(method="one_euro", one_euro_min_cutoff=0.9,
                       one_euro_beta=0.03, gaussian_window=7)
            _sx = _smooth_1d(_cx, scale_norm=max(float(W), 1.0), **_kw)
            _sy = _smooth_1d(_cy, scale_norm=max(float(H), 1.0), **_kw)
            _sw = _smooth_1d(_bw, scale_norm=max(float(_bw.mean()), 1.0), **_kw)
            _sh = _smooth_1d(_bh, scale_norm=max(float(_bh.mean()), 1.0), **_kw)
            for _k, _i in enumerate(_bb_idx):
                _o = list(bboxes[_i])
                _o[0] = float(_sx[_k] - _sw[_k] * 0.5)
                _o[1] = float(_sy[_k] - _sh[_k] * 0.5)
                _o[2] = float(_sx[_k] + _sw[_k] * 0.5)
                _o[3] = float(_sy[_k] + _sh[_k] * 0.5)
                bboxes[_i] = _o
            _sm_jit = (float(np.abs(np.diff(_sx)).mean() + np.abs(np.diff(_sy)).mean())
                       if len(_sx) > 1 else 0.0)
            logging.getLogger(__name__).info(
                "PoseAndFaceDetectionV2: person-box jitter %.2f -> %.2f px/frame "
                "(%.0f%% removed) before it drives the pose crop; %d/%d frames had "
                "a detection.", _raw_jit, _sm_jit,
                100.0 * (1.0 - _sm_jit / max(_raw_jit, 1e-6)), len(_bb_idx), B,
            )

        # --- Pose detection (on blurred) ---
        kp2ds = []
        for img, bbox in _IC.track(
            zip(images_blurred, bboxes), B,
            "WanAnimateV2: pose keypoint extract",
        ):
            if (
                bbox is None
                or len(bbox) < 5
                or bbox[4] <= 0
                or (bbox[2] - bbox[0]) < 10
                or (bbox[3] - bbox[1]) < 10
            ):
                bbox_use = np.array([0, 0, img.shape[1], img.shape[0], 1.0], dtype=np.float32)
            else:
                bbox_use = bbox

            center, scale = bbox_from_detector(bbox_use, input_resolution, rescale=rescale)
            img_crop = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_crop = preprocess_for_pose(img_crop, use_clahe,
                                       clahe_clip=clahe_clip_limit, clahe_grid=clahe_grid_size,
                                       gamma=detect_gamma, white_balance=detect_white_balance,
                                       denoise=detect_denoise, sharpen=detect_sharpen,
                                       saturation=detect_saturation)
            img_norm = (img_crop - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            kp2ds.append(keypoints)

            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_model.cleanup()
        kp2ds = np.concatenate(kp2ds, 0)

        # --- Confidence threshold for keypoints ---
        if pose_threshold > 0.0:
            kp2ds[..., 2] = np.where(kp2ds[..., 2] < pose_threshold, 0, kp2ds[..., 2])

        pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)

        # --- Raw face bboxes (from blurred pose keypoints; values are in pixel space) ---
        raw_face_bboxes = []
        # Track detection failure EXPLICITLY instead of recognising it later by
        # comparing the box against a sentinel tuple.
        #
        # BUG THIS FIXES (2026-07-24) — the cause of "the face is off-centre in
        # every frame". The old code compared each box against
        #     _missing_bbox = (0, 0, min(W,128), min(H,128))
        # but the fallback actually appends (x1, x2, y1, y2) =
        #     (0, min(W,128), 0, min(H,128))
        # — a different tuple ORDER, so the comparison NEVER matched. Two
        # consequences, both bad:
        #   1. the hold-last-known path for missing detections never fired, and
        #   2. every failed frame was treated as a genuine face box whose centre
        #      is (64, 64) — the TOP-LEFT CORNER of the frame.
        # In the smoothed modes that corner position is fed straight into the
        # centre trajectory, so the filter drags the crop toward the top-left
        # and, because it is a temporal filter, contaminates the neighbouring
        # frames too. A handful of failed detections is enough to pull the crop
        # off the face for a long run of frames.
        # Retired crop modes (central_face / jitterless / auto /
        # reference_smooth) were slicing the SAME detected face. central_face
        # additionally SHRANK it to eyebrow-mouth, which on a wide 832x480
        # plate produced a 46x46 tile (11x upsample, ~4px eyeballs). They
        # all now use the full 68-point box the encoder was trained on.
        # Saved graphs keep loading; the name is remapped once here.
        _retired_crop = {
            "central_face", "auto", "jitterless", "reference_smooth",
        }
        if str(crop_mode) in _retired_crop:
            logging.getLogger(__name__).info(
                "PoseAndFaceDetectionV2: crop_mode=%r now uses expression_lock "
                "(full 68-point face-tight crop, raw centre, size held). "
                "The extra modes were cropping the same face; central_face "
                "was halving it.",
                crop_mode,
            )
            crop_mode = "expression_lock"

        raw_face_missing = []
        raw_face_kf = []
        for meta in pose_metas:
            # Neutralise UNCONFIDENT landmarks before the box is measured
            # (fixed 2026-08-01). pose_threshold zeroes the CONFIDENCE column
            # (kp2ds[..., 2]) but leaves the COORDINATE untouched, so a
            # landmark the detector had no idea about keeps whatever wild
            # position ViTPose emitted. get_face_bboxes then takes min/max over
            # ALL 68 face points with no regard for confidence, so a single
            # stray point drags the entire face box off the face — the
            # "points going off-scale" and the bad crop that follows it.
            # Replacing the doubtful points with the MEDIAN of the confident
            # ones keeps the array shape (get_face_bboxes slices [1:], so the
            # count must not change) while making them invisible to min/max.
            _kf = np.asarray(meta['keypoints_face'], dtype=np.float32).copy()
            if _kf.shape[1] > 2:
                _good = _kf[:, 2] >= _FACE_KP_MIN_CONF
                # slot 0 is a body anchor, not a face point; never trust it here
                _good[0] = False
                if _good.sum() >= 3:
                    _med = np.median(_kf[_good, :2], axis=0)
                    _kf[~_good, :2] = _med
                    _kf[0, :2] = _med
            raw_face_kf.append(_kf[:, :2].copy())
            bbox_face = get_face_bboxes(
                _kf[:, :2], scale=float(face_crop_scale), image_shape=(H, W)
            )
            # Ensure ints and within bounds
            x1, x2, y1, y2 = map(int, bbox_face)
            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H, y2))
            # Fallback if invalid
            _missing = (x2 <= x1 or y2 <= y1)
            if _missing:
                # Centre the placeholder on the FRAME, not the top-left corner.
                # This box is a "we found nothing" marker; parking it at (64,64)
                # meant that if anything downstream ever consumed it as a real
                # box (which is exactly what happened), the crop jumped to the
                # corner. A frame-centred placeholder degrades gracefully.
                _side = min(W, H, 128)
                x1 = max(0, (W - _side) // 2)
                y1 = max(0, (H - _side) // 2)
                x2, y2 = x1 + _side, y1 + _side
            raw_face_missing.append(bool(_missing))
            raw_face_bboxes.append((x1, x2, y1, y2))

        # Replace failed-detection boxes with the nearest SUCCESSFUL neighbour
        # before anything reads them. A placeholder box is not a measurement:
        # feeding it to the temporal filters injects a fake face position that
        # the filter then smears across the surrounding frames, dragging the
        # crop off the real face for far longer than the dropout itself. Nearest
        # -neighbour fill keeps the trajectory continuous and truthful.
        if any(raw_face_missing) and not all(raw_face_missing):
            _good = [i for i, m in enumerate(raw_face_missing) if not m]
            for _i, _m in enumerate(raw_face_missing):
                if not _m:
                    continue
                _src = min(_good, key=lambda g: abs(g - _i))
                raw_face_bboxes[_i] = raw_face_bboxes[_src]
            logging.getLogger(__name__).info(
                "PoseAndFaceDetectionV2: %d/%d frames had no usable face box; "
                "filled from the nearest detected frame so they cannot drag the "
                "smoothed crop toward a placeholder position.",
                sum(raw_face_missing), len(raw_face_missing),
            )
        elif all(raw_face_missing) and raw_face_missing:
            logging.getLogger(__name__).warning(
                "PoseAndFaceDetectionV2: NO face detected on any of %d frames — "
                "face_images will be a frame-centred placeholder crop. Lower "
                "detection_threshold/pose_threshold, or check that the subject's "
                "face is actually visible.", len(raw_face_missing),
            )

        # --- Expression-invariant box anchoring (expression_lock only) -----
        # get_face_bboxes measures min/max over ALL 68 face landmarks, and the
        # brows, lids and lips are IN that set. So raising the brows lifts the
        # box top and dropping the jaw lowers its bottom: the crop MOVES AND
        # RESCALES with the expression it is supposed to be transmitting.
        #
        # Measured on a 12-frame clip with the head held PERFECTLY still and
        # only the expression changing (222x235px box, scale=1.3):
        #     centre wobble 1.89px -> 0.00px
        #     size   wobble 5.85px -> 0.00px   (14px peak-to-peak, a 6% zoom)
        #
        # That wobble is the worst possible noise for this pipeline. Wan-Animate
        # compresses the tile to TWENTY numbers, and global scale/translation is
        # the highest-energy thing a motion encoder sees — so the leak does not
        # merely add noise, it SPENDS the budget that should have carried the
        # expression, and being perfectly correlated with that expression, no
        # temporal filter downstream can separate them.
        #
        # Known failure mode, not a theory: arXiv:2203.14512 reports that
        # "dynamic facial landmark coordinates ... generate jitters and
        # rescaling in face alignment" and that the standard mitigation is
        # cropping "excluding the eyes and mouth coordinates".
        #
        # default is left BYTE-EXACT with the reference on purpose; this runs
        # only for expression_lock, the mode whose entire job is expression.
        _rbx_mode = str(crop_mode).strip().lower()
        if _rbx_mode in ("expression_lock", "central_face") and raw_face_bboxes:
            try:
                from .nodes_extras import _rigid_box as _RBX
                _new_boxes, _rb_stats = _RBX.rigid_anchor_boxes(
                    raw_face_kf, raw_face_bboxes, (H, W))
                if _rb_stats is not None:
                    raw_face_bboxes = _new_boxes
                    logging.getLogger(__name__).info(
                        "PoseAndFaceDetectionV2 [expression_lock]: face box "
                        "re-anchored to expression-invariant landmarks on %d/%d "
                        "frames (mean shift %.2fpx, max %.2fpx). Blinks and mouth "
                        "movement no longer move the crop; the clip's median "
                        "framing is preserved so the tile stays in-distribution.",
                        _rb_stats[2], len(raw_face_bboxes), _rb_stats[0], _rb_stats[1])
            except Exception as _rbx_exc:  # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "PoseAndFaceDetectionV2: rigid box anchoring unavailable "
                    "(%s); using the raw per-frame boxes.", _rbx_exc)

        # --- Convert to centers and raw face sizes (for smoothing) ---
        raw_centers = []
        raw_face_sizes = []
        raw_face_aspects = []
        for (x1, x2, y1, y2) in raw_face_bboxes:
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            raw_centers.append(np.array([cx, cy], dtype=np.float32))
            raw_face_sizes.append(float(max(x2 - x1, y2 - y1)))
            # Wan-Animate's own crop is non-square (area-scaled + 3x upward
            # bias); keep the paper's aspect so the tile we hand the encoder
            # is framed the way it was trained. Width drives the size, aspect
            # restores the height.
            _bw, _bh = float(x2 - x1), float(y2 - y1)
            raw_face_aspects.append(_bh / max(_bw, 1.0))

        crop_mode_str = str(crop_mode)
        # expression_lock shares reference_smooth's code path but keeps the
        # RAW per-frame centre. See the branch below for why that is the whole
        # ballgame for micro-expressions.
        # central_face (HunyuanPortrait) shares expression_lock's code path
        # too; only the RAW box source differs (eyebrow..mouth-bottom
        # instead of the full 68-point face box).
        expression_lock = crop_mode_str in ("expression_lock", "central_face")
        reference_smooth = crop_mode_str == "reference_smooth"
        jitterless = crop_mode_str == "jitterless"
        crop_off = crop_mode_str == "default"
        action_mode = crop_mode_str == "action"

        if crop_off:
            # ── default: raw detected bboxes, no smoothing, no constant-size ──
            face_bboxes = list(raw_face_bboxes)
        elif action_mode:
            # ── action: constant-size box on a hold-then-jump path ─────────
            # Planner is pure numpy (action_planner.py) so its guarantees are
            # asserted in tests/test_action_planner.py without ComfyUI. It
            # RAISES on an infeasible lock rather than degrading to per-frame
            # boxes, which would silently reintroduce the breathing this mode
            # exists to remove.
            from .action_planner import ActionPlanError, plan_action_boxes

            _det_flags = [
                (float(b[2]) - float(b[0])) > 1.0 and (float(b[3]) - float(b[1])) > 1.0
                for b in raw_face_bboxes
            ]
            try:
                _plan = plan_action_boxes(
                    [[float(v) for v in b[:4]] for b in raw_face_bboxes],
                    image_w=W,
                    image_h=H,
                    size=(float(frame0_size) if int(frame0_size) > 0 else None),
                    safety_margin=float(crop_safety_margin),
                    deadband_px=float(crop_containment_tolerance),
                    detected=_det_flags,
                )
            except ActionPlanError as exc:
                raise RuntimeError(
                    "PoseAndFaceDetectionV2: " + str(exc)
                ) from exc
            log.info("[WanAnimateV2] %s", _plan.report)
            face_bboxes = [
                (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                for b in _plan.boxes
            ]
        elif reference_smooth or expression_lock:
            # ── reference_smooth ─────────────────────────────────────────
            # The reference framing, de-jittered. Nothing about the GEOMETRY
            # is invented here: the box is still exactly what get_face_bboxes
            # produced (area-scaled, 3x upward bias, natural non-square
            # aspect) and it is still stretched to 512x512 by resize_face_crop
            # — identical to Wan2.2's process_pipepline.py. The ONLY change is
            # that the four box PARAMETERS (cx, cy, w, h) are run through the
            # same temporal filter the other modes use, instead of being
            # recomputed raw from jittering landmarks every frame.
            #
            # Why this is the right lever for micro-expressions specifically:
            # the Face Adapter's encoder (wan/modules/animate/motion_encoder.py,
            # EncoderApp) is a plain conv stack over the whole 512x512 tile
            # down to one 512-dim vector. It has NO landmark input and no
            # spatial-alignment mechanism, so it cannot separate "the crop
            # wobbled 3px" from "the face moved 3px" — raw per-frame box noise
            # is encoded as motion and competes with the real expression
            # signal. Filtering the box removes that fake motion while leaving
            # every face PIXEL untouched (smoothing pixels would erase the
            # micro-expression itself, which is why only the box is filtered).
            _rc = np.stack(raw_centers, axis=0).astype(np.float32) \
                if raw_centers else np.zeros((B, 2), np.float32)
            _rw = np.array([float(b[1] - b[0]) for b in raw_face_bboxes], np.float32)
            _rh = np.array([float(b[3] - b[2]) for b in raw_face_bboxes], np.float32)
            _img_diag = float((W * W + H * H) ** 0.5)
            if expression_lock:
                # ── expression_lock ──────────────────────────────────────
                # The centre is NOT filtered. This is the whole point.
                #
                # Wan-Animate's face branch is a LIA motion encoder whose
                # output is dim_motion=20 (model_animate.py: Generator(
                # size=512, style_dim=512, motion_dim=20)). Twenty numbers per
                # frame carry EVERYTHING the face contributes: expression,
                # secondary motion, water, blood, a strip of tape lifting in
                # the wind. It has no landmark input and no alignment stage,
                # so it cannot separate "the face moved inside the tile" from
                # "the face changed".
                #
                # The per-frame face-tight box IS the registration. Filtering
                # the centre makes the box lag the head, so the face slides
                # around inside the tile — measured 26-61px of wander in a 512
                # tile on a normal pan. A micro-expression is 2-5px of eyelid
                # and lip travel. So a filtered centre hands the encoder a
                # rigid-motion signal 5-20x LARGER than the thing it is
                # supposed to encode, and the 20 dims get spent on it. That is
                # why crop_mode='default' preserves expressions and the
                # smoothed modes do not.
                #
                # So: take the centre straight from the detector (registration
                # as good as 'default', measured 0.7px wander) and stabilise
                # ONLY the box SIZE. Size drift is far cheaper than
                # translation because the face stays centred while it changes,
                # and holding it steady stops the tile breathing, which is the
                # one real artefact 'default' still has.
                _sc = _rc
            else:
                _sc = _smooth_centers(
                    _rc, method=str(smoothing_method),
                    ema_strength=face_smoothing_strength, image_diag=_img_diag,
                    one_euro_min_cutoff=crop_one_euro_min_cutoff,
                    one_euro_beta=crop_one_euro_beta,
                    gaussian_window=int(crop_gaussian_window),
                )
            _size_beta = float(crop_size_one_euro_beta)
            if expression_lock:
                # ONE scale series + a CONSTANT aspect (fixed 2026-07-31).
                #
                # Smoothing width and height as two independent series lets
                # their RATIO drift: on a head turn w and h change at
                # different rates, the two filters lag differently, and the
                # box aspect wobbles frame to frame. Every box is then
                # stretched to a SQUARE 512x512, so an aspect wobble becomes
                # the face visibly squashing and stretching — warping jitter
                # that no amount of centre stability can hide, because it is
                # anisotropic scaling, not translation.
                #
                # Smooth a single SCALE (the geometric mean of w and h, which
                # tracks the face's area faithfully) and hold the aspect at
                # the clip median. The stretch to 512 is then IDENTICAL on
                # every frame, so the only thing that can change inside the
                # tile is the face itself.
                _scale = np.sqrt(np.maximum(_rw, 1.0) * np.maximum(_rh, 1.0))
                _ss = _smooth_1d(_scale, method=str(smoothing_method),
                                 ema_strength=face_smoothing_strength,
                                 scale_norm=max(float(np.mean(_scale)), 1.0),
                                 one_euro_min_cutoff=crop_one_euro_min_cutoff,
                                 one_euro_beta=_size_beta,
                                 gaussian_window=int(crop_gaussian_window))
                _asp_const = float(np.clip(
                    np.median(np.maximum(_rh, 1.0) / np.maximum(_rw, 1.0)),
                    0.25, 4.0))
                _root = float(np.sqrt(_asp_const))
                _sw = (_ss / _root).astype(np.float32)
                _sh = (_ss * _root).astype(np.float32)
                logging.getLogger(__name__).info(
                    "PoseAndFaceDetectionV2 [expression_lock]: aspect held "
                    "constant at %.3f for the whole clip (was drifting with "
                    "independently-filtered w/h, which shows up as the face "
                    "stretching frame to frame after the square 512 resize).",
                    _asp_const,
                )
            else:
                _sw = _smooth_1d(_rw, method=str(smoothing_method),
                                 ema_strength=face_smoothing_strength,
                                 scale_norm=max(float(np.mean(_rw)), 1.0),
                                 one_euro_min_cutoff=crop_one_euro_min_cutoff,
                                 one_euro_beta=_size_beta,
                                 gaussian_window=int(crop_gaussian_window))
                _sh = _smooth_1d(_rh, method=str(smoothing_method),
                                 ema_strength=face_smoothing_strength,
                                 scale_norm=max(float(np.mean(_rh)), 1.0),
                                 one_euro_min_cutoff=crop_one_euro_min_cutoff,
                                 one_euro_beta=_size_beta,
                                 gaussian_window=int(crop_gaussian_window))
            # crop_safety_margin exists to absorb FILTER LAG: it inflates the
            # box so a crop that trails the head still contains it.
            # expression_lock has no lag by construction (the centre is the
            # detector's own, unfiltered), so that inflation buys nothing and
            # costs real face resolution — a 1.12 margin spends 20% of the
            # tile AREA on background the encoder then has to ignore, out of
            # the same 20-dim budget. Use the reference box exactly.
            # For deliberate headroom (tape or blood above the brow, hair
            # movement) raise face_crop_scale instead: that is the reference's
            # own framing parameter and it expands the box the way the encoder
            # was trained to see.
            _mar = 1.0 if expression_lock else max(1.0, float(crop_safety_margin))
            face_bboxes = []
            for _i in range(B):
                _w = float(np.clip(_sw[_i] * _mar, 8.0, float(W)))
                _h = float(np.clip(_sh[_i] * _mar, 8.0, float(H)))
                _cx, _cy = float(_sc[_i, 0]), float(_sc[_i, 1])
                _x1 = int(round(_cx - _w / 2.0))
                _y1 = int(round(_cy - _h / 2.0))
                # Left unclamped on purpose — _crop_with_padding edge-pads so
                # the face stays centred at frame edges instead of the box
                # being shoved back inside (which is what put the face
                # off-centre before).
                face_bboxes.append((_x1, _x1 + int(round(_w)),
                                    _y1, _y1 + int(round(_h))))
            logging.getLogger(__name__).info(
                "PoseAndFaceDetectionV2 [%s]: reference box geometry (scale=%.2f); "
                "centre %s, size filtered (%s). Box jitter std: cx %.2f->%.2f px, "
                "w %.2f->%.2f px.",
                crop_mode_str, float(face_crop_scale),
                ("RAW per-frame (registration preserved for the 20-dim face encoder)"
                 if expression_lock else "filtered"),
                str(smoothing_method),
                float(np.std(np.diff(_rc[:, 0]))) if B > 1 else 0.0,
                float(np.std(np.diff(_sc[:, 0]))) if B > 1 else 0.0,
                float(np.std(np.diff(_rw))) if B > 1 else 0.0,
                float(np.std(np.diff(_sw))) if B > 1 else 0.0,
            )
        elif jitterless:
            # ── Jitterless crop pipeline ─────────────────────────────────
            # 1. Resolve the frame-0 anchor (size & center) with priority:
            #      bbox_override > frame0_cx/cy widgets > raw detection.
            #    The locked size is then used for the whole clip.
            anchor_cx = None
            anchor_cy = None
            anchor_size = None
            if bbox_override is not None:
                try:
                    bb = bbox_override
                    # Accept (x1,y1,x2,y2) or (x1,x2,y1,y2) — heuristic:
                    if isinstance(bb, (list, tuple)) and len(bb) > 0 and isinstance(bb[0], (list, tuple)):
                        bb = bb[0]
                    bb = [float(v) for v in (bb[:4] if hasattr(bb, "__len__") else [])]
                    if len(bb) == 4:
                        # detect ordering by checking which pair is closer
                        a, b, c, d = bb
                        # try (x1,y1,x2,y2)
                        x1o, y1o, x2o, y2o = sorted([a, c])[0], sorted([b, d])[0], sorted([a, c])[1], sorted([b, d])[1]
                        anchor_cx = 0.5 * (x1o + x2o)
                        anchor_cy = 0.5 * (y1o + y2o)
                        anchor_size = max(x2o - x1o, y2o - y1o)
                except Exception as e:
                    print(f"[PoseAndFaceDetectionV2] bbox_override parse failed: {e}; ignoring.")
            if (anchor_cx is None) and frame0_cx >= 0 and frame0_cy >= 0:
                anchor_cx = float(frame0_cx)
                anchor_cy = float(frame0_cy)
            if anchor_cx is None and len(raw_centers) > 0:
                anchor_cx = float(raw_centers[0][0])
                anchor_cy = float(raw_centers[0][1])
            if anchor_size is None or anchor_size <= 0:
                if frame0_size and int(frame0_size) > 0:
                    anchor_size = float(frame0_size)
                else:
                    anchor_size = _locked_crop_side(
                        face_box_size_px, raw_face_bboxes, W, H, "jitterless")
            anchor_size = float(np.clip(anchor_size, 8.0, max(W, H)))
            anchor_cx = 0.0 if anchor_cx is None else float(np.clip(anchor_cx, 0.0, W - 1))
            anchor_cy = 0.0 if anchor_cy is None else float(np.clip(anchor_cy, 0.0, H - 1))

            # 1b. (The old face-scale ratio that scaled the crop by the
            #    per-frame detected face width lived here. It is gone: it is
            #    precisely what stopped "jitterless" from holding a locked
            #    size. That behaviour is still reachable via crop_mode="auto".)
            # (the old _missing_bbox sentinel lived here — replaced by the
            # explicit raw_face_missing flags built with the boxes, because the
            # sentinel's tuple order did not match what the fallback appends
            # and so it never matched anything)

            # 2. Build the per-frame target-center series.
            #    Start from raw detected centers (so the face is followed
            #    when the user adds no keyframes) and overwrite with
            #    interpolated keyframe centers wherever keyframes exist.
            kfs = _parse_keyframes_json(keyframes_json, B)
            # Always anchor frame 0 to (anchor_cx, anchor_cy, anchor_size)
            kfs = [k for k in kfs if k["frame"] != 0]
            kfs.insert(0, {"frame": 0, "cx": anchor_cx, "cy": anchor_cy, "size": int(anchor_size)})

            kf_cx, kf_cy, kf_sz = _interp_keyframes(
                kfs, B,
                default_cx=anchor_cx,
                default_cy=anchor_cy,
                default_size=anchor_size,
            )

            # If user supplied >1 keyframes (besides frame 0), trust them
            # fully for the center; otherwise blend with the smoothed raw
            # detection so the crop still tracks the face.
            user_added = len([k for k in kfs if k["frame"] != 0])
            target_centers = np.stack(raw_centers, axis=0).astype(np.float32) \
                if raw_centers else np.zeros((B, 2), dtype=np.float32)
            if user_added >= 1:
                # The user-controlled trajectory wins.
                target_centers[:, 0] = kf_cx
                target_centers[:, 1] = kf_cy
            else:
                # No extra keyframes — keep the detected centers but
                # snap frame 0 to the anchor (so the user's manual frame-0
                # override actually takes effect).
                target_centers[0, 0] = anchor_cx
                target_centers[0, 1] = anchor_cy

            # 2b. Build per-frame target crop sizes.
            #    BUG FIX (2026-07-24): jitterless now actually LOCKS the size,
            #    which is what its own tooltip has always promised ("lock crop
            #    SIZE from frame 0"). The previous implementation computed a
            #    face-scale-preserving size PER FRAME
            #    (raw_face_size[i] * anchor_scale_ratio), so asking for a
            #    locked 288px crop produced a different width on essentially
            #    every frame — measured on a 120-frame walk-toward-camera test
            #    clip: 120 distinct widths spanning 288..637px. That defeats
            #    the entire point of the mode (a Mocha-style planar hold) and
            #    made the encoder see a different effective zoom each frame.
            #    Face-scale-preserving is still available: it is what
            #    crop_mode="auto" does, and explicit key-frame sizes still win
            #    here for users who want to animate the size deliberately.
            lock_size = True
            if user_added >= 1 and kf_sz is not None:
                target_sizes = kf_sz.copy()
                lock_size = False   # user is driving size explicitly
            else:
                target_sizes = np.full((B,), float(anchor_size), dtype=np.float32)
            target_sizes = np.clip(target_sizes, 8.0, float(min(W, H)))

            # 2c. Bug-fix (Wan-Animate spec 1.3): eye-centred crop MUST be
            # folded into the target-center trajectory BEFORE smoothing, not
            # applied as a post-pass on top of the already-smoothed bboxes.
            # The old order was: raw detection -> one_euro/EMA smoothing ->
            # THEN shift y1/y2 using a FRESH, UNSMOOTHED per-frame eye-landmark
            # read (see the removed post-pass a few dozen lines below this
            # function's face_bboxes assembly). Any noise in that single
            # frame's eye-landmark estimate went straight into the final crop
            # with zero filtering — a direct source of gaze-tracking flicker.
            # Folding it in here means the eye-offset rides through the SAME
            # jitter-rejection filter as everything else.
            if str(eye_align_mode) == "eye_upper_third":
                _ey_frac = float(np.clip(eye_y_fraction, 0.05, 0.80))
                for _idx in range(B):
                    _eye_xy = compute_eye_midpoint_from_face_kps(
                        pose_metas[_idx]['keypoints_face'], W, H
                    )
                    if _eye_xy is None:
                        continue
                    _cx, _cy = apply_eye_offset_to_center(
                        (target_centers[_idx, 0], target_centers[_idx, 1]),
                        _eye_xy, float(target_sizes[_idx]), H, _ey_frac,
                    )
                    target_centers[_idx, 0] = _cx
                    target_centers[_idx, 1] = _cy

            # 3-5. Smooth, bound the filter lag, and build the boxes.
            #    Extracted to build_jitterless_boxes() so the two guarantees
            #    this mode makes can be asserted numerically without standing
            #    up ComfyUI (see the offline test harness):
            #      * every box is EXACTLY the locked size, and
            #      * the smoothed centre never lags the true centre by more
            #        than _JITTERLESS_MAX_CENTER_DRIFT_FRAC of that size.
            #    The drift bound is the second half of this fix: a temporal
            #    filter ALWAYS lags on fast motion, and previously that lag
            #    was unbounded, so during a fast whip the face slid toward
            #    (and could leave) the crop edge. Measured 26% of the tile
            #    off-centre on the test clip before the bound; 12% after.
            _hold = [
                (raw_face_missing[i] and user_added == 0)
                for i in range(B)
            ]
            face_bboxes, smoothed_sizes, smoothed_centers = build_jitterless_boxes(
                target_centers=target_centers,
                target_sizes_raw=target_sizes,
                anchor_size=float(anchor_size),
                W=W, H=H,
                smoothing_method=str(smoothing_method),
                face_smoothing_strength=face_smoothing_strength,
                one_euro_min_cutoff=crop_one_euro_min_cutoff,
                one_euro_beta=crop_one_euro_beta,
                size_one_euro_beta=crop_size_one_euro_beta,
                gaussian_window=int(crop_gaussian_window),
                lock_size=lock_size,
                hold_mask=_hold,
                safety_margin=float(crop_safety_margin),
                containment_boxes=(raw_face_bboxes if crop_containment_check else None),
                containment_tolerance=float(crop_containment_tolerance),
                aspect_ratios=(raw_face_aspects if preserve_face_aspect else None),
            )
            _cstats = getattr(build_jitterless_boxes, "last_stats", {}) or {}
            if _cstats.get("shifted") or _cstats.get("too_small"):
                logging.getLogger(__name__).info(
                    "PoseAndFaceDetectionV2 [jitterless]: containment shifted %d/%d frames; "
                    "%d frame(s) had a face LARGER than the locked crop.%s",
                    int(_cstats.get("shifted", 0)), int(_cstats.get("frames", B)),
                    int(_cstats.get("too_small", 0)),
                    ("  The locked size is too small for this shot — raise "
                     "face_box_size_px/frame0_size or crop_safety_margin. "
                     "(Not auto-grown: that would break the exact-size lock.)"
                     if _cstats.get("too_small") else ""),
                )
        else:
            # ── Legacy auto pipeline ───────────────────────────────────────
            # Bug-fix (Wan-Animate spec 1.3, legacy-path counterpart): fold
            # the eye-offset into raw_centers BEFORE the motion-adaptive
            # smoothing loop below, not as a post-pass on the finished bboxes
            # (see the new/jitterless pipeline above for the full rationale —
            # the same jitter-reintroduction bug applied here too whenever
            # crop_mode="auto"). Crop height per frame: face_box_size_px when
            # constant-size is on, else the raw bbox height inflated by the
            # same 0.3*2 y-padding the "not constant" branch below applies
            # (so the offset targets the crop height that actually ships).
            if str(eye_align_mode) == "eye_upper_third" and len(raw_centers) > 0:
                _ey_frac = float(np.clip(eye_y_fraction, 0.05, 0.80))
                for _idx in range(len(raw_centers)):
                    _eye_xy = compute_eye_midpoint_from_face_kps(
                        pose_metas[_idx]['keypoints_face'], W, H
                    )
                    if _eye_xy is None:
                        continue
                    if use_constant_face_box:
                        _crop_h = float(face_box_size_px)
                    else:
                        _rx1, _rx2, _ry1, _ry2 = raw_face_bboxes[_idx]
                        _crop_h = float(_ry2 - _ry1) * 1.6
                    _cx, _cy = apply_eye_offset_to_center(
                        (raw_centers[_idx][0], raw_centers[_idx][1]),
                        _eye_xy, _crop_h, H, _ey_frac,
                    )
                    raw_centers[_idx] = np.array([_cx, _cy], dtype=np.float32)

            # --- Temporal smoothing for centers (motion-adaptive) ---
            # auto_smoothing_method="legacy_ema" (default) keeps auto's own
            # bespoke EMA byte-for-byte so existing workflows are untouched.
            # Any other value routes auto through the SAME shared filters
            # jitterless uses, honouring crop_one_euro_* / crop_gaussian_window
            # — previously auto ignored the smoothing_method widget entirely.
            _auto_sm = str(auto_smoothing_method)
            if use_face_smoothing and len(raw_centers) > 1 and _auto_sm != "legacy_ema":
                smoothed_centers = list(_smooth_centers(
                    np.stack(raw_centers, axis=0).astype(np.float32),
                    method=_auto_sm,
                    ema_strength=face_smoothing_strength,
                    image_diag=float((W * W + H * H) ** 0.5),
                    one_euro_min_cutoff=crop_one_euro_min_cutoff,
                    one_euro_beta=crop_one_euro_beta,
                    gaussian_window=int(crop_gaussian_window),
                ))
            elif use_face_smoothing and len(raw_centers) > 1:
                base_strength = float(np.clip(face_smoothing_strength, 0.0, 1.0))
                norm = max(1.0, (W + H) / 2.0)

                def _legacy_ema(seq):
                    """auto's own motion-adaptive EMA, unchanged, one direction."""
                    out = [seq[0].copy()]
                    for _j in range(1, len(seq)):
                        curr = seq[_j]
                        prev = out[-1]
                        motion = float(np.mean(np.abs(curr - prev)) / norm)
                        # More motion -> less smoothing
                        k = 5.0
                        dynamic_strength = base_strength * np.exp(-motion * k)
                        alpha = 1.0 - dynamic_strength  # 1=no smoothing, 0=full
                        out.append((alpha * curr + (1.0 - alpha) * prev).astype(np.float32))
                    return out

                # Zero-phase, same reason as _smooth_centers. This branch is
                # auto's bespoke EMA and never went through the shared filter,
                # so it kept its causal lag after that fix: on a sustained pan
                # the crop trailed the subject by a near-constant amount and
                # the face sat to one side of the tile for the whole move
                # (measured 34px of a 512 tile, always the same direction).
                # Forward+backward average cancels it; the per-step response is
                # untouched, so a still subject smooths exactly as before.
                _fwd = _legacy_ema(raw_centers)
                _bwd = _legacy_ema(raw_centers[::-1])[::-1]
                smoothed_centers = [
                    (0.5 * (a + b)).astype(np.float32) for a, b in zip(_fwd, _bwd)
                ]
            else:
                smoothed_centers = raw_centers

            # --- Build final face bboxes from smoothed centers ---
            face_bboxes = []
            if use_constant_face_box:
                # crop_safety_margin also applies here so 'auto' gets the same
                # protection against filter lag / foreshortened detections.
                _auto_base = _locked_crop_side(
                    face_box_size_px, raw_face_bboxes, W, H, "auto")
                _auto_side = int(round(_auto_base * max(1.0, float(crop_safety_margin))))
                _auto_side = max(8, min(_auto_side, int(min(W, H))))
                half = _auto_side / 2.0
                _tol = float(crop_containment_tolerance)
                _auto_shift = 0
                for _i, c in enumerate(smoothed_centers):
                    cx, cy = float(c[0]), float(c[1])
                    # Containment: shift so the real detected face box is
                    # inside this constant-size crop (size stays constant).
                    if crop_containment_check and _i < len(raw_face_bboxes):
                        _fb = raw_face_bboxes[_i]
                        if _fb is not None:
                            fx1, fx2, fy1, fy2 = (float(_fb[0]), float(_fb[1]),
                                                  float(_fb[2]), float(_fb[3]))
                            lo_x, hi_x = (fx2 + _tol) - half, (fx1 - _tol) + half
                            lo_y, hi_y = (fy2 + _tol) - half, (fy1 - _tol) + half
                            n_cx = min(max(cx, lo_x), hi_x) if lo_x <= hi_x else 0.5 * (fx1 + fx2)
                            n_cy = min(max(cy, lo_y), hi_y) if lo_y <= hi_y else 0.5 * (fy1 + fy2)
                            if abs(n_cx - cx) > 1e-3 or abs(n_cy - cy) > 1e-3:
                                _auto_shift += 1
                            cx, cy = n_cx, n_cy
                    # Paper framing: reproduce Wan-Animate's non-square,
                    # top-biased crop instead of forcing a square (see
                    # preserve_face_aspect). Height follows the tracked
                    # aspect; width is the constant box side.
                    if preserve_face_aspect and _i < len(raw_face_aspects):
                        _asp = float(np.clip(raw_face_aspects[_i], 0.25, 4.0))
                    else:
                        _asp = 1.0
                    # Aspect-preserving fit (same rule as jitterless): scale
                    # BOTH sides if the height will not fit, never clamp the
                    # height alone — that silently rewrites the aspect.
                    # Fit into a LOCAL pair — never back into _auto_side. That
                    # variable is the clip-wide constant box side; reassigning
                    # it here fed the fitted (smaller) width into the NEXT
                    # frame, so one frame that needed shrinking shrank every
                    # frame after it and the "constant" box decayed down the
                    # clip. `half` had the same problem: it was recomputed at
                    # the END of an iteration and then used by the containment
                    # clamp at the START of the next one, so the clamp window
                    # was built from a different size than the box, which
                    # shifted the centre sideways.
                    _aw, _ah = float(_auto_side), float(_auto_side) * _asp
                    if _ah > H:
                        _ah = float(H); _aw = _ah / max(_asp, 1e-6)
                    if _aw > W:
                        _aw = float(W); _ah = _aw * _asp
                    _fit_w = max(8, int(round(_aw)))
                    _auto_h = max(8, int(round(_ah)))
                    # NOT clamped into the frame — _crop_with_padding edge-pads
                    # at extraction so the face stays centred (same fix as
                    # jitterless; clamping here pushed the face off-centre too).
                    x1 = int(round(cx - _fit_w / 2.0))
                    y1 = int(round(cy - _auto_h / 2.0))
                    face_bboxes.append((x1, x1 + _fit_w, y1, y1 + _auto_h))
                if _auto_shift:
                    logging.getLogger(__name__).info(
                        "PoseAndFaceDetectionV2 [auto]: containment shifted %d/%d frames "
                        "(constant box %dpx incl. %.2fx safety margin).",
                        _auto_shift, len(smoothed_centers), _auto_side,
                        float(crop_safety_margin),
                    )
            else:
                # If not constant size, just slightly pad the original (helps tilted heads)
                for (x1, x2, y1, y2), c in zip(raw_face_bboxes, smoothed_centers):
                    w = x2 - x1
                    h = y2 - y1
                    x_pad = int(w * 0.2)
                    y_pad = int(h * 0.3)
                    # Recenter to smoothed center but keep variable size
                    cx, cy = float(c[0]), float(c[1])
                    half_w = (w / 2.0) + x_pad
                    half_h = (h / 2.0) + y_pad
                    nx1 = int(np.clip(cx - half_w, 0, W - 1))
                    ny1 = int(np.clip(cy - half_h, 0, H - 1))
                    nx2 = int(np.clip(cx + half_w, 0, W))
                    ny2 = int(np.clip(cy + half_h, 0, H))
                    if nx2 <= nx1 or ny2 <= ny1:
                        nx1, ny1, nx2, ny2 = x1, y1, x2, y2  # fallback to raw
                    face_bboxes.append((nx1, nx2, ny1, ny2))

        # Wide-shot source-pixel floor (Kijai 128–192 / paper scale-aug).
        # A 46px full-face box Lanczos'd to 512 invents 99% of the encoder
        # input. Expand around the same centre so max(w,h) >= 128 (capped
        # to the shorter frame axis). Close-ups already above the floor
        # are unchanged — face still fills the tile.
        _floor_raised = 0
        if face_bboxes:
            face_bboxes, _floor_raised = _floor_face_boxes(
                face_bboxes, int(_SOURCE_FACE_MIN_PX), W, H,
            )
            if _floor_raised:
                _fw = face_bboxes[0][1] - face_bboxes[0][0]
                _fh = face_bboxes[0][3] - face_bboxes[0][2]
                _side = max(_fw, _fh, 1)
                logging.getLogger(__name__).info(
                    "PoseAndFaceDetectionV2 [%s]: source-crop floor raised "
                    "%d/%d boxes to %dpx (paper scale-aug / Kijai wide-shot). "
                    "Lanczos to 512 is now %.1fx, not 11x.",
                    crop_mode_str, _floor_raised, len(face_bboxes),
                    int(_SOURCE_FACE_MIN_PX), 512.0 / _side,
                )

        # Wan-Animate paper recommendation #1 (eye-centred crop) is now
        # applied EARLIER, inside each pipeline branch above, folded into the
        # center trajectory before smoothing (spec 1.3) — see the "2c. Bug-fix"
        # block (new/jitterless pipeline) and the legacy-path block right
        # before its motion-adaptive smoothing loop. No post-pass needed here
        # any more; `adjust_bbox_eye_upper_third` stays in utils.py for any
        # external caller but this node no longer uses it as a post-hoc patch.

        # Bug-fix (Wan-Animate spec 1.4): the standalone QualityScorerJitter
        # node computes a blur/jitter score but nothing wires it back to
        # protect the crop — a blurry frame (motion blur, autofocus hunt,
        # momentary occlusion) just gets cropped and fed to the face encoder
        # as-is. Add an inline safety net here: if a candidate crop scores
        # below the same Laplacian-variance threshold QualityScorerJitter
        # documents as its default (50.0), hold the PREVIOUS frame's crop
        # GEOMETRY (not pixels — face_images below still reads the current,
        # sharp source frame at that geometry) — the same hold-last-known
        # pattern already used for missing detections above. Frame 0 always
        # keeps its own bbox (nothing earlier to hold).
        # BLUR-HOLD REMOVED (2026-07-24) — it was the cause of "the face is
        # off-centre in every frame", in every crop mode.
        #
        # It used to replace a blurry frame's crop with the PREVIOUS frame's
        # box:   face_bboxes[i] = face_bboxes[i - 1]
        # That CASCADES: once one frame is held, the next frame is measured
        # against the HELD box, so a run of blurry frames freezes the crop in
        # place. Motion blur happens precisely when the head turns — i.e.
        # exactly when the face is moving — so the box locks at the pre-motion
        # position while the face travels away from it, and stays locked until
        # a sharp frame happens to land. It ran AFTER the per-mode box build,
        # on all modes, which is why default / auto / jitterless /
        # reference_smooth were all affected identically.
        #
        # Deleted rather than re-tuned: the requirement here is that the face
        # is centred in EVERY frame, and any mechanism that substitutes a stale
        # crop violates that by construction, at any threshold. The reference
        # pipeline (wan/modules/animate/preprocess/process_pipepline.py) has no
        # such step — it crops each frame from that frame's own face box.

        # --- Centring self-check (always logged) -------------------------
        # Reports how far the DETECTED face centre sits from the centre of the
        # tile that actually ships as face_images. This is the number to look
        # at when the face "looks offset" — it is measured on the real boxes,
        # so it answers the question directly instead of eyeballing a sampler
        # preview (which is mid-denoise and proves nothing about the crop).
        try:
            _offs = []
            for _i, (_bx1, _bx2, _by1, _by2) in enumerate(face_bboxes):
                if _i >= len(raw_centers):
                    break
                _tw = max(1, _bx2 - _bx1)
                _th = max(1, _by2 - _by1)
                _dx = float(raw_centers[_i][0]) - (_bx1 + _bx2) / 2.0
                _dy = float(raw_centers[_i][1]) - (_by1 + _by2) / 2.0
                _offs.append(math.hypot(_dx / _tw, _dy / _th))
            if _offs:
                _mx = max(_offs)
                _tile_w = face_bboxes[0][1] - face_bboxes[0][0]
                _tile_h = face_bboxes[0][3] - face_bboxes[0][2]
                _tile_side = max(_tile_w, _tile_h)
                _tiny = _tile_side < int(_SOURCE_FACE_MIN_PX)
                _lvl = logging.WARNING if (_mx > 0.20 or _tiny) else logging.INFO
                _tiny_msg = (
                    "  SOURCE FACE IS %dx%d PX — upscale to 512 is %.1fx, "
                    "eyeballs are roughly %d source pixels. No crop mode "
                    "recovers iris / micro-expression from that. Use a "
                    "tighter shot or a higher-res plate (1080p/4K of the "
                    "same framing). Motion blur and colour in the plate "
                    "are already in these pixels; they cannot be invented."
                    % (_tile_w, _tile_h, 512.0 / max(_tile_side, 1),
                       max(1, int(round(_tile_side * 4 / 46.0))))
                    if _tiny else
                    ("  Raise crop_safety_margin / lower face_box_size_px, or "
                     "use crop_mode='default' for the paper's exact per-frame "
                     "face-tight crop." if _mx > 0.20 else "")
                )
                logging.getLogger(__name__).log(
                    _lvl,
                    "PoseAndFaceDetectionV2 [%s]: face-centre offset within the "
                    "face_images tile — mean %.1f%%, max %.1f%% of tile "
                    "(0%% = perfectly centred; >20%% is visibly off and the face "
                    "encoder will struggle). Tile %dx%d.%s",
                    crop_mode_str, 100.0 * float(np.mean(_offs)), 100.0 * _mx,
                    _tile_w, _tile_h, _tiny_msg,
                )
        except Exception:  # noqa: BLE001 — diagnostics must never break the node
            pass

        # --- Face crops from sharp original frames ---
        # PAD, don't clamp (2026-07-24 bug fix). A crop box may legitimately
        # extend past the frame edge — that is exactly what happens when the
        # head is near the top of frame (where heads are) and the locked
        # jitterless size is a large fraction of the frame height. The old
        # code clamped the box back inside, which silently DESTROYED centring:
        # measured on 832x480 with the default 512 crop, the box could not move
        # vertically at all (H - side == 0) so the face sat ~90px = 19% of the
        # tile off-centre on EVERY frame.
        #
        # Why centring matters here specifically: Wan-Animate's Face Adapter
        # takes "the raw facial image directly as the driving input", located
        # by cropping the face region and "resized to 512x512" (paper 3.3,
        # arXiv:2509.14055), and the reference pipeline
        # (wan/modules/animate/preprocess/process_pipepline.py) crops a
        # per-frame FACE-TIGHT bbox so the face is centred and fills the tile
        # by construction. An off-centre face is out-of-distribution input to
        # that encoder. Edge-replicate padding keeps the face dead-centre and
        # introduces no hard synthetic border (the encoder was trained with
        # scale/colour/noise augmentation, not with black bars).
        # ── Take the face crop from the HIGHEST-RESOLUTION source available ──
        # (added 2026-08-13) This is the single biggest quality lever in the
        # whole node, and it is not a crop-mode choice.
        #
        # Detection runs at the working resolution because that is what ViTPose
        # wants and it is fast. But the face TILE was then cut from that same
        # small image and upscaled to 512 for the encoder. On a 832x480 plate a
        # face box measures ~46px, so the tile is an ELEVEN-times upscale —
        # 99.2% of the pixels the encoder reads are invented, and an eyeball at
        # that scale is about 4 pixels across. There is no iris in 4 pixels,
        # which is why no crop_mode ever fixed eye direction or micro-detail:
        # every mode was cropping the same starved image.
        #
        # If a hi-res plate is wired to `hires_images`, the box is scaled into
        # its coordinate space and the tile is cut from THERE. Same framing,
        # real pixels: 46px -> 106px from 1080p, 212px from 4K.
        _hi = None
        _hi_sx = _hi_sy = 1.0
        if hires_images is not None:
            _hi_np = (hires_images.detach().cpu().numpy()
                      if hasattr(hires_images, "detach") else np.asarray(hires_images))
            if _hi_np.ndim == 4 and _hi_np.shape[0] >= 1:
                _hH, _hW = int(_hi_np.shape[1]), int(_hi_np.shape[2])
                if _hW >= W and _hH >= H and (_hW > W or _hH > H):
                    _hi = _hi_np
                    _hi_sx, _hi_sy = float(_hW) / float(W), float(_hH) / float(H)
                    logging.getLogger(__name__).info(
                        "PoseAndFaceDetectionV2: face crop taken from hires_images "
                        "%dx%d instead of the %dx%d working plate (%.2fx linear). "
                        "The face tile gains that factor in REAL pixels — this is "
                        "the lever that actually raises eye/micro-expression "
                        "detail, not the crop_mode.", _hW, _hH, W, H, _hi_sx,
                    )
                else:
                    logging.getLogger(__name__).warning(
                        "PoseAndFaceDetectionV2: hires_images is %dx%d, which is not "
                        "larger than the %dx%d working plate — ignoring it. Wire the "
                        "FULL-RESOLUTION plate here (the EXR/source sequence), not a "
                        "copy of the resized video.", _hW, _hH, W, H,
                    )

        face_images = []
        for idx, (x1, x2, y1, y2) in enumerate(face_bboxes):
            if _hi is not None:
                _src = _hi[min(idx, _hi.shape[0] - 1)]
                face_image = _crop_with_padding(
                    _src,
                    int(round(x1 * _hi_sx)), int(round(x2 * _hi_sx)),
                    int(round(y1 * _hi_sy)), int(round(y2 * _hi_sy)),
                )
            else:
                face_image = _crop_with_padding(images_np[idx], x1, x2, y1, y2)
            if face_image.size == 0:
                fallback_size = int(min(H, W) * 0.3)
                fx1 = (W - fallback_size) // 2
                fx2 = fx1 + fallback_size
                fy1 = int(H * 0.1)
                fy2 = fy1 + fallback_size
                face_image = images_np[idx][fy1:fy2, fx1:fx2]
                if face_image.size == 0:
                    face_image = np.zeros((fallback_size, fallback_size, C), dtype=images_np.dtype)
            # Keep the RAW crop; the resize to 512 happens after the SR pass
            # below so SR can work on native pixels instead of on Lanczos
            # ringing. See the _face_sr note for why that ordering matters.
            face_images.append(face_image)

        # ── FACE SUPER-RESOLUTION + the resize to 512 ────────────────────
        # ORDER MATTERS. The tile must reach 512x512 because that is what
        # Wan-Animate's motion encoder takes, but HOW it gets there decides how
        # much of what the encoder reads is real.
        #
        # The old path was a single cv2 INTER_LANCZOS4 upscale. Lanczos is a
        # windowed sinc: its kernel has NEGATIVE lobes, so on a hard edge it
        # overshoots and undershoots — visible ringing. At the 11x upscale a
        # small face needs, that ringing is a large fraction of the signal, and
        # it is invented structure the encoder cannot tell from real texture.
        # It is also why a crop resized to 512 and then sampled back down does
        # NOT match the original sampled down directly: the ringing does not
        # cancel.
        #
        # So SR (when enabled) runs on the NATIVE crop first, and only its
        # output is taken to 512. Running SR after a Lanczos upscale would just
        # be sharpening the ringing.
        _sr_mode = str(face_sr or "none").strip().lower()
        _sr_note = "off"
        if _sr_mode not in ("", "none") and len(face_images) > 0:
            try:
                from .nodes_extras import _face_sr as _FSR
                _u8 = []
                for _t in face_images:
                    _a = _t
                    if np.issubdtype(_a.dtype, np.floating):
                        _a = (np.clip(_a, 0.0, 1.0) * 255.0).astype(np.uint8)
                    _u8.append(_a)
                _box_w = float(np.mean([t.shape[1] for t in face_images]))
                _iod, _verdict = _FSR.face_box_health(_box_w)
                if _sr_mode == "lanczos":
                    _sr_out, _sr_note = _FSR._backend_lanczos(_u8)
                else:
                    _sr_out, _sr_note = _FSR._backend_comfy_upscale(_u8, str(face_sr_model))
                _sr_out = _FSR.temporally_stabilise(
                    _sr_out, _u8, strength=float(face_sr_stabilise), window=5)
                _sharp0 = float(np.mean([_FSR.sharpness(t) for t in _u8]))
                _sharp1 = float(np.mean([_FSR.sharpness(t) for t in _sr_out]))
                _flick1 = _FSR.temporal_flicker(np.stack([
                    cv2.resize(t, (128, 128), interpolation=cv2.INTER_AREA) for t in _sr_out], 0))
                face_images = [t.astype(np.float32) / 255.0 for t in _sr_out]
                logging.getLogger(__name__).info(
                    "PoseAndFaceDetectionV2: face_sr=%s on a %.0fpx face box "
                    "(interocular ~%.0fpx, %s). Sharpness %.1f -> %.1f; residual "
                    "detail flicker %.5f at stabilise=%.2f.",
                    _sr_note, _box_w, _iod, _verdict, _sharp0, _sharp1,
                    _flick1, float(face_sr_stabilise),
                )
            except Exception as _sr_exc:  # noqa: BLE001
                # Loud, not silent: SR is opt-in, so a user who asked for it and
                # did not get it must be told, with the reason and the fix.
                logging.getLogger(__name__).warning(
                    "PoseAndFaceDetectionV2: face_sr=%r FAILED (%s). Falling back to "
                    "the plain resize — the tile is unchanged, not corrupted. %s",
                    _sr_mode, _sr_exc,
                    "Install an upscale model in ComfyUI/models/upscale_models, or "
                    "set face_sr='lanczos' which needs nothing."
                    if _sr_mode == "comfy_upscale" else "",
                )

        # Nuke-equivalent resample. See face_resize_filter's tooltip for the
        # measured round-trip numbers; the old hardcoded cv2 INTER_LANCZOS4 was
        # about 2.4x worse than mitchell on the 46px-face case.
        try:
            from .nodes_extras import _resize_filters as _RF
            face_images = [_RF.resize(_t, 512, 512, str(face_resize_filter))
                           for _t in face_images]
        except Exception as _rf_exc:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "PoseAndFaceDetectionV2: face_resize_filter=%r unavailable (%s); "
                "using the legacy cv2 resize.", face_resize_filter, _rf_exc)
            face_images = [resize_face_crop(_t, 512) for _t in face_images]
        face_images_np = np.stack(face_images, 0)
        face_images_tensor = torch.from_numpy(face_images_np)
        # RAM (2026-08-13): the face_images list holds N per-frame 512x512x3
        # float32 arrays (~3MB each) alongside the np.stack copy. The tensor
        # shares the stack's memory, so the list is now redundant — drop it
        # and the per-frame arrays so a long clip doesn't hold 2x its face
        # data in RAM. For a 1000-frame clip this frees ~3GB.
        del face_images

        # ---- Manual landmark corrections (Pose editor, stage 2) ----
        # The pose_gaze_viewer's Edit mode writes per-frame, per-joint body
        # keypoint corrections here so a MIS-DETECTED skeleton can be fixed by
        # hand and the fix flows through retargeting into pose_data + the
        # rendered pose images — not just the on-node preview. Coords arrive as
        # SOURCE pixels; keypoints_body is normalised 0..1, so we divide by W/H.
        # Row length is preserved so the downstream np.array(keypoints_body)
        # stays rectangular. Best-effort: a bad blob never breaks the node.
        try:
            _ov = json.loads(landmark_overrides_json) if landmark_overrides_json else None
            if isinstance(_ov, dict) and _ov:
                _n_applied = 0
                for _fk, _joints in _ov.items():
                    try:
                        _fi = int(_fk)
                    except (TypeError, ValueError):
                        continue
                    if not (0 <= _fi < len(pose_metas)) or not isinstance(_joints, dict):
                        continue
                    _kpb = pose_metas[_fi].get('keypoints_body')
                    if _kpb is None:
                        continue
                    # length template from a detected sibling → uniform rows
                    _tmpl_len = 3
                    for _s in _kpb:
                        if _s is not None and hasattr(_s, '__len__') and len(_s) >= 2:
                            _tmpl_len = len(_s)
                            break
                    for _jk, _xy in _joints.items():
                        try:
                            _j = int(_jk)
                        except (TypeError, ValueError):
                            continue
                        if not (0 <= _j < len(_kpb)):
                            continue
                        if not (isinstance(_xy, (list, tuple)) and len(_xy) >= 2):
                            continue
                        _xn = min(1.0, max(0.0, float(_xy[0]) / float(W))) if W else 0.0
                        _yn = min(1.0, max(0.0, float(_xy[1]) / float(H))) if H else 0.0
                        _old = _kpb[_j]
                        if _old is not None and hasattr(_old, '__len__') and len(_old) >= 2:
                            try:                       # numpy row / list → mutate in place
                                _old[0] = _xn
                                _old[1] = _yn
                            except (TypeError, IndexError):   # immutable (tuple) → rebuild same length
                                _new = list(_old)
                                _new[0] = _xn
                                _new[1] = _yn
                                _kpb[_j] = _new
                        else:                          # was undetected → add a visible keypoint
                            _row = [0.0] * _tmpl_len
                            if _tmpl_len >= 3:
                                _row[2] = 1.0
                            _row[0] = _xn
                            _row[1] = _yn
                            _kpb[_j] = _row
                        _n_applied += 1
                    pose_metas[_fi]['keypoints_body'] = _kpb
                if _n_applied:
                    logging.getLogger(__name__).info(
                        "PoseAndFaceDetectionV2: applied %d manual landmark correction(s).",
                        _n_applied,
                    )
        except Exception as _ov_exc:  # noqa: BLE001 — never break the node on a bad blob
            logging.getLogger(__name__).warning(
                "PoseAndFaceDetectionV2: landmark_overrides_json ignored (%s).", _ov_exc,
            )

        # Retarget onto the reference character when one was provided, else the
        # straight per-frame conversion. get_retarget_pose returns AAPoseMeta
        # objects just like from_humanapi_meta, so the draw path is unchanged.
        if refer_pose_meta is not None:
            try:
                # get_retarget_pose mutates its input metas in place (ndarray→
                # list, hands scaled to pixels). Pass a deep copy so the shared
                # pose_metas stays ndarray-typed for the iris / viewer / points
                # passes that run after this.
                import copy as _copy  # noqa: PLC0415
                _pm_for_rt = _copy.deepcopy(pose_metas)

                # ── use_flux: enhanced retargeting (Wan 2.2 Animate's 3rd mode) ─
                # FLUX.1-Kontext-dev normalizes the reference AND the first
                # template frame to a standard front-facing pose BEFORE
                # retargeting, so retargeting starts from a neutral instead of
                # carrying a 3/4-profile / head-tilt into the output. The
                # edited poses are re-detected and passed to get_retarget_pose
                # (whose 5-arg signature already accepts them). Ported from
                # wan/modules/animate/preprocess/process_pipepline.py.
                _tpl_edit_meta0 = None
                _refer_edit_meta = None
                if use_flux:
                    def _pose_of(img_np):
                        """Detect body pose on a single HxWx3 uint8 image, mirroring
                        the reference-detection block above. Returns an
                        AAPoseMeta-style dict or None on failure."""
                        _h, _w = img_np.shape[:2]
                        _shp = np.array([_h, _w])[None]
                        _dets = detector(
                            cv2.resize(img_np, (640, 640)).transpose(2, 0, 1)[None], _shp
                        )[0]
                        if isinstance(_dets, list) and len(_dets) > 0 and isinstance(_dets[0], dict):
                            _bb = _dets[0]["bbox"]
                        else:
                            _bb = None
                        if (_bb is None or len(_bb) < 5 or _bb[4] <= 0
                                or (_bb[2] - _bb[0]) < 10 or (_bb[3] - _bb[1]) < 10):
                            _bb = np.array([0, 0, _w, _h, 1.0], dtype=np.float32)
                        _rc, _rs = bbox_from_detector(_bb, input_resolution, rescale=rescale)
                        _cr = crop(img_np, _rc, _rs, (input_resolution[0], input_resolution[1]))[0]
                        _cr = preprocess_for_pose(
                            _cr, use_clahe, clahe_clip=clahe_clip_limit,
                            clahe_grid=clahe_grid_size, gamma=detect_gamma,
                            white_balance=detect_white_balance, denoise=detect_denoise,
                            sharpen=detect_sharpen, saturation=detect_saturation)
                        _nm = ((_cr - IMG_NORM_MEAN) / IMG_NORM_STD).transpose(2, 0, 1).astype(np.float32)
                        _kp = pose_model(_nm[None], np.array(_rc)[None], np.array(_rs)[None])
                        return load_pose_metas_from_kp2ds_seq(
                            _kp, width=_w, height=_h)[0]

                    # Resolve the FLUX model path: explicit widget value, or a
                    # ComfyUI folder lookup when left blank. Never silently
                    # disable use_flux — raise naming what was searched.
                    _flux_path = (flux_kontext_path or "").strip()
                    if not _flux_path:
                        try:
                            import folder_paths  # type: ignore[import-not-found]
                            for _key in ("flux", "checkpoints", "unet"):
                                for _d in folder_paths.get_folder_paths(_key) or []:
                                    if os.path.isdir(os.path.join(_d, "FLUX.1-Kontext-dev")):
                                        _flux_path = os.path.join(_d, "FLUX.1-Kontext-dev")
                                        break
                                    if os.path.basename(_d) == "FLUX.1-Kontext-dev":
                                        _flux_path = _d
                                        break
                                if _flux_path:
                                    break
                        except Exception:
                            pass
                    if not _flux_path:
                        raise FileNotFoundError(
                            "use_flux is on but FLUX.1-Kontext-dev was not found. "
                            "Set flux_kontext_path to the model folder, or drop "
                            "FLUX.1-Kontext-dev into ComfyUI/models/flux/. Download "
                            "from https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev")
                    _flux_pipe = load_flux_kontext(_flux_path)
                    _tpl_prompt, _refer_prompt = get_editing_prompts(
                        pose_metas, refer_pose_meta)
                    # Edit the reference to a standard front-facing pose.
                    _refer_edit = edit_with_flux(
                        _flux_pipe, refer_img_proc, _refer_prompt)
                    # Edit the first template frame (frames[1] in the reference;
                    # fall back to frames[0] for a single-frame clip).
                    _tpl_idx = 1 if B >= 2 else 0
                    _tpl_frame = (images[_tpl_idx].detach().cpu().numpy()
                                   * 255.0).clip(0, 255).astype(np.uint8)
                    _tpl_edit = edit_with_flux(_flux_pipe, _tpl_frame, _tpl_prompt)
                    # Re-detect pose on the edited images.
                    _refer_edit_meta = _pose_of(_refer_edit)
                    _tpl_edit_meta0 = _pose_of(_tpl_edit)
                    logging.getLogger(__name__).info(
                        "PoseAndFaceDetectionV2 [use_flux]: normalized reference "
                        "and template frame to a front-facing pose via "
                        "FLUX.1-Kontext-dev before retargeting.")
                    # VRAM (2026-08-13): FLUX.1-Kontext-dev is ~12B params and
                    # is only used for these TWO edits (not per-frame), so free
                    # its pipeline + cached VRAM immediately. Without this the
                    # 12B model stays resident for the whole ComfyUI session
                    # (the global _FLUX_CACHE never cleared) — the "eating a
                    # lot of VRAM" the user reported. A re-run that needs FLUX
                    # simply reloads (the 30s load is one-shot, not per-frame).
                    try:
                        free_flux_cache()
                    except Exception:  # noqa: BLE001
                        pass
                retarget_pose_metas = get_retarget_pose(
                    _pm_for_rt[0], refer_pose_meta, _pm_for_rt,
                    _tpl_edit_meta0, _refer_edit_meta
                )
            except Exception as _rt_exc:  # noqa: BLE001 — fall back to non-retargeted
                logging.getLogger(__name__).warning(
                    "PoseAndFaceDetectionV2: get_retarget_pose failed (%s); using non-retargeted pose.",
                    _rt_exc,
                )
                retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in pose_metas]
        else:
            if use_flux:
                raise ValueError(
                    "use_flux=True requires retarget_image to be connected. FLUX "
                    "normalizes the REFERENCE pose before retargeting, so a "
                    "reference is mandatory. (Mirrors the Wan 2.2 Animate "
                    "preprocessor's own assertion: 'Image editing with FLUX "
                    "can only be used when pose retargeting is enabled'.)")
            retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in pose_metas]

        # ---- force_eyes_open ------------------------------------------------
        # Applied to retarget_pose_metas ONLY. Those become pose_data
        # ["pose_metas"], while pose_data["pose_metas_original"] keeps the
        # pristine detected dicts — so this edit shows up as exactly the kind of
        # landmark DELTA that DrawViTPoseV2.apply_pose_edits_to_face already
        # knows how to execute as a real pixel warp of the face crop. That is
        # the whole mechanism: Wan-Animate never reads landmarks for content
        # (LIA motion encoder reads raw crop pixels), so the landmark edit is
        # only the instruction — DrawViTPoseV2 does the pixel work.
        # Editing `pose_metas` here instead would poison BOTH sides of the
        # comparison and produce a zero delta, i.e. silently no-op.
        if float(force_eyes_open) > 0.0:
            try:
                from .nodes_extras.expression_3d_coeffs import (
                    _read_face_normalised as _eo_read,
                    _write_face_normalised as _eo_write,
                )
                _eo_thresh = float(eye_open_blink_ear)
                _eo_all = (str(eye_open_mode) == "all_frames")
                _eo_hit = 0
                _eo_skipped = 0
                for _fi, _rm in enumerate(retarget_pose_metas):
                    _k = _eo_read(_rm)
                    if _k is None or _k.shape[0] <= max(_LEFT_EYE_IDX):
                        _eo_skipped += 1
                        continue
                    if not _eo_all:
                        _er = _eye_aspect_ratio(_k, _RIGHT_EYE_IDX, W, H)
                        _el = _eye_aspect_ratio(_k, _LEFT_EYE_IDX, W, H)
                        _vals = [v for v in (_er, _el) if np.isfinite(v)]
                        if not _vals or min(_vals) >= _eo_thresh:
                            continue          # eyes already open on this frame
                    _new, _info = force_eye_open_landmarks(
                        _k, W, H, float(force_eyes_open),
                    )
                    if _info.get("changed"):
                        _eo_write(_rm, _new[:, :2])
                        _eo_hit += 1
                logging.getLogger(__name__).info(
                    "PoseAndFaceDetectionV2: force_eyes_open=%.2f (%s) opened %d/%d "
                    "frames%s. Requires DrawViTPoseV2.apply_pose_edits_to_face='warp' "
                    "+ face_images wired, or the edit stays landmark-only and never "
                    "reaches the face encoder.",
                    float(force_eyes_open), str(eye_open_mode), _eo_hit, B,
                    (f" ({_eo_skipped} had no usable face landmarks)" if _eo_skipped else ""),
                )
            except Exception as _eo_exc:                                  # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "force_eyes_open skipped (%s); landmarks left untouched.", _eo_exc,
                )

        # use first bbox for return (legacy)
        bbox0 = bboxes[0]
        bbox = np.array(bbox0).flatten() if bbox0 is not None else np.array([0, 0, 0, 0])
        bbox_ints = tuple(int(v) for v in bbox[:4]) if bbox.shape[0] >= 4 else (0, 0, 0, 0)

        # key frame points (unchanged)
        key_points_index = [0, 1, 2, 5, 8, 11, 10, 13]
        body_key_points = pose_metas[0]['keypoints_body']
        keypoints_body = np.array([body_key_points[i] for i in key_points_index if body_key_points[i] is not None])[:, :2]
        wh = np.array([[pose_metas[0]['width'], pose_metas[0]['height']]])
        points = (keypoints_body * wh).astype(np.int32)
        points_dict_list = [{"x": int(p[0]), "y": int(p[1])} for p in points]

        # --- Iris + gaze estimation ---
        # Preferred path: FaceLandmarker (Tasks API) — returns 478-pt mesh,
        # iris ring, and 52 ARKit blend shapes from which we derive
        # head-pose-corrected per-eye yaw/pitch (radians). Falls back to
        # legacy FaceMesh and finally to the OpenCV pupil voter.
        all_iris = []
        all_lip_ratios = []
        mp_enabled = bool(use_mediapipe_face) and _MP_AVAILABLE
        # Resolve the gaze engine. `use_blendshape_gaze=False` forces the
        # legacy iris-offset path regardless of `gaze_engine`.
        _engine = str(gaze_engine or "blendshape_head_corrected").strip()
        # Honest gaze-engine status: track what the user REQUESTED vs what
        # actually ran, and why it fell back, so the viewer can tell the user
        # (the accurate engines silently degraded before — the "gaze not
        # accurate" complaint). _gaze_note is set at each fallback below.
        _gaze_requested = _engine
        _gaze_note = None
        # gaze_engine DECIDES the engine (fixed 2026-07-31).
        #
        # This used to be:
        #     if not bool(use_blendshape_gaze):
        #         _engine = "legacy_iris_offset"
        # use_blendshape_gaze defaults to False, so whatever the user selected
        # in gaze_engine — ethxgaze, l2cs_gaze360, iris_geometric — was thrown
        # away and replaced by the hand-written pixel-darkness voter, silently.
        # That is why every engine behaved identically and identically badly:
        # none of them was running. The dropdown was decorative.
        #
        # What the real engines actually need is MEDIAPIPE (they read the iris
        # centres from landmarks 468/473, i.e. Google's own iris model), not a
        # legacy checkbox. So gate on that, and say so when falling back.
        _wants_real_engine = _engine not in ("", "legacy_iris_offset")
        if _wants_real_engine and not mp_enabled:
            _gaze_note = (
                f"gaze_engine={_engine!r} needs MediaPipe face landmarks (the "
                f"iris comes from landmarks 468/473) but use_mediapipe_face is "
                f"off or MediaPipe is unavailable, so it fell back to the "
                f"legacy pixel-darkness pupil search. Turn use_mediapipe_face "
                f"ON for the engine you selected to actually run."
            )
            logging.getLogger(__name__).warning(
                "PoseAndFaceDetectionV2: %s", _gaze_note,
            )
            _engine = "legacy_iris_offset"
        elif not _wants_real_engine and bool(use_blendshape_gaze):
            # Legacy workflows that only ticked the old checkbox still get a
            # real engine rather than the voter.
            _engine = "blendshape_head_corrected"
        # C0.4: ethxgaze is a *post-process* over a base engine — the
        # ResNet-50 face-level estimator only runs after iris_data has
        # been populated. We swap to blendshape_head_corrected for the
        # per-frame iris pass and remember to override at the end.
        _ethxgaze_post = (_engine == "ethxgaze")
        if _ethxgaze_post:
            _engine = "blendshape_head_corrected"
        if _engine.startswith("l2cs") and not (
            _GAZE_L2CS_IMPORTED and _gaze_l2cs is not None
        ):
            logging.getLogger(__name__).warning(
                "gaze_engine=%s requested but gaze_l2cs unavailable; "
                "falling back to blendshape_head_corrected.",
                _engine,
            )
            _gaze_note = (f"{_engine} unavailable (install l2cs-net / check weights) "
                          "— using blendshape_head_corrected")
            _engine = "blendshape_head_corrected"
        # The pose_normalized_resnet50 engine requires the normalizer
        # module, the ResNet50 estimator module, AND a usable checkpoint
        # at <ComfyUI>/models/gaze/. If any prerequisite is missing we
        # fall back to plain l2cs_gaze360 (still a strong baseline).
        _pose_norm_resnet50_enabled = (_engine == "pose_normalized_resnet50")
        _pose_norm_model = None
        if _pose_norm_resnet50_enabled:
            _missing_reason = None
            if not (_GAZE_POSE_NORM_IMPORTED and _gaze_pose_norm is not None
                    and _gaze_pose_norm.is_available()):
                _missing_reason = "gaze_pose_norm module / cv2 unavailable"
            elif not (_GAZE_NORM_EST_IMPORTED
                      and _gaze_normalized_estimator is not None
                      and _gaze_normalized_estimator.is_available()):
                _missing_reason = "gaze_normalized_estimator / torch unavailable"
            else:
                try:
                    _pose_norm_model = _gaze_normalized_estimator.get_model()
                except Exception as exc:  # noqa: BLE001
                    _pose_norm_model = None
                    _missing_reason = f"checkpoint load failed: {exc!r}"
                if _pose_norm_model is None and _missing_reason is None:
                    _missing_reason = (
                        "no ResNet50 gaze checkpoint at "
                        "<ComfyUI>/models/gaze/pose_normalized_resnet50.pth.tar"
                    )
            if _missing_reason is not None:
                _fb_engine = "l2cs_gaze360" if (
                    _GAZE_L2CS_IMPORTED and _gaze_l2cs is not None
                ) else "blendshape_head_corrected"
                logging.getLogger(__name__).warning(
                    "gaze_engine=pose_normalized_resnet50 disabled (%s); "
                    "falling back to %s.", _missing_reason, _fb_engine,
                )
                _gaze_note = (f"pose_normalized_resnet50 disabled ({_missing_reason}) "
                              f"— using {_fb_engine}. Drop a ResNet50 gaze checkpoint at "
                              "models/gaze/pose_normalized_resnet50.pth.tar to enable ~3-4°.")
                _pose_norm_resnet50_enabled = False
                _engine = _fb_engine
        if _engine == "iris_geometric" and not (
            _GAZE_IRIS_IMPORTED and _gaze_iris is not None
        ):
            logging.getLogger(__name__).warning(
                "gaze_engine=iris_geometric requested but gaze_iris_geometric "
                "unavailable; falling back to blendshape_head_corrected.",
            )
            _engine = "blendshape_head_corrected"
        if _engine == "blendshape_head_corrected" and not (
            _GAZE_3D_IMPORTED and _gaze_3d is not None
        ):
            logging.getLogger(__name__).warning(
                "gaze_engine=blendshape_head_corrected requested but "
                "gaze_3d unavailable; falling back to blendshape_only.",
            )
            _engine = "blendshape_only"
        _iris_geo_enabled = (_engine == "iris_geometric")
        # iris_geometric keeps the blendshape/FaceLandmarker pass enabled: it
        # supplies R_head for the composition and a blendshape fallback for
        # blink-gated frames. The measured-iris source simply outranks it.
        bs_enabled = (_engine in ("blendshape_head_corrected", "blendshape_only",
                                  "iris_geometric")
                      and _GAZE_BS_IMPORTED
                      and _gaze_bs is not None and _gaze_bs.is_available())
        l2cs_enabled = _engine.startswith("l2cs")
        head_correct = (_engine in ("blendshape_head_corrected", "iris_geometric")
                        and _GAZE_3D_IMPORTED and _gaze_3d is not None)
        max_yaw_rad = math.radians(float(gaze_max_yaw_deg))
        max_pitch_rad = math.radians(float(gaze_max_pitch_deg))
        # Lazy-init L2CS pipeline and per-eye Kalman filters once per node call.
        _l2cs_pipeline = None
        if l2cs_enabled:
            try:
                _variant = "gaze360" if _engine == "l2cs_gaze360" else "mpiigaze"
                _l2cs_pipeline = _gaze_l2cs.get_pipeline(variant=_variant)
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "L2CS pipeline init failed (%s); falling back to "
                    "blendshape_head_corrected.", exc,
                )
                l2cs_enabled = False
                head_correct = True
                bs_enabled = (_GAZE_BS_IMPORTED and _gaze_bs is not None
                              and _gaze_bs.is_available())
        _kalman_dt = 1.0 / max(float(gaze_fps), 1.0)
        _kalman_meas = math.radians(float(gaze_kalman_meas_std_deg))
        _kalman_proc = float(gaze_kalman_process_std)
        _kalman = None
        if _GAZE_3D_IMPORTED and _gaze_3d is not None and (head_correct or l2cs_enabled):
            _kalman = {
                'right': _gaze_3d.AngleKalman2D(
                    dt=_kalman_dt, process_std=_kalman_proc, meas_std=_kalman_meas,
                ),
                'left': _gaze_3d.AngleKalman2D(
                    dt=_kalman_dt, process_std=_kalman_proc, meas_std=_kalman_meas,
                ),
            }
        # ── W7-G2: per-shot gaze calibration pre-pass (iris_geometric only) ──
        # Measure the eye-in-head angles on the user-designated "subject looks
        # straight at the camera" frame and make them the zero reference for
        # the whole shot. Runs BEFORE the main loop so every frame (including
        # ones earlier than the calibration frame) gets the corrected origin.
        if _iris_geo_enabled:
            _gaze_iris.reset_calibration()   # never leak a previous run's offsets
            _cal_idx = int(gaze_calibration_frame)
            if 0 <= _cal_idx < B:
                try:
                    rx1, rx2, ry1, ry2 = raw_face_bboxes[_cal_idx]
                    _side = max(rx2 - rx1, ry2 - ry1) * 1.4
                    _ccx, _ccy = 0.5 * (rx1 + rx2), 0.5 * (ry1 + ry2)
                    _h = 0.5 * _side
                    _mx1 = int(max(0, round(_ccx - _h)))
                    _my1 = int(max(0, round(_ccy - _h)))
                    _mx2 = int(min(W, round(_ccx + _h)))
                    _my2 = int(min(H, round(_ccy + _h)))
                    _cal_res = None
                    if _mx2 - _mx1 > 8 and _my2 - _my1 > 8:
                        _cal_crop = (np.clip(images_np[_cal_idx][_my1:_my2, _mx1:_mx2], 0, 1) * 255).astype(np.uint8)
                        if bs_enabled:
                            _cal_res = _run_face_landmarker_on_face_crop(
                                _cal_crop, (_mx1, _my1), (_mx2 - _mx1, _my2 - _my1), W, H)
                        if _cal_res is None:
                            _cal_res = _run_mediapipe_on_face_crop(
                                _cal_crop, (_mx1, _my1), (_mx2 - _mx1, _my2 - _my1), W, H)
                    if _cal_res is not None:
                        _gaze_iris.calibrate_from_measurement(
                            _gaze_iris.eye_in_head_from_iris(_cal_res, "right"),
                            _gaze_iris.eye_in_head_from_iris(_cal_res, "left"),
                        )
                        logging.getLogger(__name__).info(
                            "[iris_geometric] calibrated on frame %d: %s",
                            _cal_idx, _gaze_iris.get_calibration())
                    else:
                        logging.getLogger(__name__).warning(
                            "[iris_geometric] calibration frame %d: no face found; "
                            "calibration skipped (gaze stays uncalibrated).", _cal_idx)
                except Exception as exc:  # noqa: BLE001
                    logging.getLogger(__name__).warning(
                        "[iris_geometric] calibration pre-pass failed: %s", exc)
        mp_used_count = 0
        bs_used_count = 0
        l2cs_used_count = 0
        for idx, meta in _IC.track(
            list(enumerate(pose_metas)), len(pose_metas),
            "WanAnimateV2: face/gaze per-frame",
        ):
            mp_result = None
            # IMPORTANT: MediaPipe FaceLandmarker expects the ENTIRE face
            # in its input. The YOLO/raw face bbox is often tight (eyes
            # outside the box), which makes the model produce landmarks
            # in wrong locations. We pad the *landmark* crop here, while
            # keeping `face_bboxes[idx]` as the user-configured output
            # crop (which downstream consumers and the cyan FACE rect use).
            rx1, rx2, ry1, ry2 = raw_face_bboxes[idx]
            rw, rh = rx2 - rx1, ry2 - ry1
            # Square + 40% padding around the raw face box so eyes /
            # forehead / chin always fit, then clamp to image bounds.
            side = max(rw, rh) * 1.4
            cx_r = 0.5 * (rx1 + rx2)
            cy_r = 0.5 * (ry1 + ry2)
            half = 0.5 * side
            mx1 = int(max(0, round(cx_r - half)))
            my1 = int(max(0, round(cy_r - half)))
            mx2 = int(min(W, round(cx_r + half)))
            my2 = int(min(H, round(cy_r + half)))
            mcw, mch = mx2 - mx1, my2 - my1
            crop_rgb = None
            if mcw > 8 and mch > 8:
                crop_rgb = (np.clip(images_np[idx][my1:my2, mx1:mx2], 0, 1) * 255).astype(np.uint8)
            # Output-crop coords (kept for parity with legacy code paths
            # that still expected `x1..y2` in scope below).
            x1, x2, y1, y2 = face_bboxes[idx]
            cw, ch = x2 - x1, y2 - y1

            # 1) Try FaceLandmarker Tasks API (with blend-shape gaze)
            if bs_enabled and crop_rgb is not None:
                mp_result = _run_face_landmarker_on_face_crop(
                    crop_rgb, (mx1, my1), (mcw, mch), W, H,
                )
                if mp_result is not None:
                    bs_used_count += 1
            # 2) Fall back to legacy FaceMesh (no blend shapes)
            if mp_result is None and mp_enabled and crop_rgb is not None:
                mp_result = _run_mediapipe_on_face_crop(
                    crop_rgb, (mx1, my1), (mcw, mch), W, H,
                )

            if mp_result is not None:
                mp_used_count += 1
                # Override face_kps[1:69] with MediaPipe-derived 68 landmarks
                # (face_kps[0] is the body-anchored face anchor from ViTPose;
                # leave it intact so Wan's pose encoder keeps its global hook).
                # Defensive guard: every upstream MediaPipe path *should*
                # populate 'kps68_norm' but a partial dict (gaze-only,
                # iris-only, mocked) would crash the shape assignment with
                # a confusing TypeError. Skip the override in that case;
                # downstream still gets the ViTPose-derived face_kps as
                # a fallback.
                face_kps = meta['keypoints_face']
                _kps68 = mp_result.get('kps68_norm') if isinstance(mp_result, dict) else None
                if (
                    _kps68 is not None
                    and face_kps.shape[0] >= 69
                    and getattr(_kps68, 'shape', None) == (68, 3)
                ):
                    face_kps[1:69, :] = _kps68
                    meta['keypoints_face'] = face_kps

                rix, riy = mp_result['right_iris_px']
                lix, liy = mp_result['left_iris_px']
                iris_result = {
                    'right_iris': {'x': rix, 'y': riy, 'confidence': 1.0,
                                    'radius': mp_result['right_iris_radius_px']},
                    'left_iris':  {'x': lix, 'y': liy, 'confidence': 1.0,
                                    'radius': mp_result['left_iris_radius_px']},
                    'source': mp_result.get('source', 'face_mesh'),
                }
                # Optional L2CS-Net per-frame inference (Stage 2). Runs
                # on the FULL ORIGINAL FRAME (L2CS's RetinaFace detector
                # needs context; tight crops cause profile-head misses)
                # and is routed to the matching face via face_bbox.
                # Produces a single world-frame (yaw, pitch). Both eyes
                # share that estimate. Falls silently back to blendshape
                # on per-frame failure; the outer loop keeps the last
                # good gaze if EVERY engine misses on this frame.
                _l2cs_per_eye = None
                if (l2cs_enabled and _l2cs_pipeline is not None) or \
                        (_pose_norm_resnet50_enabled and _pose_norm_model is not None):
                    try:
                        full_rgb_u8 = (np.clip(images_np[idx], 0, 1) * 255).astype(np.uint8)
                        _fb_xyxy = (
                            int(face_bboxes[idx][0]), int(face_bboxes[idx][2]),
                            int(face_bboxes[idx][1]), int(face_bboxes[idx][3]),
                        )
                        # Two inference paths:
                        #   * pose_normalized_resnet50: clean-room pose-
                        #     normalized 224x224 warp (removes head roll +
                        #     fixes distance to 600 mm) -> ResNet50+Linear(2,)
                        #     gaze head -> de-rotate via R_norm.T back to
                        #     the original camera frame.
                        #   * l2cs_*: full-frame RetinaFace + L2CS network.
                        _l2cs_out = None
                        if _pose_norm_resnet50_enabled and _pose_norm_model is not None:
                            _norm_obj = None
                            try:
                                _lm_full = mp_result.get('landmarks_px_full')
                                if _lm_full is None:
                                    _lm_full = mp_result.get('landmarks_norm_full')
                                if _lm_full is not None:
                                    _norm_obj = _gaze_pose_norm.normalize_face_for_gaze(
                                        full_rgb_u8, _lm_full, image_size=(W, H),
                                    )
                            except Exception as exc:  # noqa: BLE001
                                logging.getLogger(__name__).debug(
                                    "[PoseNorm] normalize failed: %s", exc)
                                _norm_obj = None
                            if _norm_obj is not None:
                                _raw = _gaze_normalized_estimator.infer_normalized(
                                    _pose_norm_model, _norm_obj.image,
                                )
                                if _raw is not None:
                                    _yn, _pn, _cn = _raw
                                    _yc, _pc = _gaze_pose_norm.denormalize_gaze(
                                        _yn, _pn, _norm_obj.R_norm,
                                    )
                                    _l2cs_out = (_yc, _pc, _cn)
                        elif l2cs_enabled and _l2cs_pipeline is not None:
                            _l2cs_out = _gaze_l2cs.infer_frame(
                                _l2cs_pipeline, full_rgb_u8, face_bbox=_fb_xyxy,
                            )
                    except Exception as exc:  # noqa: BLE001
                        logging.getLogger(__name__).debug(
                            "[gaze] external regressor inference failed: %s", exc)
                        _l2cs_out = None
                    if _l2cs_out is not None:
                        _y_l, _p_l, _c_l = _l2cs_out
                        _l2cs_per_eye = {
                            'right': {'yaw_rad': _y_l, 'pitch_rad': _p_l,
                                       'blink': 0.0, 'confidence': _c_l},
                            'left':  {'yaw_rad': _y_l, 'pitch_rad': _p_l,
                                       'blink': 0.0, 'confidence': _c_l},
                        }
                        l2cs_used_count += 1
                gaze_bs = mp_result.get('gaze_blendshape')
                if gaze_bs is not None or _l2cs_per_eye is not None or _iris_geo_enabled:
                    # Blend-shape path: rescale yaw/pitch to user-tuned max
                    # angles (defaults already factor in MAX_GAZE_*_RAD,
                    # so divide by them and remultiply by the new max).
                    base_yaw = _gaze_bs.MAX_GAZE_YAW_RAD if _gaze_bs else 1.0
                    base_pitch = _gaze_bs.MAX_GAZE_PITCH_RAD if _gaze_bs else 1.0
                    # Gaze-arrow screen-space convention (FIX, May 2026):
                    # The renderer in draw_debug_overlay() draws the arrow as
                    #   ex = ix + dx*L;  ey = iy + dy*L
                    # so dx/dy MUST be in image-pixel direction (right=+x,
                    # down=+y). The legacy 2D-offset path correctly derives
                    # dx/dy from the iris pixel offset to the eye centroid;
                    # we replicate the same screen-space derivation here so
                    # both gaze paths agree. Synthesising dx from
                    # ``-sin(yaw_rad)`` (anatomical-camera convention) made
                    # the arrow point opposite to the iris in the rendered
                    # debug view — that was the user-reported bug.
                    # Build per-eye reference frame from MediaPipe's own
                    # eye corner landmarks (same model that produced the
                    # iris pixels — no cross-model misalignment, no
                    # eyelid asymmetry bias). Center = midpoint of the
                    # outer+inner corner. Half-width = horizontal eye
                    # span / 2 (used as magnitude normalizer so dx ∈
                    # [-1, +1] corresponds to iris from outer to inner
                    # corner — a physically meaningful range).
                    r_out_x, r_out_y = mp_result['right_eye_outer_px']
                    r_in_x,  r_in_y  = mp_result['right_eye_inner_px']
                    l_in_x,  l_in_y  = mp_result['left_eye_inner_px']
                    l_out_x, l_out_y = mp_result['left_eye_outer_px']
                    eye_ref = {
                        'right': {
                            'cx': 0.5 * (r_out_x + r_in_x),
                            'cy': 0.5 * (r_out_y + r_in_y),
                            'hw': max(0.5 * abs(r_in_x - r_out_x), 1.0),
                        },
                        'left': {
                            'cx': 0.5 * (l_in_x + l_out_x),
                            'cy': 0.5 * (l_in_y + l_out_y),
                            'hw': max(0.5 * abs(l_out_x - l_in_x), 1.0),
                        },
                    }
                    iris_pix = {
                        'right': (float(rix), float(riy)),
                        'left':  (float(lix), float(liy)),
                    }
                    for eye_name in ('right', 'left'):
                        if _l2cs_per_eye is not None:
                            # L2CS path: yaw/pitch already in CAMERA frame
                            # (radians, our sign convention). Skip world
                            # bridge entirely; render dx/dy via L2CS's own
                            # direct formula (matches l2cs.utils.draw_gaze).
                            e = dict(_l2cs_per_eye[eye_name])
                            e['source'] = 'l2cs_' + (
                                "gaze360" if _engine == "l2cs_gaze360" else "mpiigaze"
                            )
                        elif _iris_geo_enabled:
                            # W7-G1: MEASURED iris-in-aperture eye-in-head gaze.
                            # Outranks the blendshape estimate; blink-gated
                            # frames fall back to blendshapes (if available)
                            # so the Kalman stream never starves.
                            _ge = None
                            try:
                                _ge = _gaze_iris.eye_in_head_from_iris(mp_result, eye_name)
                            except Exception:  # noqa: BLE001
                                _ge = None
                            if _ge is not None:
                                e = _ge
                                e['source'] = 'iris_geometric'
                            elif gaze_bs is not None:
                                e = dict(gaze_bs[eye_name])
                                e['yaw_rad'] = float(e['yaw_rad']) / max(base_yaw, 1e-6) * max_yaw_rad
                                e['pitch_rad'] = float(e['pitch_rad']) / max(base_pitch, 1e-6) * max_pitch_rad
                                e['source'] = 'blendshape_blink_fallback'
                            else:
                                # Blink/occlusion with no fallback: hold neutral;
                                # the Kalman velocity model coasts through it.
                                e = {'yaw_rad': 0.0, 'pitch_rad': 0.0,
                                     'blink': 1.0, 'confidence': 0.0,
                                     'source': 'iris_geometric_gated'}
                        else:
                            e = dict(gaze_bs[eye_name])
                            e['yaw_rad'] = float(e['yaw_rad']) / max(base_yaw, 1e-6) * max_yaw_rad
                            e['pitch_rad'] = float(e['pitch_rad']) / max(base_pitch, 1e-6) * max_pitch_rad
                            e['source'] = 'blendshape'
                        _yaw = float(e['yaw_rad'])
                        _pitch = float(e['pitch_rad'])
                        if _l2cs_per_eye is not None:
                            # --- L2CS branch -------------------------------
                            # Kalman in CAMERA-frame (yaw, pitch) directly.
                            if _kalman is not None:
                                _yaw, _pitch = _kalman[eye_name].step(_yaw, _pitch)
                            e['yaw_rad'] = float(_yaw)
                            e['pitch_rad'] = float(_pitch)
                            if _GAZE_3D_IMPORTED and _gaze_3d is not None:
                                _dx, _dy = _gaze_3d.screen_dx_dy_from_camera_yaw_pitch(
                                    _yaw, _pitch,
                                )
                            else:
                                _dx = -math.sin(_yaw) * math.cos(_pitch)
                                _dy = -math.sin(_pitch)
                        elif head_correct and _GAZE_3D_IMPORTED and _gaze_3d is not None:
                            # --- Blendshape + head-pose composition --------
                            R_head_frame = mp_result.get('R_head')
                            _dx, _dy, _yaw_w, _pitch_w = _gaze_3d.world_gaze_from_eye_in_head(
                                _yaw, _pitch, R_head_frame,
                            )
                            if _kalman is not None:
                                _yaw_w, _pitch_w = _kalman[eye_name].step(_yaw_w, _pitch_w)
                                _dx2, _dy2, _, _ = _gaze_3d.world_gaze_from_eye_in_head(
                                    _yaw_w, _pitch_w, None,
                                )
                                _dx, _dy = _dx2, _dy2
                            e['yaw_rad'] = float(_yaw_w)
                            e['pitch_rad'] = float(_pitch_w)
                            e['source'] = ('iris_geometric_head_corrected'
                                           if _iris_geo_enabled and
                                           str(e.get('source', '')).startswith('iris_geometric')
                                           else 'blendshape_head_corrected')
                            _yaw = float(_yaw_w)
                            _pitch = float(_pitch_w)
                        else:
                            # --- Blendshape-only (no head correction) ------
                            _dx = -math.sin(_yaw) * math.cos(_pitch)
                            _dy = -math.sin(_pitch)
                        _mag = math.hypot(_dx, _dy)
                        # Magnitude proxy: how far yaw/pitch are from
                        # neutral, normalized to user-configured maxima.
                        mag_norm = float(min(1.0, math.hypot(
                            _yaw / max(max_yaw_rad, 1e-6),
                            _pitch / max(max_pitch_rad, 1e-6),
                        )))
                        # Dead-zone: below ~10% of max, treat as
                        # forward-gaze (no arrow).
                        if mag_norm < 0.10 or _mag < 1e-6:
                            e['dx'] = 0.0
                            e['dy'] = 0.0
                            e['magnitude_norm'] = 0.0
                        else:
                            # dx/dy CARRY the magnitude (fixed 2026-07-31).
                            # These used to be a bare unit vector: for a
                            # subject looking near the camera _dx and _dy are a
                            # fraction of a degree each, so dividing by their
                            # own length discarded the gaze and kept a
                            # direction made almost entirely of estimator
                            # noise, stretched to full length. Scaling by
                            # mag_norm makes the vector shrink to nothing as
                            # the gaze approaches the camera axis, which is the
                            # behaviour every consumer wants.
                            e['dx'] = round(_dx / _mag * mag_norm, 4)
                            e['dy'] = round(_dy / _mag * mag_norm, 4)
                            e['magnitude_norm'] = mag_norm
                        # Keep the pixel-space refs available for the
                        # diagnostic log below.
                        ipx, ipy = iris_pix[eye_name]
                        ref = eye_ref[eye_name]
                        ddx = ipx - ref['cx']
                        ddy = ipy - ref['cy']
                        # Diagnostic log for the gaze direction
                        # investigation (user memory
                        # `comfyui_workflow_rules.md`, May 2026).
                        # By default writes the first frame only (cheap
                        # one-shot sanity check).  Set the env var
                        # `MEC_GAZE_DEBUG_ALL_FRAMES=1` to dump every
                        # frame — useful when chasing a per-frame
                        # direction artefact.  The hard-coded "frame=0"
                        # tag is replaced with the real `idx` so multi-
                        # frame logs are readable.
                        import os as _os
                        _all_frames = _os.environ.get(
                            "MEC_GAZE_DEBUG_ALL_FRAMES", ""
                        ) not in ("", "0", "false", "False")
                        if idx == 0 or _all_frames:
                            try:
                                _dbg_path = _os.path.join(
                                    _os.path.dirname(__file__),
                                    "_gaze_debug.log",
                                )
                                # Also dump raw blendshape coefficients
                                # for this frame so we can compare the
                                # corner-midpoint result against the
                                # MediaPipe-calibrated eye-look signals.
                                _bs_raw = mp_result.get('blendshapes', {}) or {}
                                if eye_name == 'right':
                                    _bs_keys = ('eyeLookInRight','eyeLookOutRight','eyeLookUpRight','eyeLookDownRight','eyeBlinkRight')
                                else:
                                    _bs_keys = ('eyeLookInLeft','eyeLookOutLeft','eyeLookUpLeft','eyeLookDownLeft','eyeBlinkLeft')
                                _bs_str = ' '.join(f"{k}={float(_bs_raw.get(k,0.0)):.3f}" for k in _bs_keys)
                                with open(_dbg_path, "a", encoding="utf-8") as _fh:
                                    _fh.write(
                                        f"frame={idx} eye={eye_name} "
                                        f"iris_px=({ipx:.1f},{ipy:.1f}) "
                                        f"eye_corner_mid=({ref['cx']:.1f},{ref['cy']:.1f}) "
                                        f"hw={ref['hw']:.1f} "
                                        f"ddx={ddx:+.2f} ddy={ddy:+.2f} "
                                        f"mag_norm={e.get('magnitude_norm',0.0):.3f} "
                                        f"unit=({e['dx']:+.4f},{e['dy']:+.4f}) "
                                        f"yaw={float(e.get('yaw_rad',0.0)):+.4f} "
                                        f"pitch={float(e.get('pitch_rad',0.0)):+.4f} "
                                        f"bs[{_bs_str}] "
                                        f"W={W} H={H}\n"
                                    )
                            except Exception:
                                pass
                        e['magnitude'] = float(math.hypot(e['yaw_rad'], e['pitch_rad']))
                        iris_result[f'{eye_name}_gaze'] = e
                        # Eye-socket reference frame for the 3D→2D iris
                        # projection in the gaze-repaint step (C0.1).
                        iris_result[f'{eye_name}_eye_ref'] = {
                            'cx': float(ref['cx']), 'cy': float(ref['cy']),
                            'hw': float(ref['hw']),
                        }
                    iris_result['blendshapes'] = mp_result.get('blendshapes', {})
                else:
                    # Legacy fallback: 2D iris-offset gaze (kept for
                    # backward compatibility when blend shapes are off).
                    # Uses the same MediaPipe eye-corner-midpoint reference
                    # frame as the blendshape branch above — eliminates the
                    # eyelid-asymmetry bias that the old dlib-eye-contour
                    # centroid suffered from (lower-lid lands wider than
                    # upper, pushing the centroid 2 px down regardless of
                    # true gaze direction).
                    r_out_x, r_out_y = mp_result['right_eye_outer_px']
                    r_in_x,  r_in_y  = mp_result['right_eye_inner_px']
                    l_in_x,  l_in_y  = mp_result['left_eye_inner_px']
                    l_out_x, l_out_y = mp_result['left_eye_outer_px']
                    eye_ref = {
                        'right': {
                            'cx': 0.5 * (r_out_x + r_in_x),
                            'cy': 0.5 * (r_out_y + r_in_y),
                            'hw': max(0.5 * abs(r_in_x - r_out_x), 1.0),
                        },
                        'left': {
                            'cx': 0.5 * (l_in_x + l_out_x),
                            'cy': 0.5 * (l_in_y + l_out_y),
                            'hw': max(0.5 * abs(l_out_x - l_in_x), 1.0),
                        },
                    }
                    for eye_name, iris_xy in (
                        ('right', (float(rix), float(riy))),
                        ('left',  (float(lix), float(liy))),
                    ):
                        ref = eye_ref[eye_name]
                        ddx = iris_xy[0] - ref['cx']
                        ddy = iris_xy[1] - ref['cy']
                        nrm = float(math.hypot(ddx, ddy))
                        mag_norm = float(min(1.0, nrm / ref['hw']))
                        if mag_norm < 0.15:
                            iris_result[f'{eye_name}_gaze'] = {
                                'dx': 0.0, 'dy': 0.0,
                                'magnitude_norm': 0.0,
                                'yaw_rad': 0.0, 'pitch_rad': 0.0,
                                'source': 'iris_offset_2d',
                            }
                        else:
                            iris_result[f'{eye_name}_gaze'] = {
                                'dx': round(ddx / nrm, 4),
                                'dy': round(ddy / nrm, 4),
                                'magnitude_norm': mag_norm,
                                'yaw_rad': 0.0,
                                'pitch_rad': 0.0,
                                'source': 'iris_offset_2d',
                            }
                        # Eye-socket reference frame for the 3D→2D iris
                        # projection in the gaze-repaint step (C0.1).
                        iris_result[f'{eye_name}_eye_ref'] = {
                            'cx': float(ref['cx']), 'cy': float(ref['cy']),
                            'hw': float(ref['hw']),
                        }
                all_iris.append(iris_result)
                all_lip_ratios.append(float(mp_result['lip_openness_ratio']))
            else:
                # Fallback: legacy ViTPose + image-based pupil voter
                iris_result = estimate_iris_positions(
                    meta['keypoints_face'], images_np[idx], W, H,
                )
                iris_result['source'] = 'pupil_voter'
                all_iris.append(iris_result)
                all_lip_ratios.append(0.0)

        if mp_enabled or bs_enabled:
            logging.getLogger(__name__).info(
                "Face mesh: %d/%d frames (%.1f%%); blend-shape gaze: %d/%d frames (%.1f%%)",
                mp_used_count, B, 100.0 * mp_used_count / max(B, 1),
                bs_used_count, B, 100.0 * bs_used_count / max(B, 1),
            )

        # --- Temporal smoothing ---
        # Iris pixel positions: choose method (one_euro recommended).
        if use_iris_smoothing and len(all_iris) > 1 and iris_smoothing_method != "none":
            if iris_smoothing_method == "one_euro":
                # Per-eye, per-axis One-Euro filter on the iris pixel
                # coordinates. Far better than EMA at separating jitter
                # from real saccades — exactly what the Wan 2.2 face
                # encoder needs to reproduce stable gaze.
                try:
                    from .gaze_blendshape import OneEuroFilter as _OEF
                    fps_est = 30.0
                    filt_kw = dict(
                        freq=fps_est,
                        min_cutoff=float(iris_one_euro_min_cutoff),
                        beta=float(iris_one_euro_beta),
                    )
                    filters = {
                        ("right_iris", "x"): _OEF(**filt_kw),
                        ("right_iris", "y"): _OEF(**filt_kw),
                        ("left_iris",  "x"): _OEF(**filt_kw),
                        ("left_iris",  "y"): _OEF(**filt_kw),
                    }
                    for fr in all_iris:
                        for eye_key in ("right_iris", "left_iris"):
                            iris = fr.get(eye_key)
                            if not isinstance(iris, dict):
                                continue
                            # Skip blink frames so the filter doesn't drift.
                            if float(iris.get("confidence", 1.0)) < 0.05:
                                continue
                            iris["x"] = filters[(eye_key, "x")](float(iris["x"]))
                            iris["y"] = filters[(eye_key, "y")](float(iris["y"]))
                except Exception as _exc:
                    logging.getLogger(__name__).warning(
                        "Iris one-euro smoothing failed (%s); falling back to EMA.", _exc,
                    )
                    iris_smoothing_method = "ema"

            if iris_smoothing_method == "ema":
                strength = float(np.clip(iris_smoothing_strength, 0.0, 1.0))
                for eye_key in ('right_iris', 'left_iris'):
                    prev_x = all_iris[0][eye_key]['x']
                    prev_y = all_iris[0][eye_key]['y']
                    for i in range(1, len(all_iris)):
                        cur = all_iris[i][eye_key]
                        alpha = 1.0 - strength
                        cur['x'] = alpha * cur['x'] + strength * prev_x
                        cur['y'] = alpha * cur['y'] + strength * prev_y
                        prev_x, prev_y = cur['x'], cur['y']

        # Gaze yaw/pitch: one-euro filter per eye (low-lag, kills jitter).
        if bs_used_count > 0 and _GAZE_BS_IMPORTED and _gaze_bs is not None:
            try:
                fps_est = 30.0
                smoother = _gaze_bs.GazeStreamSmoother(
                    fps=fps_est,
                    min_cutoff=float(gaze_one_euro_min_cutoff),
                    beta=float(gaze_one_euro_beta),
                )
                for fr in all_iris:
                    rg = fr.get('right_gaze')
                    lg = fr.get('left_gaze')
                    if not (isinstance(rg, dict) and isinstance(lg, dict)
                            and 'yaw_rad' in rg and 'yaw_rad' in lg):
                        continue
                    smoothed = smoother.step({
                        'left':  {'yaw_rad': float(lg['yaw_rad']),
                                  'pitch_rad': float(lg['pitch_rad'])},
                        'right': {'yaw_rad': float(rg['yaw_rad']),
                                  'pitch_rad': float(rg['pitch_rad'])},
                    })
                    for side, key in (('right', 'right_gaze'), ('left', 'left_gaze')):
                        e = fr[key]
                        e['yaw_rad'] = smoothed[side]['yaw_rad']
                        e['pitch_rad'] = smoothed[side]['pitch_rad']
                        # Dead-code cleanup (May 2026): the smoother does
                        # NOT touch dx/dy (see gaze_blendshape.step), so
                        # reading smoothed[side]['dx'] used to KeyError
                        # and silently fall into the outer except. Keep
                        # dx/dy as set by the iris-pixel-offset path.
                        e['magnitude'] = smoothed[side]['magnitude']
            except Exception as _exc:
                logging.getLogger(__name__).warning(
                    "Gaze one-euro smoothing failed (%s); using raw values.", _exc,
                )

        # --- Cross-eye gaze locking (NEW) ---------------------------------
        # The single most common Wan-Animate artefact is the two eyes
        # pointing in slightly different directions. Per-eye OneEuro
        # smoothing leaves them independent; here we pull each eye toward
        # the per-frame average (yaw, pitch) of the two eyes.
        # Re-derive dx/dy/magnitude after the blend so debug arrows match.
        if gaze_lock_eyes and len(all_iris) > 0:
            lock = float(np.clip(gaze_lock_strength, 0.0, 1.0))
            if lock > 0.0:
                for fr in all_iris:
                    rg = fr.get('right_gaze')
                    lg = fr.get('left_gaze')
                    if not (isinstance(rg, dict) and isinstance(lg, dict)
                            and 'yaw_rad' in rg and 'yaw_rad' in lg):
                        continue
                    avg_yaw = 0.5 * (float(rg['yaw_rad']) + float(lg['yaw_rad']))
                    avg_pitch = 0.5 * (float(rg['pitch_rad']) + float(lg['pitch_rad']))
                    # Bug-fix (re-examined): the May-2026 fix froze dx/dy at
                    # their PRE-lock values because an earlier attempt to
                    # regenerate them from the blended yaw/pitch used the
                    # wrong formula (bare `-sin(blended_yaw)`, missing the
                    # `cos(pitch)` term) and produced arrows pointing opposite
                    # to the iris. That masked the real bug instead of fixing
                    # it: every eligible engine here (guarded by 'yaw_rad' in
                    # rg/lg — L2CS / blendshape / iris_geometric-head-corrected)
                    # ALREADY derives dx/dy from yaw/pitch via the exact same
                    # `screen_dx_dy_from_camera_yaw_pitch` used a few lines
                    # above, BEFORE locking. Freezing dx/dy here means the
                    # debug arrow (and anything else reading dx/dy) shows the
                    # UNLOCKED per-eye direction while yaw_rad/pitch_rad — the
                    # values that actually drive Wan-Animate's conditioning —
                    # are the LOCKED/blended ones. At the default
                    # gaze_lock_strength=0.7 this is a real, consistent
                    # mismatch between what the user sees and what the model
                    # receives ("the gaze arrow doesn't match"). Fix: reuse the
                    # SAME vetted conversion so dx/dy/magnitude_norm track the
                    # locked yaw/pitch exactly like they tracked the raw
                    # per-eye values before locking.
                    for e in (rg, lg):
                        e['yaw_rad']   = (1.0 - lock) * float(e['yaw_rad'])   + lock * avg_yaw
                        e['pitch_rad'] = (1.0 - lock) * float(e['pitch_rad']) + lock * avg_pitch
                        e['magnitude'] = float(math.hypot(e['yaw_rad'], e['pitch_rad']))
                        _ly, _lp = e['yaw_rad'], e['pitch_rad']
                        if _GAZE_3D_IMPORTED and _gaze_3d is not None:
                            _ldx, _ldy = _gaze_3d.screen_dx_dy_from_camera_yaw_pitch(_ly, _lp)
                        else:
                            _ldx = -math.sin(_ly) * math.cos(_lp)
                            _ldy = -math.sin(_lp)
                        _lmag_norm = float(min(1.0, math.hypot(
                            _ly / max(max_yaw_rad, 1e-6), _lp / max(max_pitch_rad, 1e-6),
                        )))
                        _lmag = math.hypot(_ldx, _ldy)
                        if _lmag_norm < 0.10 or _lmag < 1e-6:
                            e['dx'] = 0.0
                            e['dy'] = 0.0
                            e['magnitude_norm'] = 0.0
                        else:
                            # Same magnitude-carrying convention as the
                            # engine block above. gaze_lock_eyes defaults ON,
                            # so this block runs LAST and its unit-normalised
                            # dx/dy overwrote every engine's correct output -
                            # which is why the arrow was wrong in all modes.
                            e['dx'] = round(_ldx / _lmag * _lmag_norm, 4)
                            e['dy'] = round(_ldy / _lmag * _lmag_norm, 4)
                            e['magnitude_norm'] = _lmag_norm

        _src_sides = [
            max(x2 - x1, y2 - y1) for x1, x2, y1, y2 in face_bboxes
        ] if face_bboxes else []
        _median_src = float(np.median(_src_sides)) if _src_sides else 0.0
        _face_q = {
            "tile_px": (
                [int(face_bboxes[0][1] - face_bboxes[0][0]),
                 int(face_bboxes[0][3] - face_bboxes[0][2])]
                if face_bboxes else [0, 0]
            ),
            "median_src_side": int(round(_median_src)),
            "upscale_to_512": round(512.0 / max(1.0, _median_src), 2),
            "floor_applied": int(_floor_raised),
            "floor_min_px": int(_SOURCE_FACE_MIN_PX),
            "pixels_unmodified": (
                str(apply_gaze_to_face_image or "off").strip().lower() == "off"
                and float(au_amplify) <= 1.001
            ),
        }

        # Build per-frame iris output
        iris_output = []
        for idx, iris in enumerate(all_iris):
            iris_output.append({
                'frame': idx,
                'right_iris': iris.get('right_iris'),
                'left_iris': iris.get('left_iris'),
                'right_gaze': iris.get('right_gaze'),
                'left_gaze': iris.get('left_gaze'),
                'lip_openness_ratio': all_lip_ratios[idx] if idx < len(all_lip_ratios) else 0.0,
            })

        pose_data = {
            "pose_metas": retarget_pose_metas,
            "pose_metas_original": pose_metas,
            # V1-parity retarget payload — present only when retarget_image was
            # connected; lets the draw node pad/resize onto the reference.
            "refer_pose_meta": refer_pose_meta if retarget_image is not None else None,
            "retarget_image": refer_img_proc if retarget_image is not None else None,
            # use_flux (enhanced retargeting) — True when FLUX.1-Kontext-dev
            # normalized the reference + first template frame before retargeting.
            # Downstream nodes can read this to know the retarget started from
            # a FLUX-normalized neutral pose rather than the raw reference.
            "use_flux": bool(use_flux) and retarget_image is not None,
            "iris_data": all_iris,
            "lip_openness_ratios": all_lip_ratios,
            # MANUAL bug-fix (Apr 2026): expose source frame dims + target
            # render dims so DrawViTPoseV2 can map iris pixel coords (which
            # live in the *original* frame coord system) into the retargeted
            # canvas using the same padding_resize transform that body
            # keypoints went through.
            "source_size": (int(H), int(W)),
            "target_size": (int(height), int(width)),
            # Expression-edit delivery (2026-07-24): the per-frame face-crop
            # boxes (x1,x2,y1,y2 in source-frame pixels) that produced
            # face_images. DrawViTPoseV2 needs them to map FC3D's edited
            # landmarks (full-frame normalised) into face-crop space so the
            # edits can be WARPED into the actual face pixels the Wan-Animate
            # face encoder sees — without this, landmark edits only move the
            # skeleton dots and the model keeps following the unedited crop.
            "face_crop_boxes": [list(map(int, bb)) for bb in face_bboxes],
            # Preprocessor quality for the 512 tile Wan 2.2 Animate 14B
            # actually encodes. Not a texture-copy guarantee — the Face
            # Adapter still compresses to motion_dim=20 — but this is the
            # honest report of what we fed it.
            "face_images_quality": _face_q,
        }

        # C0.4: ETH-XGaze post-process override.
        # Replace iris_data[*]['left_gaze'/'right_gaze'] with predictions
        # from the ETH-XGaze ResNet-50 model. Requires the third_party
        # repo + checkpoint; on any failure we keep the original engine's
        # output and emit a warning.
        _ethxgaze_ok = False
        if _ethxgaze_post:
            try:
                from .nodes_extras.gaze_ethxgaze import WanGazeETHXGazeV2 as _ETHX
                _node = _ETHX()
                _patched_bundle, _info = _node.run(
                    pose_data, images, checkpoint="", checkpoint_path_override="",
                    device="auto", blend=1.0, batch_size=8,
                )
                pose_data["iris_data"] = _patched_bundle.get("iris_data", pose_data["iris_data"])
                all_iris = pose_data["iris_data"]
                _ethxgaze_ok = True
                _engine = "ethxgaze"
                logging.getLogger(__name__).info("ethxgaze post-process: %s", _info)
            except Exception as _exc:                                    # noqa: BLE001
                _gaze_note = (
                    "ETH-XGaze (~2.5°) checkpoint missing — using "
                    f"{_engine}. Drop epoch_24_ckpt.pth.tar into models/ethxgaze/ "
                    "to enable it. (" + str(_exc)[:80] + ")")
                logging.getLogger(__name__).warning(
                    "gaze_engine=ethxgaze post-process failed (%s); "
                    "keeping previous engine output.", _exc,
                )

        # C0.1: Per-frame iris repaint at gaze-corrected position.
        # When apply_gaze_to_face_image != "off", deliver the gaze correction
        # into each 512x512 face crop at the position implied by the
        # corrected gaze unit vector. Mutates face_images_np in place
        # (face_images_tensor shares memory). Failures are non-fatal.
        _gaze_paint_mode = str(apply_gaze_to_face_image or "off").strip().lower()

        # Wan-Animate spec 1.5: deliver gaze correction as a REAL-PIXEL warp
        # instead of a synthetic iris-disk paint. The face encoder's training
        # augmentations were scale/color-jitter/noise — never a hard-edged
        # synthetic object — so a flat cv2.circle() stamp is out-of-distribution
        # input. This moves the actual iris texture using the same Delaunay
        # piecewise-affine engine WanFaceController3DV2 uses for FACS edits:
        # the 68 iBUG face landmarks stay anchored (src==dst for all of them,
        # so eyelid shape / rest of the face never deforms) and a small ring
        # of points synthesized around the DETECTED iris (real, tracked
        # position, not a guess) is the only thing that moves, to the
        # gaze-corrected position — dragging real nearby pixels with it via
        # the triangle-mask compositing already proven in _face_warp.py.
        if _gaze_paint_mode == "warp" and face_images_np.size > 0:
            try:
                from .nodes_extras._face_warp import warp_face, warp_available
                if not warp_available():
                    raise RuntimeError("cv2 unavailable for _face_warp")
                _is_float = np.issubdtype(face_images_np.dtype, np.floating)
                _GAIN = 3.0
                _RING_N = 8               # points synthesized around the iris
                _APERTURE_CLAMP = 0.65    # spec 1.5.3: clamp to ~0.6-0.7x aperture
                _n = min(face_images_np.shape[0], len(all_iris), len(face_bboxes),
                         len(pose_metas))
                # Blur gate, RELATIVE to this clip (fixed 2026-07-30). It was an
                # ABSOLUTE Laplacian-variance threshold of 50.0, which is
                # meaningless without a reference: a sharp studio portrait measures
                # ~62, so anything even slightly softer - compressed video, shallow
                # depth of field, or simply a face small enough in frame that its
                # crop is UPSCALED into the 512 tile - fell under it. The result was
                # that on most real footage EVERY frame was skipped and
                # apply_gaze_to_face_image silently did nothing, with no message.
                # What the gate is for is 'do not warp a frame much blurrier than
                # the rest of the shot' (motion blur, autofocus hunt), so measure
                # against this clip's own median instead of an absolute number.
                _scores = np.array(
                    [compute_frame_blur_score(face_images_np[_i]) for _i in range(_n)],
                    dtype=np.float32,
                )
                _blur_gate = max(1.0, 0.5 * float(np.median(_scores))) if _n else 1.0
                _gz_warped = 0
                _gz_skipped = 0
                for _idx in range(_n):
                    _bb = face_bboxes[_idx]
                    if _bb is None:
                        continue
                    _x1, _x2, _y1, _y2 = _bb
                    _cw = max(1, int(_x2) - int(_x1))
                    _ch = max(1, int(_y2) - int(_y1))
                    _sx = 512.0 / float(_cw)
                    _sy = 512.0 / float(_ch)
                    _crop = face_images_np[_idx]
                    # Spec 1.5.6: gate through the same quality check as the
                    # crop pipeline — skip warping (leave the crop as the
                    # normal pipeline produced it) on a low-quality frame
                    # rather than warp already-bad pixels.
                    if _scores[_idx] < _blur_gate:
                        _gz_skipped += 1
                        continue
                    _it = all_iris[_idx]
                    # Anchor points: the full 68-ish iBUG face landmark set,
                    # mapped into this crop's local pixel space. Identical in
                    # src/dst (never moves) unless an eye's ring is appended.
                    _face_kps = pose_metas[_idx].get('keypoints_face')
                    if _face_kps is None:
                        continue
                    _face_kps = np.asarray(_face_kps, dtype=np.float32)
                    if _face_kps.ndim != 2 or _face_kps.shape[0] < 3:
                        continue
                    _face_px = _face_kps[:, :2] * (W, H)
                    _face_crop_xy = np.stack([
                        (_face_px[:, 0] - float(_x1)) * _sx,
                        (_face_px[:, 1] - float(_y1)) * _sy,
                    ], axis=1)
                    _src_rows = [_face_crop_xy]
                    _dst_rows = [_face_crop_xy]
                    _any_eye = False
                    for _eye in ("right", "left"):
                        _iris = _it.get(f"{_eye}_iris") if isinstance(_it, dict) else None
                        _gaze = _it.get(f"{_eye}_gaze") if isinstance(_it, dict) else None
                        if not _iris or not _gaze:
                            continue
                        _ipx = float(_iris.get("x", 0.0))
                        _ipy = float(_iris.get("y", 0.0))
                        _ir  = float(_iris.get("radius", 4.0))
                        _dx  = float(_gaze.get("dx", 0.0))
                        _dy  = float(_gaze.get("dy", 0.0))
                        _mn  = float(_gaze.get("magnitude_norm", 0.0))
                        _yaw   = float(_gaze.get("yaw_rad", 0.0))
                        _pitch = float(_gaze.get("pitch_rad", 0.0))
                        _ref = _it.get(f"{_eye}_eye_ref") if isinstance(_it, dict) else None
                        _cr  = max(2, int(round(_ir * 0.5 * (_sx + _sy))))
                        _ex0 = int(round((_ipx - float(_x1)) * _sx))
                        _ey0 = int(round((_ipy - float(_y1)) * _sy))
                        # Same eyeball-projection math as the paint path
                        # (kept identical so "warp" and "overlay"/"replace"
                        # agree on WHERE the corrected iris should be).
                        if _ref and float(_ref.get("hw", 0.0)) > 1.0:
                            _ecx = (float(_ref["cx"]) - float(_x1)) * _sx
                            _ecy = (float(_ref["cy"]) - float(_y1)) * _sy
                            _hw_px = float(_ref["hw"]) * 0.5 * (_sx + _sy)
                            _R = _hw_px * 1.04
                            _ang = math.hypot(_yaw, _pitch)
                            if _ang > 1e-6:
                                _off = _R * math.sin(min(_ang, 1.2))
                            else:
                                _off = _hw_px * _mn
                            _ux, _uy, _un = _dx, _dy, math.hypot(_dx, _dy)
                            if _un < 1e-6 and _ang > 1e-6:
                                _ux, _uy = math.sin(_yaw), -math.sin(_pitch)
                                _un = math.hypot(_ux, _uy)
                            if _un > 1e-6:
                                _ux, _uy = _ux / _un, _uy / _un
                            _aperture = _hw_px
                        else:
                            _hw_px = None
                            _aperture = _cr * _GAIN if _cr * _GAIN > 1e-6 else _ir * 3.0
                            _ux, _uy = _dx, _dy
                            _off = _cr * _GAIN * _mn
                        # Spec 1.5.3: clamp displacement to ~0.6-0.7x the eye
                        # aperture so the warp can't push texture past the
                        # eyelid boundary (where piecewise-affine warps tear).
                        _disp = min(_off, _aperture * _APERTURE_CLAMP) if _aperture else 0.0
                        _cx1 = _ex0 + _ux * _disp
                        _cy1 = _ey0 + _uy * _disp
                        _ex0c = max(_cr, min(511 - _cr, _ex0))
                        _ey0c = max(_cr, min(511 - _cr, _ey0))
                        _cx1c = max(_cr, min(511 - _cr, _cx1))
                        _cy1c = max(_cr, min(511 - _cr, _cy1))
                        if math.hypot(_cx1c - _ex0c, _cy1c - _ey0c) < 0.5:
                            continue  # negligible correction — nothing to warp
                        _ring_ang = np.linspace(0, 2 * np.pi, _RING_N, endpoint=False)
                        _ring_src = np.stack([
                            _ex0c + _cr * np.cos(_ring_ang),
                            _ey0c + _cr * np.sin(_ring_ang),
                        ], axis=1).astype(np.float32)
                        _ring_dst = _ring_src + np.array(
                            [_cx1c - _ex0c, _cy1c - _ey0c], dtype=np.float32,
                        )
                        # Anchor points AT the iris centre too so the interior
                        # of the ring (not just its rim) drags along smoothly.
                        _src_rows.append(np.vstack([_ring_src, [[_ex0c, _ey0c]]]))
                        _dst_rows.append(np.vstack([_ring_dst, [[_cx1c, _cy1c]]]))
                        _any_eye = True
                    if not _any_eye:
                        continue
                    _gz_warped += 1
                    _src_lms = np.vstack(_src_rows).astype(np.float32)
                    _dst_lms = np.vstack(_dst_rows).astype(np.float32)
                    _crop_f = _crop.astype(np.float32) / 255.0 if not _is_float else _crop.astype(np.float32)
                    _warped = warp_face(_crop_f, _src_lms, _dst_lms)
                    face_images_np[_idx] = (
                        _warped if _is_float else np.clip(_warped * 255.0, 0, 255).astype(_crop.dtype)
                    )
                face_images_tensor = torch.from_numpy(face_images_np)
                # Never fail silently: this is the ONLY route from a gaze
                # estimate to the render, so if it does nothing the user must be
                # told rather than left guessing.
                if _gz_warped:
                    logging.getLogger(__name__).info(
                        "PoseAndFaceDetectionV2: gaze warp moved iris pixels on %d/%d "
                        "face crops (%d skipped as blurrier than half this clip's median "
                        "sharpness of %.1f).", _gz_warped, _n, _gz_skipped,
                        float(np.median(_scores)) if _n else 0.0,
                    )
                else:
                    logging.getLogger(__name__).warning(
                        "PoseAndFaceDetectionV2: apply_gaze_to_face_image='warp' changed "
                        "nothing on any of %d frames (%d skipped as too blurry). Usually "
                        "this means the gaze estimate already agrees with the performer's "
                        "real iris position - the healthy case, the eyes are already "
                        "correct in face_images. It can also mean no iris was detected.",
                        _n, _gz_skipped,
                    )
            except Exception as _exc:                                    # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "apply_gaze_to_face_image=warp failed (%s); "
                    "leaving face_images unmodified.", _exc,
                )
        elif _gaze_paint_mode in ("overlay", "replace") and face_images_np.size > 0:
            try:
                _is_float = np.issubdtype(face_images_np.dtype, np.floating)
                _IRIS_COLOR = (0.16, 0.16, 0.16) if _is_float else (40, 40, 40)
                _EYE_WHITE  = (0.92, 0.91, 0.88) if _is_float else (235, 232, 225)
                _GAIN = 3.0   # multiples of iris radius per unit magnitude_norm
                _n = min(face_images_np.shape[0], len(all_iris), len(face_bboxes))
                for _idx in range(_n):
                    _bb = face_bboxes[_idx]
                    if _bb is None:
                        continue
                    _x1, _x2, _y1, _y2 = _bb
                    _cw = max(1, int(_x2) - int(_x1))
                    _ch = max(1, int(_y2) - int(_y1))
                    _sx = 512.0 / float(_cw)
                    _sy = 512.0 / float(_ch)
                    _crop = face_images_np[_idx]   # view (512,512,C)
                    _it = all_iris[_idx]
                    for _eye in ("right", "left"):
                        _iris = _it.get(f"{_eye}_iris") if isinstance(_it, dict) else None
                        _gaze = _it.get(f"{_eye}_gaze") if isinstance(_it, dict) else None
                        if not _iris or not _gaze:
                            continue
                        _ipx = float(_iris.get("x", 0.0))
                        _ipy = float(_iris.get("y", 0.0))
                        _ir  = float(_iris.get("radius", 4.0))
                        _dx  = float(_gaze.get("dx", 0.0))
                        _dy  = float(_gaze.get("dy", 0.0))
                        _mn  = float(_gaze.get("magnitude_norm", 0.0))
                        _yaw   = float(_gaze.get("yaw_rad", 0.0))
                        _pitch = float(_gaze.get("pitch_rad", 0.0))
                        _ref = _it.get(f"{_eye}_eye_ref") if isinstance(_it, dict) else None
                        _cr  = max(2, int(round(_ir * 0.5 * (_sx + _sy))))
                        # Detected iris position in crop coords (erased in
                        # "replace" mode so two pupils never coexist).
                        _ex0 = int(round((_ipx - float(_x1)) * _sx))
                        _ey0 = int(round((_ipy - float(_y1)) * _sy))
                        if _ref and float(_ref.get("hw", 0.0)) > 1.0:
                            # 3D→2D eyeball projection (user spec): anchor at
                            # the eye-SOCKET centre and place the iris at
                            # R_eye·sin(gaze_angle) along the head-corrected
                            # gaze direction — the absolute position a real
                            # eyeball shows. The old code shifted the DETECTED
                            # iris (which already encodes gaze) even further —
                            # double-counting — by the arbitrary scale
                            # iris_radius×3, so Wan Animate received wrong
                            # eyeball directions.
                            _ecx = (float(_ref["cx"]) - float(_x1)) * _sx
                            _ecy = (float(_ref["cy"]) - float(_y1)) * _sy
                            _hw_px = float(_ref["hw"]) * 0.5 * (_sx + _sy)
                            _R = _hw_px * 1.04        # eyeball radius ≈ socket half-width
                            _ang = math.hypot(_yaw, _pitch)
                            if _ang > 1e-6:
                                _off = _R * math.sin(min(_ang, 1.2))
                            else:
                                # legacy 2D source: magnitude_norm is defined
                                # as offset/half-width — reconstruct directly.
                                _off = _hw_px * _mn
                            _ux, _uy, _un = _dx, _dy, math.hypot(_dx, _dy)
                            if _un < 1e-6 and _ang > 1e-6:
                                _ux, _uy = math.sin(_yaw), -math.sin(_pitch)
                                _un = math.hypot(_ux, _uy)
                            if _un > 1e-6:
                                _ux, _uy = _ux / _un, _uy / _un
                            # Clamp inside the palpebral opening so the iris
                            # is never painted over the eyelids.
                            _u = max(-0.92 * _hw_px, min(0.92 * _hw_px, _ux * _off))
                            _v = max(-0.55 * _hw_px, min(0.55 * _hw_px, _uy * _off))
                            _cx1 = int(round(_ecx + _u))
                            _cy1 = int(round(_ecy + _v))
                        else:
                            # Old bundles without eye_ref: keep legacy shift.
                            _shift = _cr * _GAIN * _mn
                            _cx1 = int(round(_ex0 + _dx * _shift))
                            _cy1 = int(round(_ey0 + _dy * _shift))
                        # Clamp to crop interior.
                        _ex0 = max(_cr, min(511 - _cr, _ex0))
                        _ey0 = max(_cr, min(511 - _cr, _ey0))
                        _cx1 = max(_cr, min(511 - _cr, _cx1))
                        _cy1 = max(_cr, min(511 - _cr, _cy1))
                        if _gaze_paint_mode == "replace":
                            cv2.circle(_crop, (_ex0, _ey0), _cr, _EYE_WHITE, -1, lineType=cv2.LINE_AA)
                        cv2.circle(_crop, (_cx1, _cy1), _cr, _IRIS_COLOR, -1, lineType=cv2.LINE_AA)
                # face_images_tensor shares memory with face_images_np;
                # rebind defensively in case cv2 returned a new array.
                face_images_tensor = torch.from_numpy(face_images_np)
            except Exception as _exc:                                    # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "apply_gaze_to_face_image=%s failed (%s); "
                    "leaving face_images unmodified.",
                    _gaze_paint_mode, _exc,
                )

        # C2.3: Wan-Animate spec 2.3 — pre-encode AU amplification. Pushes
        # each frame's face a bit further along the direction it already
        # moved from a neutral reference (au_amplify_neutral_frame), so more
        # of a genuinely subtle real microexpression survives the face
        # encoder's fixed-capacity motion-basis compression. Amplifies only
        # DETECTED motion (never synthesizes); runs AFTER any gaze warp above
        # so the two compose. Uses MediaPipe FaceMesh directly on each
        # 512x512 crop (self-contained — independent of which gaze_engine is
        # selected) and a 2D eye-line rigid alignment (translation+rotation+
        # uniform-scale from the two outer eye corners) to separate head
        # motion from expression motion before amplifying the residual.
        if float(au_amplify) > 1.001 and face_images_np.size > 0:
            try:
                from .nodes_extras._face_warp import warp_face, warp_available
                if not warp_available():
                    raise RuntimeError("cv2 unavailable for _face_warp")
                _is_float = np.issubdtype(face_images_np.dtype, np.floating)
                _BLUR_GATE = 50.0
                _neutral_idx = int(np.clip(au_amplify_neutral_frame, 0, face_images_np.shape[0] - 1))
                _neutral_crop_u8 = (
                    face_images_np[_neutral_idx] if not _is_float
                    else np.clip(face_images_np[_neutral_idx] * 255.0, 0, 255).astype(np.uint8)
                )
                _neutral_mp = _run_mediapipe_on_face_crop(_neutral_crop_u8, (0, 0), (512, 512), 512, 512)
                if _neutral_mp is None:
                    raise RuntimeError("MediaPipe found no face on the neutral reference frame")
                _neutral_pts = _neutral_mp['kps68_norm'][:, :2] * 512.0
                _neutral_eye_l = np.asarray(_neutral_mp['right_eye_outer_px'], dtype=np.float64)
                _neutral_eye_r = np.asarray(_neutral_mp['left_eye_outer_px'], dtype=np.float64)
                _n_amp = face_images_np.shape[0]
                # TWO PASSES (fixed 2026-07-31). This used to measure, amplify
                # and warp in a single per-frame loop, with NO temporal
                # filtering anywhere. MediaPipe landmarks jitter 1-3px frame to
                # frame even on a perfectly still face; multiplying that by
                # au_amplify and handing it to a Delaunay warp as control
                # points meant EVERY FRAME GOT A DIFFERENT WARP, driven by
                # amplified noise. That is a pixel-level jitter, so it survives
                # any crop_mode — a perfectly locked jitterless box still
                # produced a wobbling face, because the box was never the thing
                # moving.
                #
                # Pass 1 measures the displacement the warp would apply.
                # Pass 2 smooths that displacement per landmark across time
                # (zero-phase, so real expression onsets are not delayed) and
                # only then warps. Genuine expression is low-frequency and
                # survives; detector noise is high-frequency and does not.
                _disp = np.zeros((_n_amp, len(_neutral_pts), 2), np.float32)
                _src_pts = [None] * _n_amp
                for _idx in range(_n_amp):
                    _crop = face_images_np[_idx]
                    if compute_frame_blur_score(_crop) < _BLUR_GATE:
                        continue  # spec 2.3.6-style gate: skip a low-quality frame
                    _crop_u8 = _crop if not _is_float else np.clip(_crop * 255.0, 0, 255).astype(np.uint8)
                    _cur_mp = _run_mediapipe_on_face_crop(_crop_u8, (0, 0), (512, 512), 512, 512)
                    if _cur_mp is None:
                        continue
                    _cur_pts = _cur_mp['kps68_norm'][:, :2] * 512.0
                    _cur_eye_l = np.asarray(_cur_mp['right_eye_outer_px'], dtype=np.float64)
                    _cur_eye_r = np.asarray(_cur_mp['left_eye_outer_px'], dtype=np.float64)
                    _amp_pts = amplify_landmarks_from_neutral(
                        _cur_pts, _neutral_pts, _cur_eye_l, _cur_eye_r,
                        _neutral_eye_l, _neutral_eye_r, float(au_amplify),
                    )
                    _src_pts[_idx] = _cur_pts.astype(np.float32)
                    _d = (_amp_pts - _cur_pts).astype(np.float32)
                    # PIN THE FACE OUTLINE (fixed 2026-07-31). In the 68-point
                    # iBUG layout, indices 0-16 are the JAW CONTOUR — the outer
                    # boundary of the face. warp_face pins the tile border, so
                    # the region between the jaw and that border is spanned by
                    # triangles whose inner vertices are these contour points.
                    # Amplifying them shears every one of those triangles, which
                    # drags the forehead, ears, neck and background with it: the
                    # whole image warps instead of just the expression. Holding
                    # the outline fixed confines the warp to the interior, which
                    # is where expression actually lives (brows, eyes, nose,
                    # mouth). Jaw OPENING still comes through, because the mouth
                    # and chin interior points are not pinned.
                    if _d.shape[0] >= 17:
                        _d[:17] = 0.0
                    _disp[_idx] = _d

                # Gaussian, not one_euro. Measured against a known ground truth
                # (real expression + 1.5px detector noise, au_amplify=1.3),
                # frame-to-frame warp wobble and error vs the true amplification:
                #     no filter          0.512px wobble, 0.361px error
                #     one_euro           0.259px       , 0.239px
                #     gaussian window 11 0.063px       , 0.147px
                # A symmetric kernel is zero-phase, so an expression onset is
                # not delayed, and this is offline work — there is no reason to
                # accept the causal filter's lag or its weaker rejection. The
                # thing being smoothed is a DISPLACEMENT FIELD, which is
                # smooth in time by nature; only the detector noise is not.
                if _n_amp > 2:
                    for _li in range(_disp.shape[1]):
                        for _ax in range(2):
                            _disp[:, _li, _ax] = _smooth_1d(
                                _disp[:, _li, _ax], method="gaussian",
                                gaussian_window=11,
                            )

                _n_warped_au = 0
                for _idx in range(_n_amp):
                    _cur_pts = _src_pts[_idx]
                    if _cur_pts is None:
                        continue
                    _amp_pts = _cur_pts + _disp[_idx]
                    if np.allclose(_amp_pts, _cur_pts, atol=0.25):
                        continue  # negligible amplification — skip the warp
                    _crop = face_images_np[_idx]
                    _crop_f = _crop.astype(np.float32) / 255.0 if not _is_float else _crop.astype(np.float32)
                    # CLOSED PINNED RING around the face. Pinning the jaw
                    # (indices 0-16) is not enough on its own: the iBUG contour
                    # is an open ARC with no forehead points, so the face is
                    # unbounded at the top and triangles from the brow region
                    # run straight out to the tile border — dragging forehead,
                    # hair and background. A closed ellipse just outside the
                    # landmarks, identical in src and dst, seals that gap, so
                    # every triangle touching anything beyond the face has all
                    # its vertices fixed and cannot move.
                    _ring = _face_pin_ring(_cur_pts, 512, 512)
                    _s_all = np.vstack([_cur_pts, _ring]).astype(np.float32)
                    _d_all = np.vstack([_amp_pts, _ring]).astype(np.float32)
                    _warped = warp_face(_crop_f, _s_all, _d_all)
                    face_images_np[_idx] = (
                        _warped if _is_float else np.clip(_warped * 255.0, 0, 255).astype(_crop.dtype)
                    )
                    _n_warped_au += 1
                logging.getLogger(__name__).info(
                    "PoseAndFaceDetectionV2: au_amplify=%.2f warped %d/%d crops; "
                    "warp displacement temporally smoothed (mean %.2fpx, max %.2fpx) "
                    "so detector jitter is not amplified into the pixels.",
                    float(au_amplify), _n_warped_au, _n_amp,
                    float(np.abs(_disp).mean()), float(np.abs(_disp).max()),
                )
                face_images_tensor = torch.from_numpy(face_images_np)
            except Exception as _exc:                                    # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "au_amplify failed (%s); leaving face_images unamplified.", _exc,
                )

        # --- Debug visualisation ---
        debug_frames = []
        for idx in _IC.track(
            range(B), B, "WanAnimateV2: per-frame finalize",
        ):
            frame = images_np[idx]
            if frame.dtype != np.uint8:
                frame_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            else:
                frame_u8 = frame.copy()
            vis = draw_debug_overlay(
                frame_u8, pose_metas[idx]['keypoints_face'],
                all_iris[idx], face_bboxes[idx], bboxes[idx], W, H,
            )
            # Fold the full OP18 body skeleton + hands onto the same debug image
            # (this is what the old standalone WanPoseOverlayV2 node did — now
            # built in, so debug_image is a complete verification overlay:
            # body + face + hands + iris + gaze, no second node to wire).
            try:
                from .nodes_extras.pose_overlay import draw_body_skeleton_rgb as _draw_body
                _draw_body(cv2, vis, pose_metas[idx], W, H)
            except Exception as _ov_exc:                                  # noqa: BLE001
                logging.getLogger(__name__).debug(
                    "body-skeleton overlay skipped on frame %d: %s", idx, _ov_exc,
                )
            debug_frames.append(vis)
        debug_np = np.stack(debug_frames, 0).astype(np.float32) / 255.0
        debug_tensor = torch.from_numpy(debug_np)

        # --- Aggregate per-frame eye/lip outputs ---
        right_pupil_seq = [
            [round(it['right_iris']['x'], 3), round(it['right_iris']['y'], 3)]
            for it in all_iris
        ]
        left_pupil_seq = [
            [round(it['left_iris']['x'], 3), round(it['left_iris']['y'], 3)]
            for it in all_iris
        ]
        mean_lip_openness = float(np.mean(all_lip_ratios)) if all_lip_ratios else 0.0

        # Per-frame paste-back metadata. Always emit so downstream nodes
        # can rely on it regardless of crop_mode.
        restore_info = {
            "frame_shape": [int(H), int(W)],
            "resized_to": [512, 512],
            "crop_mode": str(crop_mode),
            "crops": [
                {
                    "frame": int(i),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "size": [int(y2 - y1), int(x2 - x1)],
                }
                for i, (x1, x2, y1, y2) in enumerate(face_bboxes)
            ],
            "face_images_quality": _face_q,
        }

        # ── face_images_512: force-resized to 512x512 for the Wan 2.2 ──
        # Animate face encoder. Always bilinear, regardless of source
        # crop size. Computed lazily here (single resize at the end) so
        # the per-frame loop stays untouched. Falls back to the original
        # tensor if either dimension already equals 512.
        try:
            import torch.nn.functional as _F
            if (face_images_tensor.shape[1] == 512
                    and face_images_tensor.shape[2] == 512):
                face_images_512_tensor = face_images_tensor
            else:
                _t = face_images_tensor.permute(0, 3, 1, 2).contiguous()
                # Bug-fix (Wan-Animate spec 1.1): was always "bilinear" — the
                # softest available filter, blind to upscale-vs-downscale.
                # For the common half-/full-body case the source crop is
                # SMALLER than 512, so this was an upscale using the softest
                # filter, blurring the fine detail the face encoder needs.
                # torch has no Lanczos; use bicubic (sharper upsample) for
                # upscales and area (correct anti-aliasing) for downscales —
                # same area/sharp split as utils.resize_face_crop.
                _src_h, _src_w = face_images_tensor.shape[1], face_images_tensor.shape[2]
                if _src_h > 512 or _src_w > 512:
                    _t = _F.interpolate(_t, size=(512, 512), mode="area")
                else:
                    _t = _F.interpolate(_t, size=(512, 512),
                                        mode="bicubic", align_corners=False)
                face_images_512_tensor = _t.permute(0, 2, 3, 1).contiguous()
        except Exception:  # noqa: BLE001 — never break the node on resize
            face_images_512_tensor = face_images_tensor

        # C.2/C0.6 — UI payload for pose_gaze_viewer.js
        # Compact per-frame summary (skeleton+iris+gaze unit-vectors).
        # Capped at 240 frames to keep websocket cheap; viewer falls
        # back to "no data" if absent.
        _viewer_frames = []
        try:
            _max_f = min(240, len(all_iris))
            for _fi in range(_max_f):
                _ir = all_iris[_fi] or {}
                _ri = _ir.get('right_iris') or {}
                _li = _ir.get('left_iris') or {}
                _rg = _ir.get('right_gaze') or {}
                _lg = _ir.get('left_gaze') or {}
                _kp_body = pose_metas[_fi].get('keypoints_body') if _fi < len(pose_metas) else None
                _skel = []
                if _kp_body is not None:
                    for _kp in _kp_body:
                        if _kp is None:
                            _skel.append(None)
                        else:
                            # Carry the CONFIDENCE (fixed 2026-07-31). This
                            # used to pack x/y only and drop _kp[2], so joints
                            # the detector never found — hips, knees and ankles
                            # in a head-and-shoulders shot, or anything zeroed
                            # by pose_threshold — were still emitted with
                            # whatever coordinates they happened to carry, and
                            # the viewer drew edges to them. That is the fan of
                            # lines shooting off across the frame that made the
                            # skeleton look shattered. The reference drawer
                            # (draw_aapose_new) gates every joint on
                            # threshold=0.5; the viewer never got that gate.
                            _c = float(_kp[2]) if len(_kp) > 2 else 1.0
                            _skel.append([round(float(_kp[0]) * float(W), 1),
                                          round(float(_kp[1]) * float(H), 1),
                                          round(_c, 3)])
                _viewer_frames.append({
                    "frame": _fi,
                    "skeleton": _skel,
                    "right_iris": [round(float(_ri.get('x', 0.0)), 1),
                                   round(float(_ri.get('y', 0.0)), 1)],
                    "left_iris":  [round(float(_li.get('x', 0.0)), 1),
                                   round(float(_li.get('y', 0.0)), 1)],
                    "right_gaze": [round(float(_rg.get('dx', 0.0)), 3),
                                   round(float(_rg.get('dy', 0.0)), 3),
                                   round(float(_rg.get('magnitude_norm', 0.0)), 3)],
                    "left_gaze":  [round(float(_lg.get('dx', 0.0)), 3),
                                   round(float(_lg.get('dy', 0.0)), 3),
                                   round(float(_lg.get('magnitude_norm', 0.0)), 3)],
                })
        except Exception:  # noqa: BLE001
            _viewer_frames = []
        # Clean downscaled frame previews so the viewer can REVEAL the
        # skeleton/iris/gaze overlaid on the ACTUAL image (not a blank tile).
        # Capped + downscaled to keep the websocket cheap; the frontend draws
        # the nearest available preview as the backdrop for the current frame.
        _viewer_previews = []
        try:
            import base64  # noqa: PLC0415
            import cv2 as _cv2  # noqa: N813
            _pv_max = min(60, int(B))
            # Bug-fix: this feeds pose_gaze_viewer.js's ONLY visual check for
            # micro-expression fidelity (skeleton/iris/gaze drawn over the
            # ACTUAL frame). 320px/quality-72 was fine for "does the skeleton
            # look roughly right" but far too soft to judge subtle expression
            # detail — the browser then upscales this already-blurry, heavily
            # compressed thumbnail to fill the (much larger) canvas panel,
            # compounding the softness. 640px/quality-92 is near-lossless for
            # photographic content and still cheap over a localhost websocket
            # (this is a UI preview only — never feeds face_images/pose_data).
            _pv_long = 640  # longest side px
            _idxs = ([0] if _pv_max <= 1
                     else [round(i * (B - 1) / (_pv_max - 1)) for i in range(_pv_max)])
            for _pi in sorted(set(_idxs)):
                _fr = images_np[_pi]  # H,W,C float 0..1
                _hh, _ww = _fr.shape[:2]
                _sc = _pv_long / float(max(_hh, _ww))
                if _sc < 1.0:
                    _fr = _cv2.resize(_fr, (max(1, int(_ww * _sc)), max(1, int(_hh * _sc))),
                                      interpolation=_cv2.INTER_AREA)
                _u8 = (np.clip(_fr, 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, ::-1]  # RGB->BGR
                _ok, _buf = _cv2.imencode(".jpg", _u8, [int(_cv2.IMWRITE_JPEG_QUALITY), 92])
                if _ok:
                    _viewer_previews.append({
                        "frame": int(_pi),
                        "b64": "data:image/jpeg;base64," + base64.b64encode(_buf.tobytes()).decode("ascii"),
                    })
        except Exception:  # noqa: BLE001
            _viewer_previews = []
        _GAZE_ACCURACY = {
            # Honest labels: the CNN engines' MAE figures are WITHIN-dataset
            # lab benchmarks. Cross-domain (in-the-wild portraits/video) they
            # carry systematic bias — verified 2026-07-24 by reproducing the
            # official ETH-XGaze demo output exactly (faithful port), then
            # observing ~15° downward pitch bias on two clean at-camera
            # photos. That is the model's domain gap, not a pipeline bug.
            "ethxgaze": "~2.5° lab / wild varies", "pose_normalized_resnet50": "~3-4° MAE",
            "l2cs_mpiigaze": "~3.9° lab / wild varies", "l2cs_gaze360": "~10.4° MAE",
            "iris_geometric": "iris-measured (deterministic)",
            "blendshape_head_corrected": "blendshape + head (approx)",
            "blendshape_only": "eye-in-head (approx)",
            "legacy_iris_offset": "rough",
        }
        _ui_payload = {
            "viewer_meta": [json.dumps({
                "src_w": int(W), "src_h": int(H),
                "n_frames": int(B),
                "engine": str(_engine),
                "engine_requested": str(_gaze_requested),
                "engine_accuracy": _GAZE_ACCURACY.get(str(_engine), ""),
                "engine_status": _gaze_note,   # non-null only when it fell back
                "frames": _viewer_frames,
                "previews": _viewer_previews,
            })],
        }

        # Wan-Animate spec 3.1 (closed-loop critic, foundation): export the
        # ARKit-52 blendshapes this run measured, per frame, in the same
        # schema WanExpressionCoefficientsV2 used before it was folded in
        # here. Off by default (export_expression_coeffs=False) — zero cost
        # when unused. Never raises: a failure just leaves the JSON empty.
        expression_coeffs_json = "{}"
        if export_expression_coeffs:
            try:
                from .nodes_extras.expression_coeffs import _extract_blendshapes, ARKIT_52
                per_frame_bs = [_extract_blendshapes(entry) for entry in iris_output]
                names = []
                for bs in per_frame_bs:
                    if bs:
                        names = [n for n in ARKIT_52 if n in bs] + [n for n in bs if n not in ARKIT_52]
                        break
                names = names or list(ARKIT_52)
                expression_coeffs_json = json.dumps({
                    "fps": float(gaze_fps),
                    "names": names,
                    "frames": [
                        {"frame": i, "blendshapes": {n: float(bs.get(n, 0.0)) for n in names}}
                        for i, bs in enumerate(per_frame_bs)
                    ],
                })
            except Exception as _exc:                                    # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "export_expression_coeffs failed (%s); expression_coeffs_json left empty.", _exc,
                )

        # VRAM/RAM (2026-08-13): release transient detection buffers and
        # fragmented VRAM before returning. The blurred copy (a full
        # per-frame array) is only needed during detection/pose and is now
        # dead; drop it. empty_cache returns fragmented blocks from the
        # ONNX detection passes to the session so the next node (and the
        # sampler downstream) starts from a clean pool. Outputs are already
        # built, so this is safe.
        try:
            if use_blur_for_pose:
                images_blurred = None
        except Exception:  # noqa: BLE001
            pass
        try:
            import gc
            gc.collect()
            # NO `import torch` HERE (fixed 2026-08-13). torch is already
            # imported at module scope. A bare `import torch` inside this
            # function makes `torch` a LOCAL for the ENTIRE function body —
            # Python decides that at compile time, not at execution — so every
            # earlier use in the same function raises
            #     UnboundLocalError: cannot access local variable 'torch'
            # even though the module-level import is right there. It crashed at
            # line ~3165 (`torch.from_numpy(face_images_np)`), roughly 1800
            # lines BEFORE this cleanup block ever ran.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

        return {
            "ui": _ui_payload,
            "result": (
                pose_data,
                face_images_tensor,
                json.dumps(points_dict_list),
                [bbox_ints],
                # BBOX CONTRACT (fixed 2026-08-13). Internally the face box is
                # carried as (x1, x2, y1, y2) — a historical ordering this file
                # uses everywhere, and the source of an earlier sentinel bug.
                # ComfyUI's BBOX type, and every consumer of it (SAM2/SAM
                # segmentation, crop nodes, the mask stack), expects
                # (x1, y1, x2, y2). Emitting the internal order through a BBOX
                # output handed SAM2 a box whose "y1" was actually x2, i.e. a
                # garbage region. `bboxes` above was already correct because it
                # comes straight from YOLO in xyxy.
                # Converted HERE, at the boundary only, so no internal maths
                # changes and nothing else has to be touched.
                [(int(_b[0]), int(_b[2]), int(_b[1]), int(_b[3])) for _b in face_bboxes],
                json.dumps(iris_output),
                debug_tensor,
                json.dumps(right_pupil_seq),
                json.dumps(left_pupil_seq),
                mean_lip_openness,
                restore_info,
                float(face_cfg_scale),
                face_images_512_tensor,
                expression_coeffs_json,
            ),
        }


# ---------------------------------------------------
# Draw ViTPose
# ---------------------------------------------------
class DrawViTPoseV2:
    DESCRIPTION = (
        "Render the detected skeleton, face landmarks, iris pupils and gaze "
        "arrows onto a clean canvas at the target Wan 2.2 latent resolution. "
        "Outputs an IMAGE batch ready to drop into a Wan-Animate sampler."
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data":         ("POSEDATA", {"tooltip": "From Pose and Face Detection (V2)."}),
                "width":             ("INT",   {"default": 832, "min": 64, "max": 2048, "tooltip": "Render canvas width (px). Match the sampler latent size."}),
                "height":            ("INT",   {"default": 480, "min": 64, "max": 2048, "tooltip": "Render canvas height (px). Match the sampler latent size."}),
                "retarget_padding":  ("INT",   {"default": 16,  "min": 0,  "max": 512, "tooltip": "Padding (px) added around the body bbox when retargeting. Larger = more headroom for big motions."}),
                "body_stick_width":  ("INT",   {"default": -1,  "min": -1, "max": 20,  "tooltip": "Body skeleton stick width in px. -1 = auto from canvas size."}),
                "hand_stick_width":  ("INT",   {"default": -1,  "min": -1, "max": 20,  "tooltip": "Hand skeleton stick width in px. -1 = auto."}),
                "draw_head":         ("BOOLEAN", {"default": True, "tooltip": "Draw the head/face skeleton (eyes, nose, ears)."}),
                "pose_draw_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Per-keypoint score threshold for drawing."}),
            },
            # MANUAL bug-fix (Apr 2026): MediaPipe iris/gaze integration.
            # The Pose-and-Face-Detection node already produces per-frame
            # iris pixel coords + gaze vectors in pose_data["iris_data"];
            # these optional widgets let the rendered pose image carry
            # explicit pupil + gaze cues that the Wan 2.2 Animate sampler
            # consumes through cross-attention.  All defaults preserve the
            # legacy behaviour when the operator does not opt in.
            "optional": {
                "draw_iris": ("BOOLEAN", {"default": True,
                    "tooltip": "Draw iris/pupil markers from MediaPipe iris_data."}),
                "draw_gaze": ("BOOLEAN", {"default": True,
                    "tooltip": "Draw gaze direction arrows from iris_data."}),
                "iris_radius": ("INT", {"default": 4, "min": 1, "max": 20,
                    "tooltip": "Pupil circle radius in pixels."}),
                "gaze_arrow_len": ("INT", {"default": 30, "min": 4, "max": 200,
                    "tooltip": "Length of gaze direction arrow in pixels."}),
                "iris_min_confidence": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Skip iris frames whose detection confidence is below this."}),
                "iris_color": (["white", "magenta", "yellow", "green"], {"default": "white",
                    "tooltip": "Color of the drawn pupil; magenta gives strongest sampler signal."}),
                # ---- C0.5: face passthrough (Wan 2.2 Animate face encoder convenience) ----
                "face_images": ("IMAGE", {"tooltip": "OPTIONAL face crop IMAGE batch (typically the face_images_512 output of PoseAndFaceDetectionV2). When wired, the node validates frame-count parity with the pose batch, optionally force-resizes to 512x512, and forwards it on the 'face_video' output so a single DrawViTPoseV2 can feed the Wan-Animate sampler's pose+face inputs in one place."}),
                "face_cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1, "forceInput": True, "tooltip": "Passthrough face CFG scale, CONNECTION-ONLY (forceInput) so there is exactly one source of truth: PoseAndFaceDetectionV2.face_cfg_scale. It used to be a second independently-editable widget with the same default, so you could set 2.0 upstream, leave 1.0 here, and get no warning that they had diverged. Unconnected = 1.0 (no-op), which matches the old default. NOTE: Kijai's ComfyUI-WanVideoWrapper has no face-CFG input to wire this into today — for real control over expression adherence use WanVideoAnimateEmbeds.face_strength (spec 2.2's stronger, more direct block-scale lever) instead."}),
                "enforce_512_face": ("BOOLEAN", {"default": True, "tooltip": "If True and 'face_images' is provided at a non-512 size, force-resize each frame to 512x512 (bilinear) before forwarding. Default True so the encoder always sees the trained input shape."}),
                "reference_expression_coeffs_json": ("STRING", {"multiline": True, "default": "", "tooltip": "Wan-Animate spec 3.1 (closed-loop critic): wire in the 'expression_coeffs_json' output of a PoseAndFaceDetectionV2 run (export_expression_coeffs=True) on the SOURCE driving video. When non-empty, this node measures ARKit-52 blendshapes from ITS OWN pose_data.iris_data (i.e. the GENERATED Wan-Animate output side, since this node is downstream of the generation pass) and reports per-AU + per-segment error against the reference — a numeric fidelity signal instead of eyeballing frames. Leave empty to skip entirely (zero extra cost)."}),
                "segment_length": ("INT", {"default": 77, "min": 1, "max": 100000, "tooltip": "Frames per segment for the critic's worst-segment breakdown — match WanVideoAnimateEmbeds.frame_window_size (default 77) so segments line up with Wan-Animate's own splice boundaries (spec 2.5/3.5). Only used when reference_expression_coeffs_json is wired."}),
                "top_k_aus": ("INT", {"default": 10, "min": 1, "max": 52, "tooltip": "How many worst-tracked AUs the critic reports, worst-first. Only used when reference_expression_coeffs_json is wired."}),
                "apply_pose_edits_to_face": (["warp", "off"], {"default": "warp", "tooltip": "Expression-edit DELIVERY (2026-07-24). When pose_data carries edited face landmarks (WanFaceController3DV2 expression dials / dragged landmarks) AND face_images is wired, 'warp' moves the ACTUAL face-crop pixels from the original landmark positions to the edited ones (same Delaunay piecewise-affine engine as FC3D's preview), so the Wan-Animate face encoder sees the edit. Without this, landmark edits only change the drawn skeleton — the photographic face crop stays neutral and the sampler follows the crop, i.e. your expression edits silently do nothing. No-op when landmarks are unedited (zero cost), so the default stays 'warp'."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("pose_images", "face_video", "face_cfg_scale", "critic_report_json", "worst_aus_csv", "overall_mae")
    OUTPUT_TOOLTIPS = (
        "Rendered skeleton IMAGE batch. Feed into your Wan 2.2 Animate sampler.",
        "Passthrough face IMAGE batch (512x512 if enforce_512_face). Empty single-frame zero tensor if 'face_images' was not wired.",
        "Passthrough face_cfg_scale (Wan-Animate paper Sec. 4.3). 1.0 = CFG off.",
        "Wan-Animate spec 3.1 closed-loop critic report (JSON): per-AU mean-absolute-error, per-frame error curve, per-segment breakdown worst-first. '{}' when reference_expression_coeffs_json was not wired.",
        "CSV 'name,mae' for the top_k_aus worst-tracked AUs, worst first. Empty string when the critic did not run.",
        "Mean of all per-AU MAE values (0.0 = perfect match to the reference). 0.0 when the critic did not run.",
    )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess_V2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    @staticmethod
    def _padding_resize_transform(src_h, src_w, out_h, out_w):
        """Replicate utils.padding_resize math as a (scale, ox, oy) transform.

        Returns the per-pixel scale and (offset_x, offset_y) that map a
        source-coord (x, y) into the padded target canvas of size out_h*out_w.
        """
        if (src_h / max(src_w, 1)) > (out_h / max(out_w, 1)):
            new_w = int(out_h / src_h * src_w)
            scale = out_h / src_h
            ox = (out_w - new_w) // 2
            oy = 0
        else:
            new_h = int(out_w / src_w * src_h)
            scale = out_w / src_w
            ox = 0
            oy = (out_h - new_h) // 2
        return scale, ox, oy

    def _draw_iris_overlay(self, canvas, iris_dict, transform,
                            iris_radius, gaze_arrow_len, min_conf,
                            color_bgr, draw_iris, draw_gaze):
        if iris_dict is None:
            return
        scale, ox, oy = transform
        H, W = canvas.shape[:2]
        for eye_key, gaze_key in (("right_iris", "right_gaze"),
                                    ("left_iris", "left_gaze")):
            iris = iris_dict.get(eye_key)
            if not isinstance(iris, dict):
                continue
            try:
                conf = float(iris.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            if conf < min_conf:
                continue
            try:
                src_x = float(iris["x"]); src_y = float(iris["y"])
            except (KeyError, TypeError, ValueError):
                continue
            cx = int(round(src_x * scale + ox))
            cy = int(round(src_y * scale + oy))
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            if draw_iris:
                cv2.circle(canvas, (cx, cy), iris_radius, color_bgr, -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), max(iris_radius + 2, 6),
                           (0, 0, 0), 1, cv2.LINE_AA)
            if draw_gaze:
                gaze = iris_dict.get(gaze_key)
                if isinstance(gaze, dict):
                    try:
                        dx = float(gaze.get("dx", 0.0))
                        dy = float(gaze.get("dy", 0.0))
                    except (TypeError, ValueError):
                        dx = dy = 0.0
                    if abs(dx) > 1e-4 or abs(dy) > 1e-4:
                        # Magnitude-aware shrink (same convention as
                        # PoseAndFaceDetectionV2's draw_debug_overlay).
                        mag = 1.0
                        try:
                            mag = float(gaze.get('magnitude_norm', 1.0))
                        except (TypeError, ValueError):
                            mag = 1.0
                        eff_len = max(6, int(round(gaze_arrow_len * mag)))
                        ex = int(round(cx + dx * eff_len))
                        ey = int(round(cy + dy * eff_len))
                        cv2.arrowedLine(canvas, (cx, cy), (ex, ey),
                                        color_bgr, 2, cv2.LINE_AA, tipLength=0.3)

    def process(self, pose_data, width, height, body_stick_width, hand_stick_width,
                draw_head, pose_draw_threshold, retarget_padding=64,
                draw_iris=True, draw_gaze=True,
                iris_radius=4, gaze_arrow_len=30,
                iris_min_confidence=0.05, iris_color="white",
                face_images=None, face_cfg_scale=1.0, enforce_512_face=True,
                reference_expression_coeffs_json="", segment_length=77, top_k_aus=10,
                apply_pose_edits_to_face="warp"):
        with torch.inference_mode():
            return self._process_impl(
                pose_data, width, height, body_stick_width, hand_stick_width,
                draw_head, pose_draw_threshold, retarget_padding,
                draw_iris, draw_gaze,
                iris_radius, gaze_arrow_len,
                iris_min_confidence, iris_color,
                face_images, face_cfg_scale, enforce_512_face,
                reference_expression_coeffs_json, segment_length, top_k_aus,
                apply_pose_edits_to_face=apply_pose_edits_to_face,
            )

    def _process_impl(self, pose_data, width, height, body_stick_width, hand_stick_width,
                draw_head, pose_draw_threshold, retarget_padding=64,
                draw_iris=True, draw_gaze=True,
                iris_radius=4, gaze_arrow_len=30,
                iris_min_confidence=0.05, iris_color="white",
                face_images=None, face_cfg_scale=1.0, enforce_512_face=True,
                reference_expression_coeffs_json="", segment_length=77, top_k_aus=10,
                apply_pose_edits_to_face="warp"):
        # Migration guard for the face_cfg_scale -> forceInput change.
        # face_cfg_scale used to be an editable WIDGET at position 7 of this
        # node's optional widgets; forceInput removes it from widgets_values,
        # so a workflow saved BEFORE the change has one extra leading value and
        # everything after it shifts by one. The dangerous one is
        # apply_pose_edits_to_face (last), which would receive top_k_aus's INT
        # and silently disable the expression-delivery warp. Coerce the
        # shift-prone params back to something valid instead of trusting them.
        if str(apply_pose_edits_to_face) not in ("warp", "off"):
            logging.getLogger(__name__).warning(
                "DrawViTPoseV2: apply_pose_edits_to_face=%r is not a valid mode — "
                "this is the signature of a workflow saved before face_cfg_scale "
                "became a connection-only input (widget values shifted by one). "
                "Falling back to 'warp'. Re-save the workflow to clear this.",
                apply_pose_edits_to_face,
            )
            apply_pose_edits_to_face = "warp"
        try:
            segment_length = max(1, int(segment_length))
        except (TypeError, ValueError):
            segment_length = 77
        try:
            top_k_aus = max(1, min(52, int(top_k_aus)))
        except (TypeError, ValueError):
            top_k_aus = 10
        try:
            face_cfg_scale = float(face_cfg_scale)
        except (TypeError, ValueError):
            face_cfg_scale = 1.0

        pose_metas = pose_data["pose_metas"]
        draw_hand = hand_stick_width != 0

        # MANUAL bug-fix (Apr 2026): support optional iris drawing on top of
        # the rendered pose canvas.  iris_data is always rendered into the
        # *target* (width, height) coord system using the same padding-resize
        # transform that body keypoints went through.
        iris_data = pose_data.get("iris_data") or []
        src_size = pose_data.get("source_size")
        # RGB (cv2 expects BGR but we draw on a uint8 canvas that is later
        # converted to a float [0,1] tensor as RGB; OpenCV draws in BGR order
        # numerically, but since values are symmetric (white) or chosen to
        # match the eventual sampler signal we pick a single consistent
        # palette).  Here color tuples are (R, G, B) on the array directly.
        color_map = {
            "white":   (255, 255, 255),
            "magenta": (255, 0, 255),
            "yellow":  (255, 255, 0),
            "green":   (0, 255, 0),
        }
        iris_color_rgb = color_map.get(iris_color, (255, 255, 255))

        if src_size and len(src_size) == 2:
            transform = self._padding_resize_transform(
                int(src_size[0]), int(src_size[1]), int(height), int(width)
            )
        else:
            transform = None  # cannot retarget without source dims

        comfy_pbar = ProgressBar(len(pose_metas))
        progress = 0
        pose_images = []

        for idx, meta in _IC.track(
            list(enumerate(pose_metas)), len(pose_metas),
            "WanAnimateV2: draw pose images",
        ):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            pose_image = draw_aapose_by_meta_new(
                canvas,
                meta,
                draw_hand=draw_hand,
                draw_head=draw_head,
                body_stick_width=body_stick_width,
                hand_stick_width=hand_stick_width,
                threshold=pose_draw_threshold,
            )
            pose_image = padding_resize(pose_image, height, width)
            if transform is not None and idx < len(iris_data) and (draw_iris or draw_gaze):
                self._draw_iris_overlay(
                    pose_image, iris_data[idx], transform,
                    int(iris_radius), int(gaze_arrow_len),
                    float(iris_min_confidence), iris_color_rgb,
                    bool(draw_iris), bool(draw_gaze),
                )
            pose_images.append(pose_image)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_images_np = np.stack(pose_images, 0)
        pose_images_tensor = torch.from_numpy(pose_images_np).float() / 255.0

        # ---- C0.5: face passthrough -------------------------------------
        # If the user wired face_images, validate frame-count parity, optionally
        # force-resize to 512x512, and forward. Otherwise emit a single-frame
        # zero tensor (Comfy refuses None for declared IMAGE outputs).
        face_video_out = None
        if face_images is not None:
            try:
                if hasattr(face_images, "detach"):
                    _fi = face_images
                else:
                    _fi = torch.from_numpy(np.asarray(face_images))
                if not isinstance(_fi, torch.Tensor) or _fi.ndim != 4 or _fi.shape[-1] != 3:
                    raise ValueError(
                        f"DrawViTPoseV2: face_images expected (B,H,W,3); got {tuple(_fi.shape)}"
                    )
                if _fi.shape[0] != pose_images_tensor.shape[0]:
                    logging.getLogger(__name__).warning(
                        "DrawViTPoseV2: face_images frame count (%d) != pose frame count (%d); forwarding face_images as-is.",
                        int(_fi.shape[0]), int(pose_images_tensor.shape[0]),
                    )
                if bool(enforce_512_face) and (int(_fi.shape[1]) != 512 or int(_fi.shape[2]) != 512):
                    # (B,H,W,3) -> (B,3,H,W) -> resize -> (B,H,W,3)
                    _t = _fi.permute(0, 3, 1, 2).float()
                    # Bug-fix (Wan-Animate spec 1.1, same class as the
                    # per-frame crop resize): always-bilinear is blind to
                    # upscale-vs-downscale and blurs an upscaled face crop.
                    if int(_fi.shape[1]) > 512 or int(_fi.shape[2]) > 512:
                        _t = torch.nn.functional.interpolate(_t, size=(512, 512), mode="area")
                    else:
                        _t = torch.nn.functional.interpolate(
                            _t, size=(512, 512), mode="bicubic", align_corners=False,
                        )
                    face_video_out = _t.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)
                else:
                    face_video_out = _fi
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "DrawViTPoseV2: face passthrough failed (%s); forwarding original face_images.", e,
                )
                face_video_out = face_images

        # ---- Expression-edit delivery (2026-07-24) ----------------------
        # FC3D (WanFaceController3DV2) edits land in pose_data["pose_metas"]
        # as moved face landmarks, while pose_metas_original keeps the
        # detector's pristine set. Until now those edits only changed the
        # DRAWN skeleton — the photographic face crops passed through
        # untouched, and the Wan-Animate face encoder (which follows the
        # crop, not the dots) never saw the user's expression edits. Warp
        # each crop from original→edited landmark positions with the same
        # Delaunay engine FC3D's own preview uses. All landmark rows are
        # passed as control points — unmoved rows anchor themselves
        # (src==dst), so no row-indexing convention is assumed. No-op when
        # nothing was edited.
        if (str(apply_pose_edits_to_face) == "warp"
                and face_video_out is not None
                and isinstance(pose_data, dict)):
            try:
                _metas_e = pose_data.get("pose_metas") or []
                _metas_o = pose_data.get("pose_metas_original") or []
                _crops = pose_data.get("face_crop_boxes") or []
                _src_hw = pose_data.get("source_size") or None
                # Retarget moves the edited metas into the TARGET canvas's
                # coordinate space while the originals stay in source space —
                # a wholesale coordinate difference that is NOT a user edit.
                # Only deliver when not retargeting.
                if (_metas_e and _metas_o and _crops and _src_hw
                        and pose_data.get("retarget_image") is None):
                    from .nodes_extras._face_warp import warp_face as _warp_face, warp_available as _warp_ok
                    # The bundle's pose_metas are AAPoseMeta OBJECTS while
                    # pose_metas_original are plain dicts — use FC3D's own
                    # dual-shape accessor so the delta we see is exactly the
                    # edit FC3D wrote (returns full-frame-normalised (N,2)).
                    from .nodes_extras.expression_3d_coeffs import _read_face_normalised as _read_fn
                    if _warp_ok():
                        _Hs, _Ws = float(_src_hw[0]), float(_src_hw[1])
                        _fv = (face_video_out.detach().cpu().numpy()
                               if hasattr(face_video_out, "detach")
                               else np.asarray(face_video_out)).copy()
                        _n = min(len(_metas_e), len(_metas_o), len(_crops), int(_fv.shape[0]))
                        _n_warped = 0
                        for _i in range(_n):
                            _ke = _read_fn(_metas_e[_i])
                            _ko = _read_fn(_metas_o[_i])
                            if _ke is None or _ko is None:
                                continue
                            if _ke.shape != _ko.shape or _ke.shape[0] < 3:
                                continue
                            # Normalised full-frame units; ~1e-5 ≈ sub-0.01px.
                            if float(np.abs(_ke - _ko).max()) < 1e-5:
                                continue
                            _x1, _x2, _y1, _y2 = [float(v) for v in _crops[_i][:4]]
                            _cw = max(1.0, _x2 - _x1)
                            _ch = max(1.0, _y2 - _y1)
                            _th, _tw = float(_fv.shape[1]), float(_fv.shape[2])
                            _src_px = np.empty_like(_ko)
                            _dst_px = np.empty_like(_ke)
                            _src_px[:, 0] = (_ko[:, 0] * _Ws - _x1) / _cw * _tw
                            _src_px[:, 1] = (_ko[:, 1] * _Hs - _y1) / _ch * _th
                            _dst_px[:, 0] = (_ke[:, 0] * _Ws - _x1) / _cw * _tw
                            _dst_px[:, 1] = (_ke[:, 1] * _Hs - _y1) / _ch * _th
                            # Drop control points far outside the crop (e.g.
                            # the body-anchor row on half-body shots): warp_face
                            # would CLAMP them onto the crop border, where they
                            # form long degenerate triangles with the moved
                            # interior landmarks and smear the shift to the
                            # frame edge (proven in the offline synthetic test).
                            # Face landmarks — the ones that actually move —
                            # always live inside the face crop.
                            _pad = 32.0
                            _keep = ((_src_px[:, 0] > -_pad) & (_src_px[:, 0] < _tw + _pad)
                                     & (_src_px[:, 1] > -_pad) & (_src_px[:, 1] < _th + _pad)
                                     & (_dst_px[:, 0] > -_pad) & (_dst_px[:, 0] < _tw + _pad)
                                     & (_dst_px[:, 1] > -_pad) & (_dst_px[:, 1] < _th + _pad))
                            if int(_keep.sum()) < 3:
                                continue
                            _warped = _warp_face(_fv[_i].astype(np.float32),
                                                 _src_px[_keep], _dst_px[_keep])
                            _fv[_i] = np.clip(_warped, 0.0, 1.0)
                            _n_warped += 1
                        if _n_warped:
                            face_video_out = torch.from_numpy(_fv).float()
                            logging.getLogger(__name__).info(
                                "DrawViTPoseV2: delivered pose-landmark edits into "
                                "%d/%d face frames (apply_pose_edits_to_face=warp).",
                                _n_warped, _n,
                            )
            except Exception as _exc:                                    # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "DrawViTPoseV2: expression-edit face warp failed (%s); "
                    "forwarding unwarped face_images.", _exc,
                )
        if face_video_out is None:
            face_video_out = torch.zeros((1, 512, 512, 3), dtype=torch.float32)

        # Dead-knob guard (Kijai wrapper behaviour, verified in
        # nodes_sampler.py): predict_with_cfg() returns noise_pred_cond
        # immediately when math.isclose(cfg_scale, 1.0), so the uncond pass —
        # the ONLY place wananim_face_pixel_values is zeroed — never runs. In
        # that regime (any distilled few-step setup, e.g. a 4-step lightx2v
        # LoRA at cfg=1.0) face_cfg_scale has literally no effect path. Warn at
        # RUNTIME rather than only in a tooltip, so the user finds out before
        # spending a session tuning a knob that cannot do anything. The lever
        # that always works, in every CFG regime, is
        # WanVideoAnimateEmbeds.face_strength, which multiplies the face
        # adapter residual unconditionally (x.add(residual_out, alpha=strength)).
        try:
            if abs(float(face_cfg_scale) - 1.0) > 1e-6:
                logging.getLogger(__name__).warning(
                    "DrawViTPoseV2: face_cfg_scale=%.3f is a PASSTHROUGH only. "
                    "Kijai's sampler skips the uncond pass entirely when cfg==1.0 "
                    "(distilled/lightx2v few-step workflows), so face_cfg_scale "
                    "cannot affect face-conditioning strength there. Use "
                    "WanVideoAnimateEmbeds.face_strength (multiplies the face "
                    "adapter residual on every pass, in every CFG regime) — and "
                    "pose_strength for the pose latents.",
                    float(face_cfg_scale),
                )
        except (TypeError, ValueError):
            pass

        # Wan-Animate spec 3.1 (closed-loop critic): folded in here (not a
        # standalone node) because this class already receives pose_data's
        # iris_data — this run's own measured blendshapes — as its natural
        # input. Ported verbatim from the (now-removed) standalone critic
        # prototype: per-AU MAE, per-frame error curve, segment breakdown
        # matching WanVideoAnimateEmbeds.frame_window_size, worst-AU sort.
        # Off unless reference_expression_coeffs_json is wired — zero cost
        # otherwise. Never raises: a failure just leaves the report empty.
        critic_report_json = "{}"
        worst_aus_csv = ""
        overall_mae = 0.0
        ref_json = (reference_expression_coeffs_json or "").strip()
        if ref_json and ref_json != "{}":
            try:
                from .nodes_extras.expression_coeffs import _extract_blendshapes

                try:
                    ref_data = json.loads(ref_json)
                except json.JSONDecodeError:
                    ref_data = {}
                ref_names = ref_data.get("names") or [] if isinstance(ref_data, dict) else []
                ref_frames = ref_data.get("frames") or [] if isinstance(ref_data, dict) else []

                gen_bs_per_frame = [_extract_blendshapes(entry) for entry in iris_data]
                gen_names = set()
                for bs in gen_bs_per_frame:
                    gen_names.update(bs.keys())

                names = sorted(set(ref_names) | gen_names)
                n = min(len(ref_frames), len(gen_bs_per_frame))
                truncated = len(ref_frames) != len(gen_bs_per_frame)

                per_au_abs_err = {name: [] for name in names}
                per_frame_err = []
                for i in range(n):
                    ref_entry = ref_frames[i] if isinstance(ref_frames[i], dict) else {}
                    ref_bs = ref_entry.get("blendshapes", {}) if isinstance(ref_entry.get("blendshapes"), dict) else {}
                    gen_bs = gen_bs_per_frame[i]
                    frame_errs = []
                    for name in names:
                        e = abs(float(ref_bs.get(name, 0.0)) - float(gen_bs.get(name, 0.0)))
                        per_au_abs_err[name].append(e)
                        frame_errs.append(e)
                    per_frame_err.append(float(sum(frame_errs) / len(frame_errs)) if frame_errs else 0.0)

                per_au_mae = {
                    name: (float(sum(errs) / len(errs)) if errs else 0.0)
                    for name, errs in per_au_abs_err.items()
                }
                overall_mae = float(sum(per_au_mae.values()) / len(per_au_mae)) if per_au_mae else 0.0
                worst_aus = sorted(per_au_mae.items(), key=lambda kv: kv[1], reverse=True)[:max(1, int(top_k_aus))]

                seg_len = max(1, int(segment_length))
                segments = []
                for s0 in range(0, n, seg_len):
                    s1 = min(n, s0 + seg_len)
                    seg_errs = per_frame_err[s0:s1]
                    segments.append({
                        "start_frame": s0,
                        "end_frame": s1 - 1,
                        "n_frames": s1 - s0,
                        "mean_error": float(sum(seg_errs) / len(seg_errs)) if seg_errs else 0.0,
                    })
                segments_worst_first = sorted(segments, key=lambda s: s["mean_error"], reverse=True)

                report = {
                    "n_frames_compared": n,
                    "frame_count_mismatch": truncated,
                    "reference_n_frames": len(ref_frames),
                    "generated_n_frames": len(gen_bs_per_frame),
                    "overall_mae": overall_mae,
                    "per_au_mae": per_au_mae,
                    "worst_aus": [{"name": name, "mae": mae} for name, mae in worst_aus],
                    "per_frame_error": per_frame_err,
                    "segments": segments,
                    "segments_worst_first": segments_worst_first,
                }
                if truncated:
                    report["note"] = (
                        f"reference has {len(ref_frames)} frames, this run has "
                        f"{len(gen_bs_per_frame)} — compared the first {n} "
                        f"(frame-index truncation, no re-alignment attempted)."
                    )
                critic_report_json = json.dumps(report)
                worst_aus_csv = "\n".join(f"{name},{mae:.4f}" for name, mae in worst_aus)
            except Exception as _exc:                                    # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "reference_expression_coeffs_json critic failed (%s); critic outputs left empty.", _exc,
                )

        return (
            pose_images_tensor, face_video_out, float(face_cfg_scale),
            critic_report_json, worst_aus_csv, float(overall_mae),
        )


# ====================================================================
# Wan-Animate paper recommendation #4: face-quality gating.
# ====================================================================
class WanAnimateFaceQualityCheckV2:
    DESCRIPTION = (
        "Score each face crop on (a) Laplacian-variance sharpness and "
        "(b) eye-region brightness, then optionally repair bad frames by "
        "copying the previous good frame or by simple sharpening. Bad "
        "face conditioning frames cause the Wan-Animate face encoder to "
        "produce drifting / wrong-direction gaze (paper Sec. 4.3). "
        "Connect this BETWEEN Pose-and-Face-Detection (V2)'s `face_images` "
        "output and your downstream face-id encoder."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_images":           ("IMAGE", {"tooltip": "Per-frame 512x512 face crops (output of Pose and Face Detection V2)."}),
                "blur_threshold":        ("FLOAT", {"default": 50.0, "min": 0.0, "max": 5000.0, "step": 1.0, "tooltip": "Laplacian-variance threshold below which a frame is flagged as blurry. Typical sharp 512x512 frames score 100-1000; <50 indicates motion blur or out-of-focus."}),
                "min_eye_brightness":    ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Minimum mean luma of the eye-region strip (rows 30%-55%). Below this, eyes are likely closed or the frame is too dark for the encoder to read gaze."}),
                "auto_repair_bad_frames": ("BOOLEAN", {"default": True, "tooltip": "If true, repair frames flagged as bad. If false, just report stats."}),
                "repair_strategy":       (["copy_previous_good", "unsharp_mask", "skip"], {"default": "copy_previous_good", "tooltip": "copy_previous_good: replace with last good frame. unsharp_mask: deconvolve-style sharpening. skip: leave untouched but report."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("face_images_repaired", "good_frame_ratio", "report_json")
    OUTPUT_TOOLTIPS = (
        "Repaired face IMAGE batch (same shape as input).",
        "Fraction of frames that passed BOTH thresholds (0..1).",
        "JSON report: per-frame blur score, eye brightness, verdict, repair action.",
    )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess_V2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def _unsharp(self, frame_np):
        # Frame is float32 [0,1].
        u8 = (np.clip(frame_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        blurred = cv2.GaussianBlur(u8, (0, 0), sigmaX=1.5)
        sharp = cv2.addWeighted(u8, 1.5, blurred, -0.5, 0)
        return np.clip(sharp.astype(np.float32) / 255.0, 0.0, 1.0)

    def process(self, face_images, blur_threshold, min_eye_brightness,
                auto_repair_bad_frames, repair_strategy):
        if not isinstance(face_images, torch.Tensor) or face_images.ndim != 4 or face_images.shape[-1] != 3:
            raise ValueError(
                f"WanAnimateFaceQualityCheckV2: expected (B,H,W,3); got {tuple(getattr(face_images, 'shape', ()))}"
            )
        with torch.inference_mode():
            return self._process_impl(
                face_images, blur_threshold, min_eye_brightness,
                auto_repair_bad_frames, repair_strategy,
            )

    def _process_impl(self, face_images, blur_threshold, min_eye_brightness,
                auto_repair_bad_frames, repair_strategy):
        if hasattr(face_images, "detach"):
            arr = face_images.detach().cpu().numpy()
        else:
            arr = np.asarray(face_images)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(
                f"WanAnimateFaceQualityCheckV2: expected (B,H,W,3); got {arr.shape}"
            )
        B = arr.shape[0]
        report = []
        good_count = 0
        repaired = arr.copy().astype(np.float32)
        last_good_idx = -1

        for i in range(B):
            frame = repaired[i]
            blur = compute_frame_blur_score(frame)
            eye_lum = compute_eye_region_brightness(frame)
            blur_ok = blur >= float(blur_threshold)
            lum_ok = eye_lum >= float(min_eye_brightness)
            ok = blur_ok and lum_ok
            action = "none"
            if ok:
                good_count += 1
                last_good_idx = i
            elif auto_repair_bad_frames:
                if repair_strategy == "copy_previous_good" and last_good_idx >= 0:
                    repaired[i] = repaired[last_good_idx]
                    action = f"copied_from_frame_{last_good_idx}"
                elif repair_strategy == "unsharp_mask":
                    repaired[i] = self._unsharp(frame)
                    action = "unsharp_mask"
                else:
                    action = "skipped_no_prior_good_frame"
            report.append({
                "frame": int(i),
                "blur_score": round(blur, 2),
                "eye_brightness": round(eye_lum, 4),
                "blur_ok": bool(blur_ok),
                "brightness_ok": bool(lum_ok),
                "verdict": "ok" if ok else "bad",
                "action": action,
            })

        ratio = float(good_count) / float(max(1, B))
        report_json = json.dumps({
            "good_frame_ratio": round(ratio, 4),
            "blur_threshold": float(blur_threshold),
            "min_eye_brightness": float(min_eye_brightness),
            "frames": report,
        })
        return (torch.from_numpy(repaired.astype(np.float32)), ratio, report_json)


# ====================================================================
# Standalone Depth + Pose + Canny composer
# ====================================================================
class DepthPoseCannyCombinedV2:
    DESCRIPTION = (
        "Self-contained ControlNet preprocessor producing depth, pose, canny, "
        "normal, layout-combined preview, AND a weighted blended map.\n\n"
        "DEPTH backends (set via `depth_backend`):\n"
        "  - auto       : prefer external_depth_map -> any wired loader -> built_in_midas\n"
        "  - external   : require external_depth_map IMAGE input\n"
        "  - built_in_midas : MiDaS small via torch.hub (downloads ~80MB to torch hub cache, no extra node pack needed)\n"
        "  - damodel_v2     : kijai/ComfyUI-DepthAnythingV2 (models/depthanything/)\n"
        "  - da3            : PozzettiAndrea/ComfyUI-DepthAnythingV3 (models/depthanything3/) - delegates to V3 pack\n"
        "  - depthcrafter   : akatz-ai/ComfyUI-DepthCrafter-Nodes (models/depthcrafter/)\n"
        "  - depth_pro      : spacepxl/ComfyUI-Depth-Pro (models/depth/ml-depth-pro/)\n\n"
        "POSE source priority: external_pose_map > posemodel.\n\n"
        "NORMAL map: Sobel-from-depth (Lambertian-style RGB). No extra model.\n\n"
        "BLEND modes (research-backed, Wikipedia/W3C Compositing 1.0):\n"
        "  - none           : returns the depth_map\n"
        "  - weighted_avg   : per-channel sum normalised by total weight (perceptually balanced)\n"
        "  - screen         : 1 - prod(1 - layer_i*w_i)  (avoids highlight clipping, good for stacking depth+canny gradients)\n"
        "  - linear_dodge   : min(1, sum(layer_i*w_i))  (additive; sharpens edges; preferred for pose+canny per Fooocus/SDXL controlnet community)\n"
        "  - max            : per-pixel maximum across weighted layers (preserves strongest cue per pixel)\n"
        "  - multiply       : prod(layer_i^w_i)  (darkening; emphasises overlap)\n"
        "  - overlay        : combined multiply/screen S-curve on weighted_avg base\n"
        "  - channel_split  : R=depth, G=canny, B=pose (Fun-Control / IP-Adapter style multi-condition packing)\n\n"
        "OUTPUTS: depth_map, pose_map, canny_map, normal_map, combined_map (layout), blended_map (per blend_mode)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":             ("IMAGE", {"tooltip": "Input video frames (B,H,W,3) float32 [0,1]."}),
                "width":              ("INT", {"default": 832, "min": 64, "max": 4096, "tooltip": "Output canvas width."}),
                "height":             ("INT", {"default": 480, "min": 64, "max": 4096, "tooltip": "Output canvas height."}),
                "enable_depth":       ("BOOLEAN", {"default": True, "tooltip": "Run the depth pass. Requires at least ONE depth source wired."}),
                "enable_pose":        ("BOOLEAN", {"default": True, "tooltip": "Run the pose pass."}),
                "enable_canny":       ("BOOLEAN", {"default": True, "tooltip": "Run the canny pass."}),
                "canny_threshold1":   ("INT", {"default": 100, "min": 0, "max": 500, "tooltip": "Canny lower hysteresis threshold."}),
                "canny_threshold2":   ("INT", {"default": 200, "min": 0, "max": 500, "tooltip": "Canny upper hysteresis threshold."}),
                "canny_aperture":     ([3, 5, 7], {"default": 3, "tooltip": "Sobel aperture for Canny (odd: 3/5/7)."}),
                "depth_colorize":     ("BOOLEAN", {"default": False, "tooltip": "If true, colorize grayscale depth with INFERNO colormap. Skipped when external_depth_map is already RGB."}),
                "depth_invert":       ("BOOLEAN", {"default": False, "tooltip": "Invert depth (1 - depth). Use when source produces 'far = bright' but you want 'near = bright' (typical ControlNet expectation)."}),
                "pose_detection_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "YOLO confidence threshold (only used when posemodel is wired)."}),
                "pose_draw_threshold":      ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Per-keypoint score threshold for drawing the skeleton."}),
                "combined_layout":    (["horizontal_3", "vertical_3", "grid_2x2", "depth_only", "pose_only", "canny_only"], {"default": "horizontal_3", "tooltip": "Layout for the combined output. grid_2x2 = depth | pose // canny | original."}),
                # ---- Task 2: self-contained additions (appended at end so saved workflows keep their positional values) ----
                "depth_backend":      (["auto", "external", "built_in_midas", "damodel_v2", "da3", "depthcrafter", "depth_pro"], {"default": "auto", "tooltip": "Which depth backend to use. 'auto' tries: external_depth_map -> any wired loader -> built_in_midas. 'built_in_midas' makes the node fully self-contained (downloads MiDaS small via torch.hub on first use, ~80MB)."}),
                "enable_normal":      ("BOOLEAN", {"default": True, "tooltip": "Compute Sobel-from-depth NORMAL map. No model required (uses depth pass output)."}),
                "normal_strength":    ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Scales the Sobel gradients before normalisation. Higher = stronger normal contrast."}),
                "blend_mode":         (["none", "weighted_avg", "screen", "linear_dodge", "max", "multiply", "overlay", "channel_split"], {"default": "weighted_avg", "tooltip": "How to combine depth+pose+canny+normal into blended_map. linear_dodge=additive (sharp), screen=highlight-safe, channel_split=Fun-Control (R=depth/G=canny/B=pose)."}),
                "depth_weight":       ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Weight of depth in blended_map."}),
                "pose_weight":        ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Weight of pose in blended_map."}),
                "canny_weight":       ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Weight of canny in blended_map."}),
                "normal_weight":      ("FLOAT", {"default": 0.5, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Weight of normal map in blended_map."}),
            },
            "optional": {
                "external_depth_map":  ("IMAGE", {"tooltip": "Pre-computed depth IMAGE batch from ANY upstream node. Highest priority."}),
                "damodel_v2":          ("DAMODEL", {"tooltip": "DepthAnything V2 model bundle from kijai/ComfyUI-DepthAnythingV2 (DownloadAndLoadDepthAnythingV2Model). Models: ComfyUI/models/depthanything/."}),
                "da3_model":           ("DA3MODEL", {"tooltip": "DepthAnything V3 config bundle from PozzettiAndrea/ComfyUI-DepthAnythingV3. Use the V3 pack's Inference node and feed its IMAGE output into external_depth_map. Models: ComfyUI/models/depthanything3/."}),
                "depthcrafter_model":  ("DEPTHCRAFTER_MODEL", {"tooltip": "DepthCrafter bundle from akatz-ai/ComfyUI-DepthCrafter-Nodes. Temporally consistent video depth. Models: ComfyUI/models/depthcrafter/."}),
                "depth_pro_model":     ("DEPTH_PRO_MODEL", {"tooltip": "Depth-Pro bundle from spacepxl/ComfyUI-Depth-Pro. Metric depth. Models: ComfyUI/models/depth/ml-depth-pro/."}),
                "posemodel":           ("POSEMODEL", {"tooltip": "From ONNX Detection Model Loader (V2) or animal-pose loader. Used if enable_pose=True AND no external_pose_map wired."}),
                "external_pose_map":   ("IMAGE", {"tooltip": "Pre-rendered pose map from any upstream node (e.g. Fannovel16/comfyui_controlnet_aux DWPose / OpenPose / AnimalPose). Highest priority for pose."}),
                "depthcrafter_steps":      ("INT", {"default": 5, "min": 1, "max": 100, "tooltip": "DepthCrafter only: diffusion inference steps."}),
                "depthcrafter_guidance":   ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "DepthCrafter only: classifier-free guidance."}),
                "depthcrafter_window":     ("INT", {"default": 110, "min": 1, "max": 200, "tooltip": "DepthCrafter only: temporal window size."}),
                "depthcrafter_overlap":    ("INT", {"default": 25, "min": 0, "max": 100, "tooltip": "DepthCrafter only: window overlap."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("depth_map", "pose_map", "canny_map", "normal_map", "combined_map", "blended_map")
    OUTPUT_TOOLTIPS = (
        "Per-frame depth IMAGE batch (3-channel, height x width).",
        "Per-frame pose IMAGE batch (3-channel, on black canvas).",
        "Per-frame canny edge IMAGE batch (3-channel grayscale).",
        "Per-frame normal map (RGB-encoded surface normals from Sobel-of-depth).",
        "Side-by-side combined preview per `combined_layout`.",
        "Weighted blend of {depth, pose, canny, normal} per `blend_mode` and per-channel weights.",
    )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess_V2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    # ---------- helpers ----------
    @staticmethod
    def _to_np(images):
        if hasattr(images, "detach"):
            return images.detach().cpu().numpy().astype(np.float32)
        return np.asarray(images, dtype=np.float32)

    @staticmethod
    def _resize_batch(arr, target_w, target_h):
        # arr: (B,H,W,3) float32 [0,1]
        if arr.shape[1] == target_h and arr.shape[2] == target_w:
            return arr
        out = np.zeros((arr.shape[0], target_h, target_w, arr.shape[3]), dtype=np.float32)
        for i in range(arr.shape[0]):
            out[i] = cv2.resize(arr[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return out

    def _depth_pass(self, images_np, target_w, target_h, colorize, invert,
                    external_depth_map, damodel_v2, da3_model,
                    depthcrafter_model, depth_pro_model,
                    dc_steps, dc_guidance, dc_window, dc_overlap,
                    depth_backend="auto"):
        """Run depth using the selected backend.

        depth_backend:
          - auto       : external_depth_map -> any wired loader -> built_in_midas (fully self-contained fallback)
          - external   : requires external_depth_map
          - built_in_midas : torch.hub MiDaS small (downloads on first use)
          - damodel_v2 / da3 / depthcrafter / depth_pro : require the matching loader wired
        Always returns (B, target_h, target_w, 3) float32 in [0, 1].
        """
        depth_2d = None
        depth_rgb = None

        backend = (depth_backend or "auto").lower()

        def _from_external():
            ext = self._to_np(external_depth_map)
            if ext.ndim != 4 or ext.shape[-1] != 3:
                raise ValueError(
                    f"external_depth_map must be IMAGE (B,H,W,3); got {ext.shape}"
                )
            return self._resize_batch(ext, target_w, target_h)

        if backend == "external":
            if external_depth_map is None:
                raise RuntimeError("depth_backend='external' but external_depth_map is not wired.")
            depth_rgb = _from_external()
        elif backend == "built_in_midas":
            depth_2d = self._infer_midas_small(images_np)
        elif backend == "damodel_v2":
            if damodel_v2 is None:
                raise RuntimeError("depth_backend='damodel_v2' but damodel_v2 is not wired.")
            depth_2d = self._infer_damodel_v2(damodel_v2, images_np)
        elif backend == "da3":
            if da3_model is None:
                raise RuntimeError("depth_backend='da3' but da3_model is not wired.")
            depth_2d = self._infer_da3model(da3_model, images_np)
        elif backend == "depthcrafter":
            if depthcrafter_model is None:
                raise RuntimeError("depth_backend='depthcrafter' but depthcrafter_model is not wired.")
            depth_2d = self._infer_depthcrafter(
                depthcrafter_model, images_np,
                int(dc_steps), float(dc_guidance), int(dc_window), int(dc_overlap),
            )
        elif backend == "depth_pro":
            if depth_pro_model is None:
                raise RuntimeError("depth_backend='depth_pro' but depth_pro_model is not wired.")
            depth_2d = self._infer_depth_pro(depth_pro_model, images_np)
        else:
            # auto: external -> any loader -> built_in_midas
            if external_depth_map is not None:
                depth_rgb = _from_external()
            elif damodel_v2 is not None:
                depth_2d = self._infer_damodel_v2(damodel_v2, images_np)
            elif da3_model is not None:
                depth_2d = self._infer_da3model(da3_model, images_np)
            elif depthcrafter_model is not None:
                depth_2d = self._infer_depthcrafter(
                    depthcrafter_model, images_np,
                    int(dc_steps), float(dc_guidance), int(dc_window), int(dc_overlap),
                )
            elif depth_pro_model is not None:
                depth_2d = self._infer_depth_pro(depth_pro_model, images_np)
            else:
                # Self-contained fallback
                depth_2d = self._infer_midas_small(images_np)

        if depth_2d is not None:
            if depth_2d.shape[1:] != (target_h, target_w):
                resized = np.zeros((depth_2d.shape[0], target_h, target_w), dtype=np.float32)
                for i in range(depth_2d.shape[0]):
                    resized[i] = cv2.resize(
                        depth_2d[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR
                    )
                depth_2d = resized
            if invert:
                depth_2d = 1.0 - depth_2d
            if colorize:
                out = np.zeros((depth_2d.shape[0], target_h, target_w, 3), dtype=np.float32)
                for i in range(depth_2d.shape[0]):
                    u8 = (np.clip(depth_2d[i], 0.0, 1.0) * 255.0).astype(np.uint8)
                    col = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
                    col_rgb = cv2.cvtColor(col, cv2.COLOR_BGR2RGB)
                    out[i] = col_rgb.astype(np.float32) / 255.0
                return out
            return np.repeat(depth_2d[..., None], 3, axis=-1).astype(np.float32)

        if invert:
            depth_rgb = 1.0 - depth_rgb
        return depth_rgb.astype(np.float32)

    # ---------- per-backend inference adapters ----------
    @staticmethod
    def _normalize_per_frame(depth_np):
        """Per-frame min-max normalize (B,H,W) -> [0,1]."""
        out = np.zeros_like(depth_np, dtype=np.float32)
        for i in range(depth_np.shape[0]):
            f = depth_np[i].astype(np.float32)
            fmin, fmax = float(f.min()), float(f.max())
            if fmax - fmin > 1e-6:
                out[i] = (f - fmin) / (fmax - fmin)
        return out

    def _infer_damodel_v2(self, damodel, images_np):
        """Mirror kijai/ComfyUI-DepthAnythingV2 inference loop."""
        import torch.nn.functional as F
        from torchvision.transforms import Normalize
        try:
            import comfy.model_management as mm
        except ImportError:
            mm = None

        device = mm.get_torch_device() if mm else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        offload_device = mm.unet_offload_device() if mm else torch.device("cpu")
        model = damodel["model"]
        dtype = damodel.get("dtype", torch.float32)
        is_metric = damodel.get("is_metric", False)

        images_t = torch.from_numpy(images_np).float()
        B, H, W, _ = images_t.shape
        images_t = images_t.permute(0, 3, 1, 2)
        new_W = W - (W % 14)
        new_H = H - (H % 14)
        if new_W != W or new_H != H:
            images_t = F.interpolate(images_t, size=(new_H, new_W), mode="bilinear")
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images_t = normalize(images_t)

        model.to(device)
        autocast_ok = (dtype != torch.float32) and (device.type == "cuda")
        out = []
        with torch.inference_mode():
            if autocast_ok:
                with torch.autocast("cuda", dtype=dtype):
                    for img in images_t:
                        d = model(img.unsqueeze(0).to(device))
                        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
                        out.append(d.detach().float().cpu())
            else:
                for img in images_t:
                    d = model(img.unsqueeze(0).to(device))
                    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
                    out.append(d.detach().float().cpu())
        try:
            model.to(offload_device)
            if mm:
                mm.soft_empty_cache()
        except Exception:
            pass
        depth = torch.cat(out, dim=0).numpy()
        if depth.ndim == 4:
            depth = depth.squeeze(1)
        if is_metric:
            depth = 1.0 - depth
        if depth.shape[1:] != (H, W):
            resized = np.zeros((B, H, W), dtype=np.float32)
            for i in range(B):
                resized[i] = cv2.resize(depth[i], (W, H), interpolation=cv2.INTER_LINEAR)
            depth = resized
        return depth.astype(np.float32)

    def _infer_da3model(self, da3_config, images_np):
        """DA3 inference is tightly coupled to V3 internals; route via V3 pack."""
        raise RuntimeError(
            "DA3MODEL is a JSON config bundle. Please run the V3 pack's own "
            "inference node (DepthAnythingV3 Inference) on the V3 pack, then "
            "feed its IMAGE output into `external_depth_map` here. We do not "
            "duplicate V3 inference internals (they are version-dependent)."
        )

    def _infer_depthcrafter(self, dc_model, images_np, steps, guidance, window, overlap):
        """Mirror akatz-ai/ComfyUI-DepthCrafter-Nodes inference logic."""
        device = dc_model.get("device") if isinstance(dc_model, dict) else None
        pipe = dc_model["pipe"] if isinstance(dc_model, dict) else dc_model
        if device is None:
            try:
                import comfy.model_management as mm
                device = mm.get_torch_device()
            except Exception:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        images_t = torch.from_numpy(images_np).float()
        B, H, W, _ = images_t.shape
        new_W = max(64, (round(W / 64) * 64) or 64)
        new_H = max(64, (round(H / 64) * 64) or 64)
        if new_W != W or new_H != H:
            x = images_t.permute(0, 3, 1, 2)
            x = torch.nn.functional.interpolate(
                x, size=(new_H, new_W), mode="bilinear", align_corners=False
            )
            images_t = x.permute(0, 2, 3, 1)
        x = images_t.permute(0, 3, 1, 2).to(device=device, dtype=torch.float16)
        x = torch.clamp(x, 0.0, 1.0)
        with torch.inference_mode():
            result = pipe(
                x,
                height=new_H,
                width=new_W,
                output_type="pt",
                guidance_scale=float(guidance),
                num_inference_steps=int(steps),
                window_size=int(window),
                overlap=int(overlap),
                track_time=False,
            )
        res = result.frames[0]
        depth = res.detach().float().cpu().numpy()
        if depth.ndim == 4:
            depth = depth.squeeze(-1) if depth.shape[-1] == 1 else depth.mean(axis=-1)
        depth = self._normalize_per_frame(depth)
        if depth.shape[1:] != (H, W):
            resized = np.zeros((depth.shape[0], H, W), dtype=np.float32)
            for i in range(depth.shape[0]):
                resized[i] = cv2.resize(depth[i], (W, H), interpolation=cv2.INTER_LINEAR)
            depth = resized
        return depth.astype(np.float32)

    def _infer_depth_pro(self, dp_model, images_np):
        """Mirror spacepxl/ComfyUI-Depth-Pro inference (relative depth)."""
        from torchvision.transforms import Normalize
        model = dp_model["model"]
        device = dp_model.get("device")
        dtype = dp_model.get("dtype", torch.float32)
        if device is None:
            try:
                import comfy.model_management as mm
                device = mm.get_torch_device()
            except Exception:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rgb = torch.from_numpy(images_np).float().movedim(-1, 1)
        transform = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        depth_list = []
        with torch.inference_mode():
            for i in range(rgb.size(0)):
                rgb_i = rgb[i].unsqueeze(0).to(device, dtype=dtype)
                rgb_i = transform(rgb_i)
                pred = model.infer(rgb_i, f_px=None)
                d = pred["depth"].detach().float().cpu().numpy()
                depth_list.append(d)
        depth = np.stack(depth_list, axis=0)
        while depth.ndim > 3:
            depth = depth.squeeze(1) if depth.shape[1] == 1 else depth.mean(axis=1)
        # Depth-Pro returns metric depth (meters). Convert to relative [0,1].
        depth = 1.0 / (1.0 + depth)
        depth = self._normalize_per_frame(depth)
        return depth.astype(np.float32)

    # ---------- built-in MiDaS (self-contained depth) ----------
    _midas_cache = {"model": None, "transform": None, "device": None}

    @classmethod
    def _infer_midas_small(cls, images_np):
        """MiDaS small via torch.hub. Self-contained depth fallback.

        First call downloads ~80MB to torch hub cache (HOME/.cache/torch/hub/).
        Subsequent calls reuse the cached model. Returns (B,H,W) float32 [0,1].
        """
        try:
            try:
                import comfy.model_management as mm
                device = mm.get_torch_device()
            except Exception:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if cls._midas_cache["model"] is None or cls._midas_cache["device"] != device:
                midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
                midas.to(device).eval()
                transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                cls._midas_cache["model"] = midas
                cls._midas_cache["transform"] = transforms.small_transform
                cls._midas_cache["device"] = device

            midas = cls._midas_cache["model"]
            transform = cls._midas_cache["transform"]

            B, H, W, _ = images_np.shape
            out = np.zeros((B, H, W), dtype=np.float32)
            with torch.inference_mode():
                for i in range(B):
                    u8 = (np.clip(images_np[i], 0.0, 1.0) * 255.0).astype(np.uint8)
                    inp = transform(u8).to(device)
                    pred = midas(inp)
                    pred = torch.nn.functional.interpolate(
                        pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
                    ).squeeze(1)
                    d = pred[0].detach().float().cpu().numpy()
                    dmin, dmax = float(d.min()), float(d.max())
                    if dmax - dmin > 1e-6:
                        d = (d - dmin) / (dmax - dmin)
                    else:
                        d = np.zeros_like(d)
                    out[i] = d.astype(np.float32)
            return out
        except Exception as e:
            raise RuntimeError(
                "DepthPoseCannyCombinedV2 built_in_midas backend failed. "
                "Cause: " + str(e) + "\n"
                "Possible fixes: (a) ensure internet is reachable for first-time torch.hub download, "
                "(b) install timm via pip (MiDaS small needs it), "
                "(c) wire an external_depth_map instead and set depth_backend='external'."
            )

    # ---------- normal map (Sobel-from-depth, no extra model) ----------
    @staticmethod
    def _normal_from_depth(depth_2d, strength=1.0):
        """Compute per-frame RGB normal map from grayscale depth.

        depth_2d: (B,H,W) float32 [0,1]. Returns (B,H,W,3) float32 [0,1].
        Encoding: R = (nx+1)/2, G = (ny+1)/2, B = (nz+1)/2 (standard tangent-space).
        """
        B, H, W = depth_2d.shape
        out = np.zeros((B, H, W, 3), dtype=np.float32)
        for i in range(B):
            d = depth_2d[i].astype(np.float32)
            # Scharr is more accurate than Sobel for small kernels
            gx = cv2.Scharr(d, cv2.CV_32F, 1, 0) * float(strength)
            gy = cv2.Scharr(d, cv2.CV_32F, 0, 1) * float(strength)
            # Normal vector: (-dz/dx, -dz/dy, 1) then normalise
            nx = -gx
            ny = -gy
            nz = np.ones_like(nx)
            norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
            nx /= norm
            ny /= norm
            nz /= norm
            # Encode to [0,1]
            out[i, ..., 0] = np.clip((nx + 1.0) * 0.5, 0.0, 1.0)
            out[i, ..., 1] = np.clip((ny + 1.0) * 0.5, 0.0, 1.0)
            out[i, ..., 2] = np.clip((nz + 1.0) * 0.5, 0.0, 1.0)
        return out

    def _normal_pass(self, depth_out, target_w, target_h, strength):
        """Convert depth_out (B,H,W,3) RGB depth -> (B,H,W,3) normal map."""
        # depth_out is RGB but for normals we need a scalar field — use channel mean.
        depth_2d = depth_out.mean(axis=-1).astype(np.float32)
        # Light blur to smooth normals (depth is noisy at edges)
        for i in range(depth_2d.shape[0]):
            depth_2d[i] = cv2.GaussianBlur(depth_2d[i], (0, 0), sigmaX=1.2)
        n = self._normal_from_depth(depth_2d, strength=float(strength))
        if n.shape[1] != target_h or n.shape[2] != target_w:
            out = np.zeros((n.shape[0], target_h, target_w, 3), dtype=np.float32)
            for i in range(n.shape[0]):
                out[i] = cv2.resize(n[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return out
        return n

    # ---------- blend (research-backed, Wikipedia/W3C Compositing 1.0) ----------
    @staticmethod
    def _blend_pass(depth, pose, canny, normal, mode, w_depth, w_pose, w_canny, w_normal):
        """Combine per-channel maps. All inputs (B,H,W,3) float32 [0,1]."""
        d = (depth * float(w_depth))
        p = (pose * float(w_pose))
        c = (canny * float(w_canny))
        n = (normal * float(w_normal))
        eps = 1e-6
        m = (mode or "weighted_avg").lower()

        if m == "none":
            return depth.astype(np.float32)
        if m == "channel_split":
            # Fun-Control style: R=depth(gray), G=canny(gray), B=pose(gray)
            out = np.zeros_like(depth, dtype=np.float32)
            out[..., 0] = np.clip(depth.mean(axis=-1) * float(w_depth), 0.0, 1.0)
            out[..., 1] = np.clip(canny.mean(axis=-1) * float(w_canny), 0.0, 1.0)
            out[..., 2] = np.clip(pose.mean(axis=-1) * float(w_pose), 0.0, 1.0)
            return out
        if m == "linear_dodge":
            return np.clip(d + p + c + n, 0.0, 1.0).astype(np.float32)
        if m == "max":
            return np.maximum.reduce([
                np.clip(d, 0.0, 1.0),
                np.clip(p, 0.0, 1.0),
                np.clip(c, 0.0, 1.0),
                np.clip(n, 0.0, 1.0),
            ]).astype(np.float32)
        if m == "screen":
            # 1 - prod(1 - x_i)
            inv = (1.0 - np.clip(d, 0.0, 1.0)) \
                * (1.0 - np.clip(p, 0.0, 1.0)) \
                * (1.0 - np.clip(c, 0.0, 1.0)) \
                * (1.0 - np.clip(n, 0.0, 1.0))
            return np.clip(1.0 - inv, 0.0, 1.0).astype(np.float32)
        if m == "multiply":
            # Use weights as exponents: x^w
            r = (np.clip(depth, eps, 1.0) ** float(w_depth)) \
              * (np.clip(pose,  eps, 1.0) ** float(w_pose)) \
              * (np.clip(canny, eps, 1.0) ** float(w_canny)) \
              * (np.clip(normal,eps, 1.0) ** float(w_normal))
            return np.clip(r, 0.0, 1.0).astype(np.float32)
        if m == "overlay":
            # Base = weighted_avg of (depth, pose, canny); top = normal
            base_w = max(eps, float(w_depth) + float(w_pose) + float(w_canny))
            base = np.clip((d + p + c) / base_w, 0.0, 1.0)
            top = np.clip(normal, 0.0, 1.0)
            lo = 2.0 * base * top
            hi = 1.0 - 2.0 * (1.0 - base) * (1.0 - top)
            r = np.where(base < 0.5, lo, hi)
            # Mix in normal weight as opacity
            alpha = float(np.clip(w_normal, 0.0, 1.0))
            r = (1.0 - alpha) * base + alpha * r
            return np.clip(r, 0.0, 1.0).astype(np.float32)
        # default: weighted_avg
        total = max(eps, float(w_depth) + float(w_pose) + float(w_canny) + float(w_normal))
        return np.clip((d + p + c + n) / total, 0.0, 1.0).astype(np.float32)

    def _canny_pass(self, images_np, target_w, target_h, t1, t2, aperture):
        B = images_np.shape[0]
        out = np.zeros((B, target_h, target_w, 3), dtype=np.float32)
        for i in range(B):
            u8 = (np.clip(images_np[i], 0.0, 1.0) * 255.0).astype(np.uint8)
            gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, int(t1), int(t2), apertureSize=int(aperture))
            edges_rgb = np.repeat(edges[..., None], 3, axis=-1)
            if edges_rgb.shape[:2] != (target_h, target_w):
                edges_rgb = cv2.resize(
                    edges_rgb, (target_w, target_h), interpolation=cv2.INTER_NEAREST
                )
            out[i] = edges_rgb.astype(np.float32) / 255.0
        return out

    def _pose_pass(self, images_np, posemodel, target_w, target_h,
                   detection_threshold, draw_threshold, external_pose_map):
        B, H, W, _ = images_np.shape
        # External pose map takes priority.
        if external_pose_map is not None:
            ext_np = self._to_np(external_pose_map)
            return self._resize_batch(ext_np, target_w, target_h)

        if posemodel is None:
            # Return a black canvas — caller already validated enable_pose.
            return np.zeros((B, target_h, target_w, 3), dtype=np.float32)

        # Render pose using YOLO + ViTPose, then draw onto target canvas.
        detector = posemodel["yolo"]
        pose_model = posemodel["vitpose"]
        if hasattr(detector, "threshold_conf"):
            detector.threshold_conf = float(detection_threshold)

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution = (256, 192)
        rescale = 1.25
        shape = np.array([H, W])[None]

        pose_canvases = []
        for img in _IC.track(
            images_np, B, "DepthPoseCannyCombined: pose render"
        ):
            detections = detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None], shape
            )[0]
            if isinstance(detections, list) and len(detections) > 0 and isinstance(detections[0], dict):
                bbox = detections[0]["bbox"]
            else:
                bbox = None
            if bbox is None or len(bbox) < 5 or bbox[4] <= 0:
                bbox_use = np.array([0, 0, W, H, 1.0], dtype=np.float32)
            else:
                bbox_use = bbox

            center, scale = bbox_from_detector(bbox_use, input_resolution, rescale=rescale)
            img_crop = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]
            img_norm = (img_crop - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)
            kp2ds = pose_model(
                img_norm[None], np.array(center)[None], np.array(scale)[None]
            )

            metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)
            meta = metas[0]
            aa = AAPoseMeta.from_humanapi_meta(meta)
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            try:
                draw_aapose_by_meta_new(
                    canvas, aa,
                    body_stick_width=-1, hand_stick_width=-1,
                    draw_head=True,
                    pose_draw_threshold=float(draw_threshold),
                )
            except TypeError:
                # Older signature without pose_draw_threshold
                draw_aapose_by_meta_new(
                    canvas, aa,
                    body_stick_width=-1, hand_stick_width=-1,
                    draw_head=True,
                )
            pose_canvases.append(canvas.astype(np.float32) / 255.0)

        try:
            detector.cleanup()
        except Exception:
            pass
        try:
            pose_model.cleanup()
        except Exception:
            pass

        return np.stack(pose_canvases, 0)

    def _compose(self, depth, pose, canny, original, layout, target_w, target_h):
        B = original.shape[0]
        zeros = np.zeros((B, target_h, target_w, 3), dtype=np.float32)
        d = depth if depth is not None else zeros
        p = pose if pose is not None else zeros
        c = canny if canny is not None else zeros

        if layout == "depth_only":
            return d
        if layout == "pose_only":
            return p
        if layout == "canny_only":
            return c
        if layout == "horizontal_3":
            return np.concatenate([d, p, c], axis=2)
        if layout == "vertical_3":
            return np.concatenate([d, p, c], axis=1)
        if layout == "grid_2x2":
            top = np.concatenate([d, p], axis=2)
            bot = np.concatenate([c, original], axis=2)
            return np.concatenate([top, bot], axis=1)
        return d

    def process(
        self, images, width, height,
        enable_depth, enable_pose, enable_canny,
        canny_threshold1, canny_threshold2, canny_aperture,
        depth_colorize, depth_invert,
        pose_detection_threshold, pose_draw_threshold,
        combined_layout,
        depth_backend="auto", enable_normal=True, normal_strength=1.0,
        blend_mode="weighted_avg",
        depth_weight=1.0, pose_weight=1.0, canny_weight=1.0, normal_weight=0.5,
        external_depth_map=None,
        damodel_v2=None, da3_model=None,
        depthcrafter_model=None, depth_pro_model=None,
        posemodel=None, external_pose_map=None,
        depthcrafter_steps=5, depthcrafter_guidance=1.0,
        depthcrafter_window=110, depthcrafter_overlap=25,
    ):
        if isinstance(images, torch.Tensor):
            if images.ndim != 4 or images.shape[-1] != 3:
                raise ValueError(
                    f"DepthPoseCannyCombinedV2: expected (B,H,W,3); got {tuple(images.shape)}"
                )
        with torch.inference_mode():
            return self._process_impl(
                images, width, height,
                enable_depth, enable_pose, enable_canny,
                canny_threshold1, canny_threshold2, canny_aperture,
                depth_colorize, depth_invert,
                pose_detection_threshold, pose_draw_threshold,
                combined_layout,
                depth_backend, enable_normal, normal_strength,
                blend_mode,
                depth_weight, pose_weight, canny_weight, normal_weight,
                external_depth_map,
                damodel_v2, da3_model,
                depthcrafter_model, depth_pro_model,
                posemodel, external_pose_map,
                depthcrafter_steps, depthcrafter_guidance,
                depthcrafter_window, depthcrafter_overlap,
            )

    def _process_impl(
        self, images, width, height,
        enable_depth, enable_pose, enable_canny,
        canny_threshold1, canny_threshold2, canny_aperture,
        depth_colorize, depth_invert,
        pose_detection_threshold, pose_draw_threshold,
        combined_layout,
        depth_backend="auto", enable_normal=True, normal_strength=1.0,
        blend_mode="weighted_avg",
        depth_weight=1.0, pose_weight=1.0, canny_weight=1.0, normal_weight=0.5,
        external_depth_map=None,
        damodel_v2=None, da3_model=None,
        depthcrafter_model=None, depth_pro_model=None,
        posemodel=None, external_pose_map=None,
        depthcrafter_steps=5, depthcrafter_guidance=1.0,
        depthcrafter_window=110, depthcrafter_overlap=25,
    ):
        images_np = self._to_np(images)
        if images_np.ndim != 4 or images_np.shape[-1] != 3:
            raise ValueError(
                f"DepthPoseCannyCombinedV2: expected (B,H,W,3); got {images_np.shape}"
            )
        B = images_np.shape[0]
        target_w, target_h = int(width), int(height)
        original_resized = self._resize_batch(images_np, target_w, target_h)

        depth_out = (
            self._depth_pass(
                images_np, target_w, target_h,
                bool(depth_colorize), bool(depth_invert),
                external_depth_map, damodel_v2, da3_model,
                depthcrafter_model, depth_pro_model,
                depthcrafter_steps, depthcrafter_guidance,
                depthcrafter_window, depthcrafter_overlap,
                depth_backend=str(depth_backend),
            ) if enable_depth else
            np.zeros((B, target_h, target_w, 3), dtype=np.float32)
        )
        pose_out = (
            self._pose_pass(
                images_np, posemodel, target_w, target_h,
                pose_detection_threshold, pose_draw_threshold, external_pose_map,
            ) if enable_pose else
            np.zeros((B, target_h, target_w, 3), dtype=np.float32)
        )
        canny_out = (
            self._canny_pass(
                images_np, target_w, target_h,
                canny_threshold1, canny_threshold2, canny_aperture,
            ) if enable_canny else
            np.zeros((B, target_h, target_w, 3), dtype=np.float32)
        )
        normal_out = (
            self._normal_pass(depth_out, target_w, target_h, float(normal_strength))
            if (enable_normal and enable_depth) else
            np.zeros((B, target_h, target_w, 3), dtype=np.float32)
        )

        combined = self._compose(
            depth_out, pose_out, canny_out, original_resized,
            combined_layout, target_w, target_h,
        )

        blended = self._blend_pass(
            depth_out, pose_out, canny_out, normal_out,
            str(blend_mode),
            float(depth_weight), float(pose_weight),
            float(canny_weight), float(normal_weight),
        )

        return (
            torch.from_numpy(depth_out.astype(np.float32)),
            torch.from_numpy(pose_out.astype(np.float32)),
            torch.from_numpy(canny_out.astype(np.float32)),
            torch.from_numpy(normal_out.astype(np.float32)),
            torch.from_numpy(combined.astype(np.float32)),
            torch.from_numpy(blended.astype(np.float32)),
        )


# =====================================================================
# KANIBUS gap nodes — derived eye-feature series from PoseAndFaceDetectionV2.
# All three consume the POSEDATA bundle (`pose_metas_original` carries
# normalized 91-pt face keypoints; `iris_data` carries per-frame iris dicts
# already in screen-space). Outputs are JSON-encoded so downstream graph
# nodes can consume them with ordinary STRING ports.
# =====================================================================


def _eye_aspect_ratio(face_kps_norm: np.ndarray, eye_idx: List[int],
                      W: int, H: int) -> float:
    """Standard Soukupová & Cech (2016) Eye-Aspect-Ratio over a 6-point eye
    contour. Returns NaN if any landmark is missing/zero-confidence (the
    pose_threshold gate in PoseAndFaceDetectionV2 may zero out kps).

    face_kps_norm : (91, 2 or 3) normalized array in [0,1].
    eye_idx       : 6 indices in the same order used elsewhere
                    (outer, upper_outer, upper_inner, inner,
                     lower_inner, lower_outer).
    """
    if face_kps_norm is None or face_kps_norm.ndim < 2:
        return float('nan')
    pts = face_kps_norm[eye_idx, :2].astype(np.float64)
    # Convert to pixels so EAR is a unit-less ratio robust to image AR.
    pts[:, 0] *= float(W)
    pts[:, 1] *= float(H)
    if (pts == 0).all(axis=1).any():
        return float('nan')
    p1, p2, p3, p4, p5, p6 = pts
    v1 = float(np.linalg.norm(p2 - p6))
    v2 = float(np.linalg.norm(p3 - p5))
    h  = float(np.linalg.norm(p1 - p4))
    if h < 1e-6:
        return float('nan')
    return (v1 + v2) / (2.0 * h)


def _one_euro_filter(series: List[float], freq_hz: float,
                     min_cutoff: float = 1.0, beta: float = 0.05,
                     d_cutoff: float = 1.0) -> List[float]:
    """OneEuro filter (Casiez, Roussel & Vogel, 2012). Adaptive low-pass:
    higher cutoff under fast motion (preserves saccade peaks), lower cutoff
    when still (kills jitter). Implemented inline so KANIBUS nodes don't
    drag in an extra dependency.

    NaN inputs are passed through unchanged so that downstream blink/saccade
    masks reflect the same missing-data periods.
    """
    if not series:
        return []
    out: List[float] = [float('nan')] * len(series)
    last_x: Optional[float] = None
    last_dx: float = 0.0
    last_t: float = 0.0
    if freq_hz <= 0:
        freq_hz = 30.0
    dt = 1.0 / float(freq_hz)
    for i, x in enumerate(series):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            out[i] = float('nan')
            continue
        if last_x is None:
            out[i] = float(x)
            last_x = float(x)
            last_dx = 0.0
            last_t = i * dt
            continue
        t = i * dt
        d = (float(x) - last_x) / max(dt, 1e-6)
        # Smooth derivative
        alpha_d = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * d_cutoff)) / max(dt, 1e-6))
        last_dx = alpha_d * d + (1.0 - alpha_d) * last_dx
        cutoff = min_cutoff + beta * abs(last_dx)
        alpha = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) / max(dt, 1e-6))
        smoothed = alpha * float(x) + (1.0 - alpha) * last_x
        out[i] = smoothed
        last_x = smoothed
        last_t = t
    return out


def _coerce_pose_data(pose_data: Any) -> Tuple[List[Dict[str, Any]],
                                                List[Dict[str, Any]],
                                                Tuple[int, int]]:
    """Pull the (face_metas, iris_seq, source_size) triple from a POSEDATA
    bundle as emitted by PoseAndFaceDetectionV2. Accepts the dict form
    only; raises a clear error otherwise so users see exactly which port
    they wired wrong instead of an opaque KeyError downstream.
    """
    if not isinstance(pose_data, dict):
        raise ValueError(
            "KANIBUS nodes expect the POSEDATA dict from "
            "PoseAndFaceDetectionV2; got %s." % type(pose_data).__name__
        )
    metas = pose_data.get('pose_metas_original') or pose_data.get('pose_metas')
    if not metas:
        raise ValueError("POSEDATA has no 'pose_metas_original' / 'pose_metas'.")
    iris_seq = pose_data.get('iris_data', []) or []
    src = pose_data.get('source_size') or pose_data.get('target_size') or (1, 1)
    H, W = int(src[0]), int(src[1])
    return list(metas), list(iris_seq), (H, W)


class EARBlinkDetectorC2C:
    """Detect blinks per frame using the Eye-Aspect-Ratio (EAR) of the
    6-point eye contour produced by PoseAndFaceDetectionV2.

    A blink is reported when *both* eyes' EAR drops below ``threshold`` for
    at least ``min_consecutive_frames`` consecutive frames (after a small
    causal median smoothing pass). Output JSON includes the raw EAR series,
    a per-frame blink mask, and aggregate stats.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "threshold": ("FLOAT", {"default": 0.21, "min": 0.05,
                                         "max": 0.5, "step": 0.005}),
                "min_consecutive_frames": ("INT", {"default": 2, "min": 1,
                                                    "max": 30}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0,
                                   "step": 0.5}),
                "smooth_window": ("INT", {"default": 3, "min": 1, "max": 9}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("blink_report_json", "ear_series_json",
                    "blink_count", "blink_rate_hz")
    FUNCTION = "detect"
    CATEGORY = "WanAnimatePreprocess_V2/KANIBUS"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def detect(self, pose_data, threshold, min_consecutive_frames, fps,
               smooth_window):
        with torch.inference_mode():
            return self._detect_impl(
                pose_data, threshold, min_consecutive_frames, fps, smooth_window,
            )

    def _detect_impl(self, pose_data, threshold, min_consecutive_frames, fps,
               smooth_window):
        metas, _iris, (H, W) = _coerce_pose_data(pose_data)
        ear_r: List[float] = []
        ear_l: List[float] = []
        for m in metas:
            face_kps = m.get('keypoints_face')
            ear_r.append(_eye_aspect_ratio(face_kps, _RIGHT_EYE_IDX, W, H))
            ear_l.append(_eye_aspect_ratio(face_kps, _LEFT_EYE_IDX,  W, H))

        # Mean of both eyes per frame (NaN-aware)
        ear_mean: List[float] = []
        for r, l in zip(ear_r, ear_l):
            vals = [v for v in (r, l) if v == v]  # filter NaN
            ear_mean.append(float(np.mean(vals)) if vals else float('nan'))

        # Causal median smoothing (odd window only)
        win = max(1, int(smooth_window) | 1)
        ear_smooth = list(ear_mean)
        if win > 1:
            half = win // 2
            for i in range(len(ear_smooth)):
                lo = max(0, i - half)
                hi = i + 1  # causal
                chunk = [v for v in ear_mean[lo:hi] if v == v]
                if chunk:
                    ear_smooth[i] = float(np.median(chunk))

        # Threshold + run-length consolidation
        below = [(v == v) and (v < float(threshold)) for v in ear_smooth]
        blink_mask = [False] * len(below)
        run_start = None
        for i, b in enumerate(below + [False]):
            if b and run_start is None:
                run_start = i
            elif (not b) and run_start is not None:
                if (i - run_start) >= int(min_consecutive_frames):
                    for j in range(run_start, i):
                        blink_mask[j] = True
                run_start = None

        # Count blinks by counting rising edges in the mask
        blink_count = 0
        prev = False
        for b in blink_mask:
            if b and not prev:
                blink_count += 1
            prev = b
        duration_s = len(blink_mask) / max(float(fps), 1e-6)
        blink_rate_hz = (blink_count / duration_s) if duration_s > 0 else 0.0

        report = {
            "num_frames": len(blink_mask),
            "fps": float(fps),
            "threshold": float(threshold),
            "min_consecutive_frames": int(min_consecutive_frames),
            "blink_count": int(blink_count),
            "blink_rate_hz": round(blink_rate_hz, 4),
            "blink_mask": [bool(b) for b in blink_mask],
        }
        ear_series = {
            "right": [None if v != v else round(v, 5) for v in ear_r],
            "left":  [None if v != v else round(v, 5) for v in ear_l],
            "mean_smoothed": [None if v != v else round(v, 5) for v in ear_smooth],
        }
        return (
            json.dumps(report),
            json.dumps(ear_series),
            int(blink_count),
            float(blink_rate_hz),
        )


class SaccadeClassifierC2C:
    """Classify per-frame saccades from the gaze (yaw, pitch) time series.

    Computes angular velocity ω(t) = ||Δ(yaw,pitch)|| / Δt, smooths the
    yaw/pitch independently with a OneEuro low-pass, then thresholds at
    ``velocity_threshold_deg_s`` (300°/s by default — the classical
    physiological saccade boundary).

    Tracks the user's stricter intent: 'saccade' only when the velocity
    is sustained above threshold for at least ``min_consecutive_frames``
    frames (kills single-frame noise spikes).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0,
                                   "step": 0.5}),
                "velocity_threshold_deg_s": ("FLOAT", {"default": 300.0,
                                                        "min": 30.0,
                                                        "max": 1000.0,
                                                        "step": 5.0}),
                "min_consecutive_frames": ("INT", {"default": 1, "min": 1,
                                                    "max": 30}),
                "one_euro_min_cutoff": ("FLOAT", {"default": 1.0, "min": 0.1,
                                                   "max": 10.0, "step": 0.1}),
                "one_euro_beta": ("FLOAT", {"default": 0.05, "min": 0.0,
                                             "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("saccade_report_json", "velocity_series_json",
                    "saccade_count", "saccade_rate_hz")
    FUNCTION = "classify"
    CATEGORY = "WanAnimatePreprocess_V2/KANIBUS"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def classify(self, pose_data, fps, velocity_threshold_deg_s,
                 min_consecutive_frames, one_euro_min_cutoff, one_euro_beta):
        with torch.inference_mode():
            return self._classify_impl(
                pose_data, fps, velocity_threshold_deg_s,
                min_consecutive_frames, one_euro_min_cutoff, one_euro_beta,
            )

    def _classify_impl(self, pose_data, fps, velocity_threshold_deg_s,
                 min_consecutive_frames, one_euro_min_cutoff, one_euro_beta):
        _metas, iris_seq, _ = _coerce_pose_data(pose_data)
        if not iris_seq:
            empty = {"num_frames": 0, "saccade_count": 0,
                     "saccade_rate_hz": 0.0, "saccade_mask": []}
            return (json.dumps(empty), json.dumps({"velocity_deg_s": []}),
                    0, 0.0)

        # Mean of both eyes per frame (radians). Missing gaze → NaN.
        yaw_series: List[float] = []
        pitch_series: List[float] = []
        for entry in iris_seq:
            r = entry.get('right_gaze') or {}
            l = entry.get('left_gaze')  or {}
            ys = [v for v in (r.get('yaw_rad'), l.get('yaw_rad'))
                  if v is not None]
            ps = [v for v in (r.get('pitch_rad'), l.get('pitch_rad'))
                  if v is not None]
            yaw_series.append(float(np.mean(ys)) if ys else float('nan'))
            pitch_series.append(float(np.mean(ps)) if ps else float('nan'))

        yaw_f = _one_euro_filter(yaw_series,   freq_hz=float(fps),
                                  min_cutoff=float(one_euro_min_cutoff),
                                  beta=float(one_euro_beta))
        pitch_f = _one_euro_filter(pitch_series, freq_hz=float(fps),
                                    min_cutoff=float(one_euro_min_cutoff),
                                    beta=float(one_euro_beta))

        dt = 1.0 / max(float(fps), 1e-6)
        vel_deg_s: List[float] = [0.0]
        for i in range(1, len(yaw_f)):
            y0, y1 = yaw_f[i - 1], yaw_f[i]
            p0, p1 = pitch_f[i - 1], pitch_f[i]
            if any(v != v for v in (y0, y1, p0, p1)):
                vel_deg_s.append(float('nan'))
                continue
            dy = y1 - y0
            dp = p1 - p0
            omega_rad_s = math.hypot(dy, dp) / dt
            vel_deg_s.append(math.degrees(omega_rad_s))

        # Threshold + min-consecutive-frames run filter
        above = [(v == v) and (v >= float(velocity_threshold_deg_s))
                 for v in vel_deg_s]
        sacc_mask = [False] * len(above)
        run_start = None
        for i, b in enumerate(above + [False]):
            if b and run_start is None:
                run_start = i
            elif (not b) and run_start is not None:
                if (i - run_start) >= int(min_consecutive_frames):
                    for j in range(run_start, i):
                        sacc_mask[j] = True
                run_start = None

        saccade_count = 0
        prev = False
        for b in sacc_mask:
            if b and not prev:
                saccade_count += 1
            prev = b
        duration_s = len(sacc_mask) / max(float(fps), 1e-6)
        saccade_rate_hz = (saccade_count / duration_s) if duration_s > 0 else 0.0

        report = {
            "num_frames": len(sacc_mask),
            "fps": float(fps),
            "velocity_threshold_deg_s": float(velocity_threshold_deg_s),
            "min_consecutive_frames": int(min_consecutive_frames),
            "saccade_count": int(saccade_count),
            "saccade_rate_hz": round(saccade_rate_hz, 4),
            "saccade_mask": [bool(b) for b in sacc_mask],
        }
        vel_payload = {
            "velocity_deg_s": [None if v != v else round(v, 3) for v in vel_deg_s],
            "yaw_smoothed_rad":   [None if v != v else round(v, 5) for v in yaw_f],
            "pitch_smoothed_rad": [None if v != v else round(v, 5) for v in pitch_f],
        }
        return (
            json.dumps(report),
            json.dumps(vel_payload),
            int(saccade_count),
            float(saccade_rate_hz),
        )


class PupilDilationTrackerC2C:
    """Track per-frame pupil dilation, normalized against an invariant
    scale to be robust against the subject moving towards/away from the
    camera.

    Normaliser options:
      * 'eye_width' (default) — horizontal distance between outer/inner
        eye corners (landmark indices 37↔40 for the right eye, 43↔46 for
        the left). This is the canonical KANIBUS choice.
      * 'face_bbox_diag' — diagonal of the per-frame face bbox.
      * 'first_frame_radius' — pupil radius itself measured on the first
        non-NaN frame. Useful for relative ('% of baseline') reporting.

    A 'dilation event' is reported when the normalized radius rises above
    ``event_threshold`` (relative units) for at least ``min_consecutive_frames``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "normaliser": (["eye_width", "face_bbox_diag",
                                 "first_frame_radius"],
                                {"default": "eye_width"}),
                "event_threshold": ("FLOAT", {"default": 1.25, "min": 1.0,
                                               "max": 3.0, "step": 0.01}),
                "min_consecutive_frames": ("INT", {"default": 3, "min": 1,
                                                    "max": 60}),
                "smooth_window": ("INT", {"default": 5, "min": 1, "max": 31}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0,
                                   "step": 0.5}),
            },
            "optional": {
                "face_bboxes": ("BBOX",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("dilation_report_json", "radius_series_json",
                    "mean_normalized", "stddev_normalized")
    FUNCTION = "track"
    CATEGORY = "WanAnimatePreprocess_V2/KANIBUS"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def _scale_eye_width(self, face_kps, W, H) -> float:
        if face_kps is None or face_kps.ndim < 2:
            return float('nan')
        try:
            r_outer = face_kps[_RIGHT_EYE_IDX[0], :2]
            r_inner = face_kps[_RIGHT_EYE_IDX[3], :2]
            l_inner = face_kps[_LEFT_EYE_IDX[0],  :2]
            l_outer = face_kps[_LEFT_EYE_IDX[3],  :2]
        except Exception:
            return float('nan')
        widths = []
        for a, b in ((r_outer, r_inner), (l_inner, l_outer)):
            if not ((a == 0).all() or (b == 0).all()):
                widths.append(math.hypot((a[0] - b[0]) * W,
                                          (a[1] - b[1]) * H))
        if not widths:
            return float('nan')
        return float(np.mean(widths))

    def track(self, pose_data, normaliser, event_threshold,
              min_consecutive_frames, smooth_window, fps,
              face_bboxes=None):
        with torch.inference_mode():
            return self._track_impl(
                pose_data, normaliser, event_threshold,
                min_consecutive_frames, smooth_window, fps, face_bboxes,
            )

    def _track_impl(self, pose_data, normaliser, event_threshold,
              min_consecutive_frames, smooth_window, fps,
              face_bboxes=None):
        metas, iris_seq, (H, W) = _coerce_pose_data(pose_data)
        n = min(len(metas), len(iris_seq)) if iris_seq else len(metas)
        if n == 0:
            empty = {"num_frames": 0, "event_count": 0}
            return (json.dumps(empty), json.dumps({"r_px": [], "l_px": [],
                    "normaliser": []}), 0.0, 0.0)

        r_px: List[float] = []
        l_px: List[float] = []
        scale: List[float] = []
        for i in range(n):
            entry = iris_seq[i] if i < len(iris_seq) else {}
            ri = (entry.get('right_iris') or {}).get('radius')
            li = (entry.get('left_iris')  or {}).get('radius')
            r_px.append(float(ri) if ri is not None else float('nan'))
            l_px.append(float(li) if li is not None else float('nan'))
            if normaliser == "eye_width":
                scale.append(self._scale_eye_width(
                    metas[i].get('keypoints_face'), W, H))
            elif normaliser == "face_bbox_diag":
                if face_bboxes and i < len(face_bboxes) and face_bboxes[i] is not None:
                    try:
                        x1, x2, y1, y2 = face_bboxes[i]
                        scale.append(math.hypot(x2 - x1, y2 - y1))
                    except Exception:
                        scale.append(float('nan'))
                else:
                    scale.append(float('nan'))
            else:  # first_frame_radius — fill below
                scale.append(float('nan'))

        if normaliser == "first_frame_radius":
            # Find the first frame where at least one eye has a valid radius.
            baseline: Optional[float] = None
            for i in range(n):
                vals = [v for v in (r_px[i], l_px[i]) if v == v and v > 0]
                if vals:
                    baseline = float(np.mean(vals))
                    break
            scale = [baseline if (baseline is not None) else float('nan')
                     for _ in range(n)]

        # Per-frame normalized radius (mean of both eyes / scale)
        norm: List[float] = []
        for ri, li, s in zip(r_px, l_px, scale):
            if not (s == s) or s <= 1e-6:
                norm.append(float('nan'))
                continue
            vals = [v for v in (ri, li) if v == v and v > 0]
            if not vals:
                norm.append(float('nan'))
                continue
            norm.append(float(np.mean(vals)) / s)

        # Causal median smoothing
        win = max(1, int(smooth_window) | 1)
        smooth = list(norm)
        if win > 1:
            half = win // 2
            for i in range(n):
                lo = max(0, i - half)
                hi = i + 1
                chunk = [v for v in norm[lo:hi] if v == v]
                if chunk:
                    smooth[i] = float(np.median(chunk))

        valid = [v for v in smooth if v == v]
        mean_n = float(np.mean(valid)) if valid else 0.0
        std_n  = float(np.std(valid))  if valid else 0.0

        # Event detection (above mean_n * event_threshold for min_frames)
        ref = mean_n if mean_n > 0 else 1.0
        ev_thr = ref * float(event_threshold)
        above = [(v == v) and (v >= ev_thr) for v in smooth]
        event_count = 0
        run_start = None
        events: List[Tuple[int, int]] = []
        for i, b in enumerate(above + [False]):
            if b and run_start is None:
                run_start = i
            elif (not b) and run_start is not None:
                if (i - run_start) >= int(min_consecutive_frames):
                    events.append((int(run_start), int(i - 1)))
                    event_count += 1
                run_start = None

        report = {
            "num_frames": n,
            "fps": float(fps),
            "normaliser": normaliser,
            "event_threshold": float(event_threshold),
            "min_consecutive_frames": int(min_consecutive_frames),
            "mean_normalized": round(mean_n, 5),
            "stddev_normalized": round(std_n, 5),
            "event_count": int(event_count),
            "events": [{"start": s, "end": e} for s, e in events],
        }
        series = {
            "r_px":              [None if v != v else round(v, 3) for v in r_px],
            "l_px":              [None if v != v else round(v, 3) for v in l_px],
            "normaliser":        [None if v != v else round(v, 4) for v in scale],
            "normalized":        [None if v != v else round(v, 5) for v in norm],
            "normalized_smooth": [None if v != v else round(v, 5) for v in smooth],
        }
        return (
            json.dumps(report),
            json.dumps(series),
            float(mean_n),
            float(std_n),
        )


try:
    from .nodes_extras import (
        EXTRA_NODE_CLASS_MAPPINGS as _EXTRA_NODE_CLASS_MAPPINGS,
        EXTRA_NODE_DISPLAY_NAME_MAPPINGS as _EXTRA_NODE_DISPLAY_NAME_MAPPINGS,
    )
except Exception as _e:  # pragma: no cover
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "WanAnimatePreprocessV2: failed to import nodes_extras: %s", _e
    )
    _EXTRA_NODE_CLASS_MAPPINGS = {}
    _EXTRA_NODE_DISPLAY_NAME_MAPPINGS = {}


NODE_CLASS_MAPPINGS = {
    "OnnxDetectionModelLoaderV2": OnnxDetectionModelLoaderV2,
    "PoseAndFaceDetectionV2": PoseAndFaceDetectionV2,
    "DrawViTPoseV2": DrawViTPoseV2,
    "WanAnimateFaceQualityCheckV2": WanAnimateFaceQualityCheckV2,
    "DepthPoseCannyCombinedV2": DepthPoseCannyCombinedV2,
    # Self-contained alias (Task 2): same class, friendlier name highlighting bundled MiDaS + Normal + Blend modes
    # Removed 2026-05-18: SelfContainedControlNetPreprocessorV2 was just an alias of DepthPoseCannyCombinedV2.
    # KANIBUS gap nodes (May 2026): derived eye-feature series.
    "EARBlinkDetectorC2C": EARBlinkDetectorC2C,
    "SaccadeClassifierC2C": SaccadeClassifierC2C,
    "PupilDilationTrackerC2C": PupilDilationTrackerC2C,
    **_EXTRA_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnnxDetectionModelLoaderV2": "ONNX Detection Model Loader (V2)",
    "PoseAndFaceDetectionV2": "Pose and Face Detection (V2)",
    "DrawViTPoseV2": "Draw ViT Pose (V2)",
    "WanAnimateFaceQualityCheckV2": "Wan-Animate Face Quality Check (V2)",
    "DepthPoseCannyCombinedV2": "Depth + Pose + Canny Combined (V2)",
    # Display-name entry removed alongside class alias above.
    "EARBlinkDetectorC2C": "EAR Blink Detector",
    "SaccadeClassifierC2C": "Saccade Classifier (300\u00b0/s)",
    "PupilDilationTrackerC2C": "Pupil Dilation Tracker",
    **_EXTRA_NODE_DISPLAY_NAME_MAPPINGS,
}
