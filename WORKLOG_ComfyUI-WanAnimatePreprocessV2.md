# WORKLOG — ComfyUI-WanAnimatePreprocessV2

**Stage 0 audit, 2026-08-29. Every number below was MEASURED this session, not inherited.**
Regenerate with the commands in the last section. Per R3 this file is the persisted source of
record; a claim that lives only in a chat transcript has now drifted five times.

> **Updated 2026-08-29 after the build passes.** The numbers above are re-measured, not the Stage-0 audit figures. A worklog that still reports its audit snapshot is the stale-record failure R3 exists to prevent.

## 1. Live inventory
| | |
|---|---|
| **Nodes registered (runtime)** | **16** |
| Registration style | V1 `NODE_CLASS_MAPPINGS` |
| `WEB_DIRECTORY` | `./js` |
| Test files | 3 |
| **Tests** | **32 passed** (was 4; action planner + keypoint contract added) |

## 2. Licence
**Apache-2.0.** MIT/Apache inbound only; GPL forbidden.

## 3. Registration smoke
PASS — 16 nodes. Two are conditional on optional weights (`WanGazeETHXGazeV2`,
`WanPose3DRefineNLFV2`).

## 4. Test status
4 tests, all passing — but that is **one file covering 16 nodes**. The gaze, expression and crop
math this pack exists for is essentially untested on CPU. Coverage is nominal, not real.

## 5. Invariant sweep
| Check | Result |
|---|---|
| `third_party` runtime imports | 0 |
| `IS_CHANGED` -> `float("nan")` | 0 in `IS_CHANGED`; `float('nan')` appears only in the helper `_scale_eye_width`, which is legitimate |
| Hardcoded `.cuda()` | 0 |
| Narrow `except ImportError` on a comfy import | **1** at `nodes.py:6160` — the comfy_kitchen skew does not reliably raise ImportError, so this guard can fail open |
| Frame loops without interrupt | **8** registered `nodes_extras/` modules |

## 6. Build queue (brief 2.2)
1. **Action/dance preset** — verified absent today. Motion-adaptive smoothing (heavy when still,
   light on snaps) plus a constant-size box, built on the existing `build_jitterless_boxes` + TV/L1.
   Build it; do not search for it again.
2. **Keypoint JSON export** — the data-format contract WanAnimal retarget consumes. Contract, not
   shared code.
3. FantasyPortrait 1024-dim expression drive — **licence unverified**; gate on that first (R6).

## 7. Blocked / decisions
FantasyPortrait licence must be checked before any port. The 8 interrupt-less extras modules should
be wired to the `_IC.track` helper already used in `nodes.py`.

## Regeneration commands

```
head -3 LICENSE

# registration smoke, the way ComfyUI loads (third_party/ComfyUI/nodes.py:2243-2263):
#   sys.modules[name] = mod   BEFORE   spec.loader.exec_module(mod)
# Anything less can report healthy for a pack that registers nothing.
python <scratch>/regsmoke.py ComfyUI-WanAnimatePreprocessV2

D:/PROJECT/ComfyUI_windows_portable/comfy_env/python.exe -m pytest tests/ -q
```

Shell python has no torch — always use the comfy_env interpreter.
