# Stage D group fixture checklist

`inputs/make_group_fixture.py` (fixed seed `20260722`) deterministically builds
`inputs/stage_d_group_target.png` exclusively from the already approved synthetic
portrait inputs. T1 and T2 are intentionally close enough for their paste ROIs to overlap.

This fixture checks source-to-target assignment, stable target ordering, unchanged targets,
and accumulated compositing. It does **not** evaluate visual blending quality; that remains
covered by the Stage C FaceFusion final-frame parity checklist.
