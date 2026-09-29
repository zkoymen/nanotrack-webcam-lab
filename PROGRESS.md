# Project Progress

Last updated: 2026-09-29

This file is a public, concise project status snapshot. It records verified changes separately from planned experiments. Detailed local notes, hardware inventory, private test footage, and screenshots stay under the Git-ignored `docs/` directory.

## Current status

**Prototype:** a live webcam app that lets the user select and follow one target with OpenCV TrackerNanoV2, with a local replay workflow for comparing trackers.

### Completed

- [x] Live camera capture, target selection, and single-object tracking.
- [x] Tight-ROI review before tracker initialization.
- [x] Optional luminance CLAHE preprocessing; disabled by default.
- [x] Configurable geometric area-drift safeguard that stops tracking and requests reselection.
- [x] Separate tracker update time and application FPS reporting.
- [x] Selectable NanoTrackV2, CSRT, and KCF backends share the same camera and ROI flow.
- [x] Add a private clip capture, sparse annotation, and same-video replay tool for repeatable NanoTrackV2 / CSRT / KCF comparisons.
- [x] Prefer Windows Media Foundation for webcam capture after measuring faster frame acquisition than DirectShow; keep DirectShow as fallback.
- [x] Review identity limits for visually identical objects and record the detector/tracker design trade-offs privately.
- [x] Record a private 1280×720 note sequence at 28.9 FPS, label 40 frames, and replay the same ROI through NanoTrackV2, CSRT, and KCF.
- [x] Public-source privacy exclusions for local environments, model weights, media, and `docs/`.
- [x] Initial technical review of small-target and rapid-motion failure modes.

### Measured baseline

The clip contains 358 frames; tracking was initialized at frame 153 and evaluated at 40 labeled frames through frame 357. This is one short sequence, not a general accuracy claim. Processing FPS is unpaced offline replay.

| Tracker | Mean IoU | IoU ≥ 0.5 | Mean update | Replay FPS |
|---|---:|---:|---:|---:|
| NanoTrackV2 | 0.6411 | 80% | 11.87 ms | 66.4 |
| CSRT | 0.6467 | 85% | 85.92 ms | 11.1 |
| KCF | 0.1617 | 20% | 28.98 ms | 77.0 |

NanoTrackV2 and CSRT first fell below IoU 0.5 at frame 193 and below 0.1 at frame 333. CSRT's small overlap gain came with much slower updates. KCF's API update failed at frame 192. Keep NanoTrackV2 as the live default pending more sequences; these results do not measure identity switches separately.

### In progress

- [ ] Run a controlled sequence with explicitly measured rapid reversals, scale change, and hand occlusion.
- [ ] Add identity-switch reporting for the two-instance overlap sequence.
- [ ] Select a target-capable detector/runtime for optional recovery before integrating it into the live loop.

### Next

1. Use the YOLO/detector findings in `docs/Tasks/IDENTITY_RESEARCH.md` to choose a model that can actually detect a selected note.
2. Keep the detector on a recovery path so it does not slow NanoTrackV2 on every frame.
3. Keep FEAR-XS deferred until an existing-environment-compatible inference route is available.
4. Tune area/motion guards only after more measurements.

Private video, annotation CSV, and per-frame result files remain in ignored `docs/`. No clip, image, model weight, or hardware identifier is part of the public result.

## How progress is recorded

- Keep this file to a short public status snapshot; update checkboxes only when there is evidence for the change.
- Keep private hardware details, webcam footage, annotated frames, and long-form investigation notes in ignored `docs/`.
- Make one focused Git commit per completed milestone, with a message describing the change.
- Track measurable work as GitHub Issues and link each issue from the related commits or pull requests. Current benchmark issue: [#1](https://github.com/zkoymen/nanotrack-webcam-lab/issues/1).
- Close an issue only after its acceptance criteria and results are recorded.
