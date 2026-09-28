# Project Progress

Last updated: 2026-09-29

This file is a public, concise project status snapshot. It records verified changes separately from planned experiments. Detailed local notes, hardware inventory, private test footage, and screenshots stay under the Git-ignored `docs/` directory.

## Current status

**Prototype:** a live webcam app that lets the user select and follow one target with OpenCV TrackerNanoV2.

### Completed

- [x] Live camera capture, target selection, and single-object tracking.
- [x] Tight-ROI review before tracker initialization.
- [x] Optional luminance CLAHE preprocessing; disabled by default.
- [x] Configurable geometric area-drift safeguard that stops tracking and requests reselection.
- [x] Separate tracker update time and application FPS reporting.
- [x] Public-source privacy exclusions for local environments, model weights, media, and `docs/`.
- [x] Initial technical review of small-target and rapid-motion failure modes.

### In progress

- [ ] Build a repeatable private test set for small targets, rapid reversals, scale changes, and brief occlusion.
- [ ] Record baseline tracking accuracy, loss/recovery behavior, frame cadence, and latency for NanoTrackV2.

### Next

1. Finish the private baseline sequences and annotate target boxes.
2. Compare NanoTrackV2 with FEAR-XS and OpenCV CSRT on the same clips and initial boxes.
3. Include KCF or MOSSE as speed-oriented reference trackers.
4. Improve the lost-state diagnostics and tune area/motion guards from measured results.
5. Revisit capture modes and image quality only after checking the private hardware notes.

No alternate tracker benchmark has been completed yet. Candidate names above are planned experiments, not performance claims.

## How progress is recorded

- Keep this file to a short public status snapshot; update checkboxes only when there is evidence for the change.
- Keep private hardware details, webcam footage, annotated frames, and long-form investigation notes in ignored `docs/`.
- Make one focused Git commit per completed milestone, with a message describing the change.
- Once the GitHub repository is available, create one GitHub Issue per measurable work item and link its issue number from the relevant commit or pull request.
- Close an issue only after its acceptance criteria and results are recorded.
