# LLM Council — 2026-05-04
## Question
How do we reliably segment "the building plan region" from everything else
on a blueprint sheet (legends, schedules, parking-detail tiles, RCPs,
finish-plan enlargement boxes), running on free CPU Streamlit (≤500 MB
RAM), without big model weights?

## Convergence — 4 of 5 advisors say the same thing
**Stop trying to auto-detect. Let the user click-and-drag a box around
the building.** It's a 2-second gesture for a contractor. The auto-
classifier will keep failing on new sheet templates forever because
"floor plan + legend on same page" is not a robust signal at 100 DPI
without a real model. Demote rules from gate to hint.

## The Contrarian (unique angle)
Pipeline is wrong: PDFs from Revit/AutoCAD/ArchiCAD ship with vector
strokes, text objects, and viewport clip rectangles. Rasterizing first
throws away structure we already have. Use `pymupdf` to extract
viewports and stroke geometry directly. The largest non-titleblock
viewport with low text-density / high stroke-length IS the floor plan.
Real walls are double-line stroke PAIRS at typical wall thickness;
legends/RCPs/details are not. Stroke-pair distance histogram is the
discriminator, not "spatial entropy of ink." Forget DocLayout/
LayoutParser; they'll burn the RAM budget for worse results.

## The First Principles Thinker
The actual job is "produce an accurate stucco bid," not "classify
pages." Right now we're spending engineering on a CPU-bound classifier
to avoid asking the human a 3-second question. That's the wrong
tradeoff. Reframe: thumbnail grid → user picks plan pages → user crops
each → wall detection runs only on the crop. Two clicks, zero false
negatives. If we MUST auto-detect, use sliding-window tile scoring
(parallel-line density + enclosed rooms + text density) to grow a
contiguous region — that fixes the "one big component swallows the
sheet" failure.

## The Expansionist
The framing is too small. Stucco contractors lose deals on pricing
speed and lose margin on missed scope (parapets, soffits, foam trim,
control joints, scaffolding by height). The real product is a
defensible itemized proposal in 10 minutes — that's $300-500/mo tooling,
not $99. Every uploaded bid is a labeled pair (architectural sheet,
human wall polygons, $/sqft, ZIP) — after 5,000 bids you own the only
regional stucco pricing dataset in existence. Worth more than the app
to insurers, GCs, Procore. Same pipeline works for drywall, framing,
painting, EIFS — stucco is just the beachhead.

## The Outsider (caught the biggest blind spot)
You have humans who already know which box is the building, and an AI
that keeps guessing wrong. Why is the AI guessing?

Two killer points:
1. **Stop segmenting. Ask.** Drag-rectangle UI takes a weekend, is 100%
   accurate forever. The user can solve in 2 seconds what we're
   spending engineering days on.
2. **The completed exhibits ARE labeled training data and we're
   ignoring it.** Humans highlighted the building areas in red — that's
   literally a segmentation dataset on disk. Extract those highlights
   as bounding boxes → free supervised labels.
3. We're conflating two problems: "which region is a floor plan" (cheap,
   solvable by a click) and "where are walls inside that region" (the
   real CV problem). Fighting both with one rule set is why rules keep
   breaking.

## The Executor (concrete plan)
- **Mon (2 hr)**: add `streamlit-cropper` to the upload flow. User
  drags a rectangle around the building → ROI passed to existing
  detector.
- **Tue**: add "run on all pages → user toggles which to include"
  mode. Two `st.checkbox` calls in a loop. False rejects disappear
  because nothing is auto-rejected — only ranked.
- **Wed**: demote the rule-based classifier to a HINT. Show "suggested:
  page 3 (floor plan)" with a confidence number. Never auto-exclude.
  The Saerom failure mode evaporates because nothing is being rejected.
- **Thu**: cache cropped ROI per (file_hash, page) in
  `st.session_state` so re-runs are instant.
- **Fri**: ship.

Skip room-variance heuristics (will fail on the next weird sheet) and
SAM-tiny (will OOM on Streamlit Cloud free tier or take 30s/page).

## Verdict
Ship the click-to-crop now. Demote the rule-based classifier to a hint
that pre-fills the crop instead of gating it. Cropper-driven ROI
guarantees the wall detector only ever sees clean floor-plan regions.

Two follow-ups (not blocking):
1. Extract building-area bboxes from the existing completed-exhibit red
   highlights → free supervised labels for a real classifier later.
2. Investigate vector-extraction path (PyMuPDF viewports, stroke pairs)
   as the long-term replacement for Hough+rules.
