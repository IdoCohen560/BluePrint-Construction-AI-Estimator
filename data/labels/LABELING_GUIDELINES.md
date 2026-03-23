# Labeling guidelines (match / no-match)

Use these rules to keep labels consistent across annotators.

1. **Match = 1** when the material row is an appropriate choice for the stated `material_type_req` and rough quantity context (`linear_ft_required`, `ceiling_height_ft`). It does not need to be the only valid SKU in the world—only a defensible pick from the fixed list.
2. **Match = 0** when the material is clearly the wrong category (e.g., concrete for a lumber requirement) or clearly inappropriate for the scale of work, given the spreadsheet fields you have.
3. If two materials could both be acceptable, prefer labeling the one closer to typical residential takeoff practice for the course scenario, and mark the other 0 unless both are genuinely equivalent—in that case pick one as 1 and one as 0 for the same `pair_id` to keep the row set deterministic.
4. **Do not** use price alone as the only signal; consider type and unit compatibility first.

Record disagreements in a note column if you add one later; the sample CSV omits it for simplicity.
