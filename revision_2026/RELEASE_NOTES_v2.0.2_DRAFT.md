# Proposed R2 evidence update — v2.0.2 (draft)

- Add 48 native E1 per-image attention-score NPZ files and 48 per-run JSON summaries as an immutable **addendum** to the prior v0.3.4 science-lock evidence.
- Publish a read-only rank-based score-to-AUROC verification script and digest manifest. Result: **576/576** layer-level E1 values match archived summaries; pooled L4 directional gap **+0.4142881933**.
- Preserve R2 score-level and bootstrap-resampling sensitivity checks in the same addendum.
- No retraining, no new checkpoint inference, no primary-result changes, and no reclassification of the **failed** original E6 random-PE replication gate.
- A future Zenodo version DOI will be entered only after the new version is actually published.

**This is a staging note, not an announcement of a published release.**

Archive: `PR_R2_E1_ATTENTION_SCORE_ADDENDUM_v1.0.0.zip`; SHA-256: `718c307e6e11113c6e0651b084d57673cbe55e85d1927f0fce788a00fcd20b8f`.
