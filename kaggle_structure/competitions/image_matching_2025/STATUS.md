# Image Matching Challenge — STATUS & HANDOFF

Read this first. Full experiment log is in `RESULTS.md`; this is the "what do I do now" summary.
Last updated 2026-07-19.

## TL;DR
- **Competition:** `image-matching-challenge-2025-ongoing` (Kaggle). ⚠️ **Community tier → NO medals**
  (it's practice / leaderboard / writeup, not a medal path). The medal IMC editions (2023/24/25 Research) are closed.
- **Task:** given unordered images, cluster them into scenes AND estimate each camera's 6-DoF pose. Metric = mAA.
- **State (2026-07-19):** we have a **REAL leaderboard score: 17.96 → rank 21/24** (leader 59.5). Not stuck
  anymore. We are near the bottom, and we now understand exactly why (below).

## 🔑 KEY FINDINGS (2026-07-19) — read this
1. **The "stuck PENDING" was OUR notebook timing out, not Kaggle.** Fixed (retrieval pairing + wall-clock
   budget). v9 finally scored **17.96**.
2. **"Mid-pack ~0.62" was a MEASUREMENT ILLUSION.** We reported **rotation-only** mAA. Kaggle's metric needs
   rotation **AND** translation *both* within threshold. Recomputing our own train poses that way gives
   **~0.13 combined**, matching the 17.96. → **Rotation is ~solved; translation is THE bottleneck.**
   The scorer now reports `combined_mAA` as the headline (`imc_metric.py`) — **stop trusting rot-only.**
3. **Why we're bad, ranked:** (a) we submitted a *crippled* baseline (4096 kp, exhaustive, default verify) —
   not our tuned pipeline; (b) the rot-only illusion; (c) **we lack rotation correction**, the one technique
   the rank-7 public notebook (`syedmohdhuzaifa18/baseline-dinov2-aliked-lightglue`, score 45.1) has that we
   don't — rotated 90/180/270° images silently fail to match → our coverage holes (heritage 47%, vineyard 64%).
4. **Confirmed win — 8192 keypoints (up from 4096):** measured **3.2× on heritage** combined (0.046→0.149),
   no regression on easy scenes. We were feature-starved. **Baked into `submission/imc_infer.py`, ready to ship.**
5. **Biggest remaining lever = rotation correction** (public baseline does it with a classifier). Next build.
6. `+final_ba` = negligible (haiper 0.312→0.315). Skip. Knob-tuning gives +2-5; rotation correction is the jump.

## ⏳ Blocked right now: Kaggle API 403s
Resubmit of the 8192-kp notebook FAILED — Kaggle's API is throwing **403 Forbidden on dataset endpoints**
(transient), so `kernels push` dropped our offline-deps datasets → the kernel ran with no pycolmap/torch →
**ERROR**. The deps datasets are fine (v9 used them hours ago). **Do NOT submit until the API recovers**; then
re-push WITH deps attached and submit. Retry: `kaggle datasets files asadibehnam/imc-2026-deps-min` should
stop 403-ing when it's back.

## Current best pipeline (what to run)
`DISK + LightGlue → DINOv3 retrieval pairing → pycolmap incremental mapping`, **tuned verification**,
**8192 keypoints**, 1280px. (Older notes said 4096/default-verify was best — that was judged on rotation-only;
on the combined metric, 8192 kp + tuned verify win.)

**Run it (local, RTX 3090, inside the `imc` docker container):**
```bash
docker start imc   # container has pycolmap 4.1.1 + kornia + torch, GPU
docker exec imc python /work/run_full.py --split train --score            # full-13 local mAA
docker exec imc python /work/run_full.py --split train --score --only imc2023_haiper   # one dataset
# flags: --matchers disk,aliked | --tuned_verify | --per_cluster | --topk N
```
Data lives at `data/extracted/{train,test}/<dataset>/`. Metric code: `imc_metric.py` (from magic-inspection-colmap).

## Key files
| File | What it is |
|---|---|
| `run_full.py` | main pipeline: extract→retrieval→match→pycolmap, per-dataset, writes mAA + poses cache |
| `experiment.py` | matcher extract/match/DB helpers (DISK/ALIKED + LightGlue) |
| `retrieval.py` | DINOv3 (`imc_deps/dinov3_vitb16`) + DINOv2 descriptors, clustering, top-k pairing |
| `run_loftr.py`, `run_roma.py` | dense matchers (detector-free) for hard scenes |
| `imc_metric.py` | ETH3D/IMC mAA scorer (robust Umeyama) |
| `analyze_failures.py` | per-dataset failure diagnostics (match counts, RANSAC inliers, connectivity) |
| `submission/imc_infer.py` | the offline **code-competition notebook** (DISK+LightGlue→pycolmap) |

## Results (local mAA, full-13 train, robust scorer)
**rot_mAA 0.617 / trans_mAA 0.333** → ≈ **mid-pack** (leaderboard leader ~59.5 on 0–100 scale).
Strong on phototourism (pt_* datasets rot ~0.97), weak on hard scenes (stairs/ETs ~0, heritage/lizard low).

## Submission status ⚠️ — CAUSE FOUND (2026-07-19), NOT Kaggle
- **The stuck PENDING was OURS, not Kaggle.** Competitors got scored the same day
  (stpete_ishii 56.5 @ 07:50); our submissions (10:23/10:36) never returned. The kernel
  "COMPLETE" is the interactive save-run on the tiny public sample (7 min); a code-comp
  *submission* re-executes on the full hidden test set — that rerun is what hung.
- **Root cause:** `submission/imc_infer.py` used **exhaustive O(n²) pairing**, no wall-clock
  budget, and a whole-dataset CPU fallback → the hidden-test rerun ran past the time limit
  and never returned a score. The tuned `run_full.py` retrieval pipeline was never ported in.
- **P0 FIX DONE** (`submission/imc_infer.py` + regenerated `.ipynb`): retrieval top-k pairing
  (global descriptors pooled from DISK feats — no extra weights/internet), wall-clock budget
  that flushes `submission.csv` after every dataset (small-first), GPU validated once up front
  (no per-scene CPU reruns), tuned verification. **Next:** repush kernel + resubmit → real LB #.
- **To submit** (code competition, needs `-k` + version): after the kernel run completes,
  `kaggle competitions submit -c image-matching-challenge-2025-ongoing -k asadibehnam/imc-2026-baseline-disk-lightglue-colmap -f submission.csv -v <N> -m "msg"`

## Failure diagnostics — evidence layer (`sfm_diagnostics.py`, 2026-07-19)
Mirrors magic-inspection's "Manual analysis — the evidence": each dataset run prints a
✓/⚠/✗ evidence table (thresholds, data-flow order — **read the first ✗ = root cause**) and
caches `work/diag_train.json`. IMC calibration: >1 COLMAP model per dataset is EXPECTED
(multi-scene), so the real fragmentation signal is **per-GT-scene coherence** (from labels).
Findings (tuned_verify): **haiper** healthy (0.97); **fbk_vineyard** geometry ✓ (reproj 0.79px)
but scores ~0 from scene fragmentation — NOT a matcher problem; **heritage** first ✗ = 47%
coverage. Different datasets fail for different reasons → stop blind matcher swaps.

## What's been TESTED and DOESN'T help (don't repeat — plateau confirmed)
| Attempt | Verdict |
|---|---|
| DINOv3 retrieval (vs v2) | wash (better clustering, but end-to-end flat) |
| DINOv3 per-cluster reconstruction | wash (heritage slight up, church down) |
| DISK+ALIKED ensemble | wash / hurts (helps stairs reg, hurts vineyard) |
| Higher resolution (1600/8192kp) | wash, 5× slower |
| Extra bundle adjustment | no gain |
| Principal-point refinement | HARMFUL (destroys hard scenes) |
| Tuned RANSAC verification | marginal (+heritage, ~neutral overall) |

## The P100 GPU fix (reusable)
Kaggle's default torch dropped Pascal (P100) → install `torch==2.5.1+cu121` (prebuilt, supports P100+3090).
Offline: bundled as Kaggle dataset `asadibehnam/imc-torch251-cu121`. Online (internet): just pip install it.

## Real remaining levers (untested / high-effort — the only path past the plateau)
1. **RoMa v2 / MASt3R with the CUDA kernel + retrieval-restricted pairs** — for the hard scenes (stairs/ETs)
   where everyone (including us) scores ~0. This is where leaderboard points are actually won.
2. **Per-scene-type routing** (transparent ETs / repetitive stairs get bespoke handling + rotation priors).
3. Multi-seed mapping to fight the high run-to-run variance on hard scenes.

## Honest recommendation
IMC-ongoing gives **no medal**. It's a strong portfolio/writeup piece and plays to the COLMAP/SLAM edge,
but for *medals* the path is the Biohub competition (needs joining) or notebook upvotes. Treat IMC as
practice — the pipeline is done; only pursue the RoMa/MASt3R hard-scene work if we want a real leaderboard climb.
