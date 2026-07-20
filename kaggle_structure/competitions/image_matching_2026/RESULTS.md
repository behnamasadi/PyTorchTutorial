# IMC 2026 — experiment log

Env: Kaggle GPU docker (`imc` container), RTX 3090, pycolmap 4.1.1 + kornia 0.8.3.

## Pipeline evolution

| Stage | Method | ETs (22) | stairs (51) | haiper train (54) |
|---|---|---|---|---|
| Plumbing test | SIFT → COLMAP | 86% reg | **0% reg** | — |
| **Tier 1 (current)** | DISK + LightGlue → pycolmap | **91% reg** | **29% reg** | 100% reg · rot_mAA 0.854 · trans_mAA 0.333 |

`stairs` (repetitive/low-texture) went 0%→29% just by swapping SIFT for deep matching — confirms the thesis. Still weak → needs dense matching.

## haiper mAA breakdown (local scorer, GT-aligned via Umeyama Sim3)

| Scene | reg | rot_mAA | trans_mAA | median_rot |
|---|---|---|---|---|
| bike | 15/15 | 1.000 | 0.478 | 0.4° |
| chairs | 16/16 | 0.562 | **0.021** | 2.6° |
| fountain | 23/23 | 1.000 | 0.500 | 0.15° |

**Read:** rotations are excellent (mAA 0.85); **position/translation is the bottleneck** (0.33), dragged down by `chairs` (symmetric/repetitive object). Rotation is basically solved; translation + hard scenes need work.

## Reused from magic-inspection-colmap
- `imc_metric.py` ← `holdout_localization.py` (rotation_error_deg, map_aa, umeyama_sim3) — the ETH3D/IMC mAA metric.
- Next: `mapanything_seed.py` pose/quaternion conversion for submission formatting; `config_matching.TwoViewGeometryParams` (min_inliers=12, max_error=4px, conf 0.999) for verification tuning; `run_pycolmap_ba` for BA.

## Experiment log (haiper train mAA / stairs reg)

| # | Change | haiper rot_mAA | haiper trans_mAA | stairs reg |
|---|---|---|---|---|
| 1 | DISK+LightGlue, 2048kp, 1024px | 0.854 | 0.333 | 29% |
| 2 | **+ 4096kp, 1280px** | **0.971** | **0.377** | 33% |
| 3 | + ALIKED ensemble (DISK+ALIKED) | 0.917 | 0.325 | **49%** |

**Findings:**
- More keypoints + resolution (exp 2) is the biggest single win — rotation near-solved (0.97).
- Ensemble (exp 3) **hurts on easy scenes** (haiper, already solved) but **big win on hard scenes** (stairs 33→49%). → route ensemble to hard scenes only, or accept the 2× cost.
- **Translation is still the bottleneck** (~0.33–0.38); tight thresholds (0.01–0.02) fail → needs bundle adjustment + denser/precise matches.
- `stairs` at 49% still low → needs **dense matching (LoFTR/RoMa)**.

## Sweep 2 — BA / principal-point / ensemble (haiper mAA, stairs reg)

| config | haiper rot | haiper trans | stairs reg |
|---|---|---|---|
| DISK (plain, 4k/1280) | **0.967** | 0.346 | 33% |
| DISK + extra BA | 0.954 | 0.342 | — |
| DISK + refine principal-point | 0.903 | 0.266 ⬇ | — |
| DISK + BA + PP | 0.958 | 0.355 | — |
| DISK+ALIKED ens | 0.967 | 0.342 | 35% |
| DISK+ALIKED ens + BA + PP | 0.954 | 0.318 | **6% ⬇⬇** |

**Negative results (measured, against prior hypotheses):**
- **Extra bundle adjustment → no gain** (incremental_mapping already BAs). Dropped.
- **Principal-point refinement → HARMFUL** — hurts easy, *destroys* hard scenes (unknown intrinsics + few views = degenerate). **Removed from pipeline.**
- **Run-to-run variance is high on hard scenes** (stairs 33–49% same config) → mapping instability, not a tuning target. Mitigate with denser matches / multi-seed mapping, and average multiple runs when comparing.
- Best stable config = **plain DISK 4096kp / 1280px**. Knob-tuning is a dead end; denser matching is the real lever.

## Sweep 3 — dense matchers (detector-free)

| matcher | stairs (hard) reg | haiper (easy) |
|---|---|---|
| DISK sparse | 33% | rot 0.967 / trans 0.346 ✅ |
| DISK+ALIKED ens | 35–49% | 0.967 / 0.342 |
| **LoFTR dense** | **55%** ✅ | **degenerate** (over-matches high-texture → fragmented, mAA≈0) |
| RoMa v2 dense (naive) | 4% ⬇, 733s | — |

RoMa naive = matches every pair densely (incl. non-overlapping) → false matches wreck mapping; pure-torch corr is very slow (no CUDA kernel). Needs retrieval-restricted pairs + CUDA build to be viable → **deprioritized**. **LoFTR is the practical dense winner.**

**Insight — per-scene routing confirmed:** dense matchers (LoFTR/RoMa) win on hard low-texture
scenes but *break* on easy high-texture multi-scene data; sparse (DISK/ALIKED) is the opposite.
Final pipeline must **route by scene difficulty**, not use one matcher everywhere.

## Full-train run (13 datasets, DISK + DINOv2 retrieval + pycolmap)

| Dataset | reg | rot_mAA | trans_mAA | note |
|---|---|---|---|---|
| pt_brandenburg | 100% | 0.985 | 0.442 | landmark ✅ |
| pt_piazzasanmarco | 99% | 0.979 | 0.409 | ✅ |
| pt_sacrecoeur_trevi_tajmahal | 100% | 0.942 | 0.566 | ✅ |
| pt_stpeters_stpauls | 100% | 0.968 | 0.459 | ✅ |
| imc2023_haiper | 100% | 0.967 | 0.339 | ✅ |
| imc2024_dioscuri_baalshamin | 85% | 0.465 | 0.205 | |
| imc2024_lizard_pond | 53% | 0.387 | 0.235 | |
| ETs | 86% | 0.500 | 0.093 | transparent, hard |
| imc2023_church | 100% | 0.188 | 0.000 | scorer frame-split? |
| imc2023_heritage | 46% | 0.067 | 0.162 | |
| amy_gardens | 55% | 0.000 | 0.000 | scorer artifact (single scene, fragmented models) |
| fbk_vineyard | 94% | 0.000 | 0.081 | scorer artifact (94% reg but rot=0) |
| stairs | 35% | 0.000 | 0.000 | hard → LoFTR gets 55% |

**Clean phototourism subset (5): rot_mAA ≈ 0.968, trans_mAA ≈ 0.443 — competitive-grade.**

### Robust-scorer correction (confirmed artifact)
Non-robust Umeyama scored several datasets at 0 because a GT scene's images split across
COLMAP models (mixed frames). `robust_umeyama` (RANSAC) fixes it — these were never broken:

| Dataset | rot_mAA before→after | trans_mAA before→after |
|---|---|---|
| fbk_vineyard | 0.000 → **0.255** | 0.081 → **0.289** |
| amy_gardens | 0.000 → **0.180** | 0.000 → **0.068** |
| imc2023_church | 0.188 → **0.595** | 0.000 → **0.299** |

Lesson: trust the metric before the model. Real weak spots that remain are the genuinely hard
scenes (stairs, ETs-transparent, heritage) → LoFTR routing + deeper retrieval.

## LoFTR vs RoMa — definitive head-to-head (train/stairs, mAA scored)

| Matcher | reg | rot_mAA | trans_mAA | time |
|---|---|---|---|---|
| **LoFTR** | **59%** | 0.000 | 0.057 | **149s** |
| RoMa (conf 0.7) | 4% | 0.000 | 0.000 | 735s |
| DISK (ref) | 37% | 0.000 | 0.047 | 37s |

**LoFTR wins decisively** (15× registration, 5× faster). RoMa (pure-torch, no CUDA kernel) not viable
here → dropped. Caveat: all matchers score rot_mAA≈0 on `stairs` — pathological repetitive scene
(COLMAP builds symmetry-flipped reconstructions regardless of matcher). Not a matcher problem;
needs rotation-prior/sequential handling. Phototourism datasets remain the prize (rot_mAA ~0.97).

**Final matcher policy:** DISK+LightGlue default (phototourism), LoFTR for low-texture/hard scenes,
RoMa dropped, ensemble only when a scene under-registers.

## Improvement attempts vs DISK baseline (all WASH — baseline stands)

Baseline = DISK+LightGlue, 4096kp/1280px, DINOv2 retrieval. Full-train: rot_mAA 0.617 / trans 0.333.

| Attempt | Result | Verdict |
|---|---|---|
| **DINOv3 retrieval** (vs v2) | weak datasets flat/slightly worse (lizard 0.48→0.39) | wash locally (helps official clustering metric only) |
| **DISK+ALIKED ensemble** | mixed: heritage 0.12→0.26 ✅, vineyard 0.20→0.06 ❌ | wash (mean flat ~0.32) |
| **Hi-res 1600px/8192kp** | trans flat/worse (haiper 0.356→0.296), 5× slower | wash |

**Conclusion: pipeline is at a plateau.** Matcher-level tweaks don't beat the DISK 4096/1280 baseline.
Remaining real gains require hard-scene-specific work (rotation priors for repetitive scenes,
MASt3R/VGGT fallback for low-registration scenes) — high effort, uncertain, on a no-medal comp.
Best config for submission = **DISK baseline**.

## Submission
Offline code-competition notebook (`submission/imc_infer.ipynb`) + deps dataset
(`imc-2026-offline-deps`: pycolmap wheel + DISK/LightGlue weights) pushed as kernel
`asadibehnam/imc-2026-baseline-disk-lightglue-colmap` — running on Kaggle to get a real LB number.

## Conclusion & the tuning we've been missing (2026-07-18)

**Meta-finding:** every *component swap* we tried washed out vs the DISK+LightGlue baseline —
DINOv2→DINOv3 (pairing), +ALIKED ensemble, +resolution. The base components are commodity;
the leaderboard spread (top ~60 vs many at 30–45, all using the same public models) is **system
tuning + per-scene engineering, not the matcher**. We're plateaued at rot_mAA 0.62 / trans 0.33
(≈ mid-pack) because we've only done the easy part.

**DINOv3 nuance:** it *is* measurably better at scene clustering (pt_brandenburg [75,75,75] vs v2's
merged [75,141,8,1]), but we're **not cashing it in** — the pipeline lets COLMAP split scenes and
never reconstructs per-cluster, so the v3 advantage is currently wasted. It also only helps the
*official* metric's scene-assignment component; our local per-GT-scene scorer is blind to it.

**Aspects/tuning we've IGNORED (real levers, ROI order):**
1. **Rotation-augmented matching** — many IMC images are rotated; DISK/LightGlue silently fail on
   large rotations. Top teams match each pair at 0/90/180/270°. We do none — likely our biggest miss.
2. **Two-view geometry / RANSAC tuning** — we used pycolmap *defaults*. The magic-inspection-colmap
   repo has tuned `TwoViewGeometryParams` (min_inliers=12, max_error=4px, conf=0.999) deciding which
   matches survive verification — untouched.
3. **COLMAP mapper tuning** — init-pair selection, min triangulation angle, multi-mapper runs +
   model merging. All defaults.
4. **Per-cluster reconstruction** — actually *use* DINOv3 clustering to reconstruct each scene in
   isolation → kills cross-scene contamination on the fragmented datasets (heritage/lizard/amy).
5. **Bespoke hard-scene handling** — ETs (transparent) + stairs (repetitive) tank the score; no
   special handling yet (rotation priors, sequential constraints).

**Next:** (1) finish the submission → get a real leaderboard number to measure against;
(2) attack **rotation-augmented matching + verification tuning** first (highest ROI).

### RANSAC / geometric-verification audit (2026-07-18)
- **We use pycolmap 4.1.1** (matches GitHub). Verification is `pycolmap.verify_matches(...)` with
  **default `TwoViewGeometryOptions`** → COLMAP's built-in RANSAC at **default settings; we tune nothing.**
- Relevant untuned knobs exposed by 4.1.1: `ransac.max_num_trials`, `confidence`, `min_num_trials`,
  `dyn_num_trials_multiplier`, `max_error`, `min_num_inliers`, `min_inlier_ratio`.
- **Paper arxiv 2503.07829 "Fixing the RANSAC Stopping Criterion"** (COLMAP group, Mar 2025): *not*
  a new algorithm — it corrects RANSAC's 1981 sampling-probability approximation that causes
  **undersampling → quits early, misses good models, worst in few-inlier/hard cases** (= our
  heritage/stairs/ETs failures). Whether the exact fix is merged into 4.1.1 is **unconfirmed**
  (needs COLMAP changelog check).
- **Actionable lever (untapped):** raise `max_num_trials` + `confidence` (and set repo's
  `max_error=4px`, `min_num_inliers=12`) in `verify_matches` → forces more sampling, approximating
  the paper's benefit even without the exact patch. Directly targets low-inlier registration failures.
  TODO: confirm changelog + benchmark tuned RANSAC on the weak datasets.

## Failure analysis (2026-07-18/19) — evidence-based diagnosis

Instrumented `analyze_failures.py`: match-count dist, RANSAC inlier ratios, graph connectivity,
registration correlation. **The three weak datasets fail for DIFFERENT reasons:**

| Dataset | matches/pair (median) | RANSAC inlier ratio | isolated | reg | **failure mode** |
|---|---|---|---|---|---|
| heritage (209) | 12 | median **0.00** / mean 0.30 | 0 | 46% | **repeated-structure false matches** — half the pairs have 0 geometric inliers (architecture repeats → LightGlue matches wrong copies) |
| stairs (51) | 9 | median 0.64 | 2 | 33% | **symmetry ambiguity** — per-pair geometry fine, global reconstruction wrong |
| amy_gardens (200) | 5 | median 0.81 | 7 | 67% | **connectivity** — registered imgs avg 12 neighbors vs 4 for unregistered |

**Key insight:** high match count ≠ usable. heritage looks well-connected (1847 "good" edges) but half
have inlier-ratio 0 → the *effective* graph is far sparser, and the mapper fragments (3 models).

**Targeted fixes (evidence-based, not blind swaps):**
- **amy_gardens (connectivity)** → increase retrieval depth (topk 30→60 / exhaustive) to link weak nodes.
- **heritage (false matches)** → stricter geometric verification: raise `min_num_inliers` / `min_inlier_ratio`
  so 0-inlier pairs never reach the mapper → cleaner graph, less fragmentation.
- **stairs (symmetry)** → hardest; needs rotation priors / sequential disambiguation.
- All benefit from the repo's tuned `TwoViewGeometryParams` (min_inliers=12, min_inlier_ratio=0.20, max_error=4px).

## ✅ FIRST WIN: tuned geometric verification (2026-07-19)

Evidence-based (from the failure analysis), unlike the earlier blind swaps. Tuned `verify_matches`:
`min_num_inliers=20`, `ransac.max_error=4px`, `confidence=0.99999`, `max_num_trials=100000`.

| Dataset | baseline rot/trans | **tuned verify** | note |
|---|---|---|---|
| heritage | 0.119 / 0.146 | **0.257 / 0.213** | rot 2.2× — dropped 0-inlier false-match pairs |
| church | 0.596 / 0.270 | 0.610 / 0.268 | slight gain |
| pt_brandenburg (control) | 0.985 / 0.490 | 0.986 / 0.491 | **no regression** ✅ |
| amy_gardens | 0.181 | 0.171 | flat |
| amy + topk60 | reg 62% | reg **74%** | deeper retrieval fixes connectivity |

**Adopt tuned verification by default.** First real improvement over the DISK baseline. Running full-13
to quantify overall mAA gain (baseline was 0.617/0.333).

## P100 GPU fix — SOLVED
torch **2.5.1+cu121** (prebuilt, no compile): arch list `[sm_50..sm_90]` incl sm_60(P100)+sm_86(3090),
works w/ kornia 0.8.3 + pycolmap 4.1.1. Bundled offline (2.7GB) → Kaggle dataset `imc-torch251-cu121`;
notebook installs it before importing torch. Documented in index.ipynb Section 2.

## Roadmap (priority order)
1. **Ensemble matchers** — add ALIKED+LightGlue and LoFTR (dense), concat matches before RANSAC → rescue stairs/chairs.
2. **Retrieval pairing (DINOv2)** — exhaustive O(n²) won't scale to 226-image datasets; retrieval top-k pairs.
3. **Bundle adjustment** — pycolmap BA refine after mapping (repo `run_pycolmap_ba`) → boost translation mAA.
4. **RoMa v2 / MASt3R** — dense/3D-foundation matchers for the hardest scenes (Tier 3).
5. **Rotation TTA** + per-scene-type routing.
6. Full-submission generation over all datasets + local mAA on the full train set.

## 🔑 THE RECKONING (2026-07-19) — first real LB score + why we're bottom-of-chart

### 1. We finally have a REAL leaderboard number: **17.96 → rank 21/24** (leader 59.5)
The two morning submissions eventually resolved: v9 (retrieval + tuned verify, GPU) scored **17.96**;
the CPU-fallback one stayed PENDING. So the "stuck" was our O(n²) notebook timing out, **never Kaggle**
(competitors scored the same day). Fixed via retrieval pairing + wall-clock budget in `imc_infer.py`.

### 2. The "mid-pack 0.62" was a MEASUREMENT ILLUSION (the big one)
We reported **rotation-only** mAA. Kaggle's metric counts a camera only if rotation **AND** translation are
BOTH within threshold. Recomputed our own cached train poses both ways:

| | rot-only (what we quoted) | trans-only | **COMBINED (official-style)** |
|---|---|---|---|
| our 3 test datasets, mean | 0.340 | 0.271 | **0.133** |
| ×100 scale | 34 | — | **13.3** ← matches Kaggle 17.96 |

→ **Rotation is ~solved; translation is THE bottleneck.** Added `combined_mAA` to `imc_metric.score_dataset`;
`run_full.py` now prints `COMBINED=` as the headline. **All prior "wash" verdicts were on rot-only and are
suspect** — retest promising ideas on `combined_mAA`.

### 3. The 27-point gap to the public baseline, explained
Top public notebook `syedmohdhuzaifa18/baseline-dinov2-aliked-lightglue` (16 votes, author = rank 7 @ **45.1**)
does 4 things we don't: (a) **rotation correction** — a `swsl_resnext50_32x4d` classifier detects each image's
0/90/180/270° rotation, `cv2.rotate`s it upright before matching (rotated imgs silently fail → our coverage
holes heritage 47% / vineyard 64%); (b) **RDD features @ 8192 kp at TWO resolutions (1024+1280)**; (c) **NetVLAD
retrieval**; (d) **matcher ensemble**. Pull with `kaggle kernels pull <slug>`.

### 4. ✅ Confirmed win — 8192 keypoints (combined metric)
| dataset | baseline (4096) COMBINED | **8192 kp** | +final_ba |
|---|---|---|---|
| imc2023_haiper (easy) | 0.291 | 0.312 | 0.315 (negligible) |
| imc2023_heritage (weak) | 0.046 | **0.149 (3.2×)** | — |

We were **feature-starved**. 8192 kp tripled the weak dataset, no regression on easy. **Baked into
`submission/imc_infer.py` (MAX_KP=8192).** `+final_ba` = negligible → skip (confirms earlier BA finding, now
on the combined metric). Knob-tuning ≈ +2-5; the real jump is rotation correction.

### 5. Per-cluster reconstruction — REFUTED (measured, not guessed)
Evidence-targeted at the fragmentation ✗ (`sfm_diagnostics.py`), but hurts: haiper 0.971→0.779 rot, heritage
0.257→0.147. Forcing DINOv3 clusters as reconstruction boundaries slices healthy scenes worse than the
fragmentation it prevents. Dropped.

### 6. Blocked: Kaggle API 403s (transient)
Resubmit of the 8192-kp notebook errored — API threw 403 on dataset endpoints, `kernels push` dropped the
deps datasets → kernel ran with no pycolmap → ERROR. Deps are fine (v9 used them). Hold until API recovers,
re-push WITH deps, submit.

### Next (ROI order, all measured on `combined_mAA`)
1. **Ship 8192-kp submission** once Kaggle API recovers → bank a real gain over 17.96.
2. **Rotation correction** — THE biggest lever. Needs an orientation classifier (like the public baseline's
   `swsl_resnext50_32x4d`) or a model-free scheme; mind pose handling (reconstruct on corrected images).
   Test locally on heritage/vineyard first.
3. Re-test ALIKED ensemble + deeper retrieval (`--topk`) on `combined_mAA` (old "wash" was rot-only).
