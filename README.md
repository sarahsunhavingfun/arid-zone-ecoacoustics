# arid-zone-ecoacoustics

Analysis and figure-generation scripts for BHC01, a regional bioacoustic classifier for
the Broken Hill Complex (arid NSW/SA), built via transfer learning on BirdNET V2.4 and
Perch V2.0 and benchmarked against foundation models on strongly-labelled test
soundscapes.

## Dependencies

This repo holds only the thesis-specific analysis scripts. The actual training,
inference, and soundscape-scoring pipelines live in three separate tools:

| Tool | Repo | Used for |
|---|---|---|
| BirdNET-Analyzer (fork) | [wcornwell/BirdNET-Analyzer](https://github.com/wcornwell/BirdNET-Analyzer) | training BHC01_BirdNET, embedding extraction, hyperparameter autotune |
| perch-head | [traitecoevo/perch-head](https://github.com/traitecoevo/perch-head) | training BHC01_Perch (Perch has no built-in training pipeline) |
| soundscape-eval | [traitecoevo/soundscape-eval](https://github.com/traitecoevo/soundscape-eval) | scoring every model against the labelled test soundscapes; produces the `per_species_metrics.csv` / `summary.csv` files most scripts here read |

A Python environment with `numpy`, `scipy`, and `matplotlib` is enough for every script
below except `species_list.qmd`, which needs R (`tidyverse`).

## Data

Scripts here read from a shared `final results/` data folder (recognisers, soundscape-eval
outputs, training libraries) that is not included in this repo — it's the output of the
tools above, run against the project's own audio data. Each script's own docstring/header
names the specific source files and folders it expects.

## Scripts

| Script | What it does |
|---|---|
| `soundscape_species_population.py` | Defines the canonical 52-species population used by every variation-analysis script below: species present in the test soundscapes that are also real trained BHC01 classes (excludes 2 ground-truth species — Mole Cricket, Plague Locust — that were never trained as a class). |
| `focal_vs_soundscape_auprc.py` | Correlates focal-recording (5-fold CV) AUPRC against soundscape AUPRC per species, for both backbones. |
| `library_size_correlation.py` | Correlates training library size (live clip counts, not a cached snapshot) against soundscape AUPRC. |
| `calling_activity_correlation.py` | Correlates calling activity (n_pos) against soundscape AUPRC and AUROC. |
| `freq_birds_only.py` | Correlates peak call frequency against soundscape AUPRC, restricted to the 43 bird species with a reliable frequency estimate. |
| `make_vocab_coverage.py` | Builds `bhc01_backbone_vocab_coverage.csv`: for each of BHC01's 203 classes, whether BirdNET 3.0 / Perch can represent it at all (vocabulary coverage, not accuracy), with verified name-synonym handling. |
| `add_vocab_coverage.py` | Same vocabulary-coverage check, applied to an external reference species list instead of BHC01's own label file. |
| `build_ensemble.py` | Builds the class-specific-replacement ensemble for each backbone: per class, uses the regional model's score if it robustly beats the foundation model (margin > the regional model's own repeat-to-repeat SD) or the foundation model can't represent that species at all; otherwise uses the foundation model's score. Produces 10 combined score caches per backbone (needs a `soundscape-eval` checkout and its score-cache dir). |
| `ensemble_eval.yaml` | `soundscape-eval` manifest for scoring the caches `build_ensemble.py` produces. |
| `win_loss_margin_test.py` | Classifies each shared species as a regional win/loss by the same margin-over-SD rule as `build_ensemble.py`, then tests whether win margins differ in size from loss margins (Mann-Whitney U). |
| `decompose_gain.py` | Decomposes each regional model's AUPRC gain over its foundation counterpart into two parts: coverage gain (species the foundation model can't represent at all) vs. improvement on shared classes. |
| `build_10avg_per_species.py` | Builds a combined per-species metrics table: the regional models' 10-repeat means alongside each foundation model's single run, as input to comparison figures. |
| `make_clip_performance_auprc_auroc.py` | Figure: regional-model clip-level (5-fold CV) AUPRC/AUROC distribution across all 203 classes. |
| `make_freq_clip_vs_soundscape.py` | Figure: peak call frequency vs. AUPRC, clips vs. soundscapes, same 43-bird population in both panels. |
| `all_model_compare_by_backbone.py` | Figure: all models' macro AUPRC/AUROC (foundation, regional, ensemble), split by backbone. |
| `optimise_hyperparameters.sh` | Reusable Bayesian hyperparameter search wrapper for either backbone (BirdNET's built-in `--autotune`, or `perch-head`'s `tune_head.py`). Not specific to this project — usable for any transfer-learning classifier head on either backbone. |
| `species_list.qmd` | R/Quarto notebook, BHC01's species list by taxonomic group. |
