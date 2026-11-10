#!/bin/bash
# Bayesian hyperparameter search for a custom classifier head, on either backbone.
#
# Both backbones search the SAME way -- Optuna's TPESampler (a Bayesian optimiser),
# seed=42, with a MedianPruner to cut weak trials early -- so results are comparable
# across backbones. They run through two separate codebases, though, because BirdNET
# and Perch classifiers are trained by two separate tools:
#   - BirdNET: this repo's own `birdnet_analyzer.train --autotune`
#   - Perch:   a companion repo's `scripts/tune_head.py` (traitecoevo/perch-head)
# This script is a thin wrapper over both, so starting a search is one command instead
# of having to know both tools' flags.
#
# Usage:
#   ./optimise_hyperparameters.sh birdnet <training_library_dir> <output_name> [trials]
#   ./optimise_hyperparameters.sh perch   <embeddings_cache.npz> <output_dir>  [trials]
#
# Examples:
#   ./optimise_hyperparameters.sh birdnet /path/to/my_library my_model_v1 20
#   ./optimise_hyperparameters.sh perch   train_caches/my_model.npz tune_runs/my_model_bayes 20
#
# Prerequisites:
#   - BirdNET: just a training library folder (one subfolder of audio clips per class).
#   - Perch:   an embeddings cache first -- run perch-head's extract_embeddings.py
#              (or extract_library_embeddings.py) over your training library to produce
#              the .npz this script's --npz expects. tune_head.py itself must run under
#              THIS repo's .venv (needs optuna; perch-head's own venv doesn't have it).
#
# Where results land:
#   - BirdNET: <output_name>_Params.csv (the winning hyperparameters) alongside the
#              trained model and its validation metrics. No trial-by-trial log is kept --
#              only the winner. If you want the search history, redirect this command's
#              stdout to a file yourself.
#   - Perch:   <output_dir>/best_params.json (winning config + which trial, train_head.py
#              command ready to copy) and <output_dir>/trials.csv (every trial, so the
#              full search history survives this time). Resumable: killing and rerunning
#              the same command continues the same study instead of starting over.
set -euo pipefail

BACKBONE="${1:?usage: $0 <birdnet|perch> <input> <output> [trials]}"
INPUT="${2:?missing input: training library dir (birdnet) or embeddings .npz (perch)}"
OUTPUT="${3:?missing output: output name (birdnet) or output dir (perch)}"
TRIALS="${4:-20}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # adjust if this script moves

case "$BACKBONE" in
  birdnet)
    echo "BirdNET: Bayesian search, $TRIALS trials, optimising focal-recording AUPRC"
    python "$REPO_ROOT/train_tf_first.py" -m birdnet_analyzer.train \
        "$INPUT" \
        --output "$OUTPUT" \
        --save_cache_to "${OUTPUT}_train_cache.npz" \
        --autotune \
        --autotune_trials "$TRIALS" \
        --autotune_metric val_AUPRC
    echo "Winning hyperparameters: ${OUTPUT}_Params.csv"
    ;;
  perch)
    PERCH_HEAD_REPO="$(dirname "$REPO_ROOT")/perch-head"   # adjust to your checkout layout
    echo "Perch: Bayesian search, $TRIALS trials, optimising macro AUPRC"
    (cd "$PERCH_HEAD_REPO" && \
     "$REPO_ROOT/.venv/bin/python" scripts/tune_head.py \
        --npz "$INPUT" \
        --out-dir "$OUTPUT" \
        --trials "$TRIALS" \
        --objective macro_auprc)
    echo "Winning hyperparameters: $PERCH_HEAD_REPO/$OUTPUT/best_params.json"
    echo "Full trial history:      $PERCH_HEAD_REPO/$OUTPUT/trials.csv"
    ;;
  *)
    echo "backbone must be 'birdnet' or 'perch', got: $BACKBONE" >&2
    exit 1
    ;;
esac
