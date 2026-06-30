#!/usr/bin/env bash
# Cluster runbook for Expert Pursuit — regenerate results on Orfeo / CINECA.
#
# Hand this file to an agent on the cluster. It drives the whole pipeline for one
# model at a time. Pick the model, pick a step, run. Artifacts land under
# $DATA_DIR/<model>/ and are exactly what the report and slides read.
#
#   Usage:   bash scripts/run_cluster.sh <model> <step>
#   Example: bash scripts/run_cluster.sh olmoe  report-repro     # reproduce the OLMoE report
#            bash scripts/run_cluster.sh gptoss circuit          # bigger gpt-oss intervention run
#            bash scripts/run_cluster.sh olmoe  rigor            # sufficiency / group / bootstrap
#
#   <model> = olmoe | gptoss
#   <step>  = extract | pursuit | concepts | analysis | circuit | rigor
#             | report-repro   (extract+pursuit+analysis for the report's headline dataset)
#             | all            (everything for this model, in order)
#
# What each step produces and which report element it backs:
#   extract   -> $DATA/<m>/extractions/<dataset>           (raw activations; prerequisite)
#   pursuit   -> pursuit/<dataset>/results.jsonl,evr_matrix  (tab:experts, median EVR, EVR range)
#   concepts  -> pursuit/<dataset>/{offensive,countries,numbers}/  (SOMP selector + tab:numbers)
#   analysis  -> analysis/<dataset>/logit_lens_comparison.json     (tab:lens: SOMP vs logit lens)
#   circuit    -> circuit/attribution/... + report.html    (gate-AtP localization + faithfulness)
#   downweight -> circuit/downweight/sweep_<concept>.json   (knockout/downweighting sweep + CIs)
#   rigor      -> circuit/rigor/...                         (sufficiency curve, group ablation, CIs)
#
# Datasets the report uses (per table):
#   tab:experts / median EVR  -> TriviaQA  (OLMoE headline; cleanest specialists)
#   tab:lens                  -> pile10k
#   circuit (tab:intervene…)  -> RealToxicityPrompts (rtp)
# Note: the gate-AtP-vs-patching faithfulness check (tab:faith) was a one-off; its grid is
# already cached in circuit/compare/faithfulness.json and is NOT part of this pipeline.

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment — adjust paths to the cluster, then leave the rest alone.
# ---------------------------------------------------------------------------
export DATA_DIR="${DATA_DIR:-./data}"            # where all artifacts are written/read
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"     # set 1 if weights/datasets are pre-staged
# export HF_HOME=/scratch/$USER/hf               # point HF cache at scratch on the cluster
PY="${PY:-.venv/bin/python}"                     # interpreter (a venv with requirements installed)

MODEL_KEY="${1:-}"
STEP="${2:-}"
[ -z "$MODEL_KEY" ] || [ -z "$STEP" ] && { sed -n '2,40p' "$0"; exit 1; }

# ---------------------------------------------------------------------------
# Per-model settings.
#   OLMoE fits one GPU  -> DEVICE_MAP=cuda (avoid 'auto' on a single GPU: it can offload to disk).
#   gpt-oss-20b needs ~2 GPUs -> DEVICE_MAP=auto for pipeline parallelism.
# ---------------------------------------------------------------------------
case "$MODEL_KEY" in
  olmoe)
    MODEL="allenai/OLMoE-1B-7B-0924-Instruct"
    export DEVICE_MAP="${DEVICE_MAP:-cuda}"
    HEADLINE_DATASET="triviaqa"   # tab:experts + median EVR
    CONCEPTS="offensive countries numbers"
    N_PROMPTS=100; N_TEST=64      # circuit train/test (OLMoE scale)
    ;;
  gptoss)
    MODEL="openai/gpt-oss-20b"
    export DEVICE_MAP="${DEVICE_MAP:-auto}"
    HEADLINE_DATASET="pile10k"    # gpt-oss descriptive run
    CONCEPTS="offensive"
    N_PROMPTS=100; N_TEST=64      # << bump from the old n_train=16 so the intervention is readable
    ;;
  *) echo "Unknown model '$MODEL_KEY' (use: olmoe | gptoss)"; exit 1 ;;
esac

export N_PROMPTS N_TEST
# Resolve the on-disk model dir exactly as the code does ('/' and '-' -> '_').
MODEL_DIR="$DATA_DIR/$($PY -c "from moe_interp.config import get_model_dir; print(get_model_dir('$MODEL').name)")"
echo "== model=$MODEL  device_map=$DEVICE_MAP  data_dir=$DATA_DIR  step=$STEP =="

# ---------------------------------------------------------------------------
# Steps.
# ---------------------------------------------------------------------------
do_extract() {   # $1 = dataset, $2 = n_docs (optional)
  local ds="$1" n="${2:-}"
  echo ">> extract $ds ${n:+(n_docs=$n)}"
  $PY main.py extract --model "$MODEL" --dataset "$ds" ${n:+--n_docs "$n"} --batch_size 8
}

do_pursuit() {   # full-vocab pursuit on $1 = dataset
  echo ">> pursuit (full) $1"
  $PY main.py pursuit --model "$MODEL" --dataset "$1" --k 50
}

do_concepts() {  # concept-restricted pursuit on rtp (the SOMP selector the circuit needs)
  for c in $CONCEPTS; do
    echo ">> pursuit --concept $c (rtp)"
    $PY main.py pursuit --model "$MODEL" --dataset rtp --concept "$c"
  done
}

do_analysis() {  # logit-lens vs SOMP on $1 = dataset (tab:lens)
  echo ">> analysis (logit lens vs SOMP) $1"
  $PY main.py analysis --model "$MODEL" --dataset "$1"
}

do_circuit() {   # gate-AtP localization + localization report (offensive)
  echo ">> circuit (offensive) n_prompts=$N_PROMPTS n_test=$N_TEST"
  $PY scripts/cineca/circuit_runner.py --model "$MODEL" \
      --n-prompts "$N_PROMPTS" --n-test "$N_TEST"
}

do_downweight() { # knockout/downweighting sweep, SOMP/AtP/random at 1% & 5% budgets (needs do_circuit grid)
  echo ">> downweight sweep (offensive) n_prompts=$N_PROMPTS n_test=$N_TEST"
  $PY scripts/cineca/downweight_runner.py --model "$MODEL" \
      --n-prompts "$N_PROMPTS" --n-test "$N_TEST"
}

do_rigor() {     # extra rigor analyses (need the offensive circuit grid from do_circuit first)
  echo ">> rigor: sufficiency curve / group ablation / bootstrap CIs"
  $PY scripts/rigor/sufficiency_curve.py --model "$MODEL" --n-prompts "$N_PROMPTS" --n-test "$N_TEST"
  $PY scripts/rigor/group_ablation.py    --model "$MODEL" --n-prompts "$N_PROMPTS" --n-test "$N_TEST"
  # bootstrap CIs run on the JSON the steps above wrote (no model needed):
  $PY scripts/rigor/bootstrap.py \
      "$MODEL_DIR/circuit/rigor/sufficiency_offensive.json" \
      "$MODEL_DIR/circuit/rigor/group_ablation_offensive.json" || true
}

case "$STEP" in
  extract)      do_extract "$HEADLINE_DATASET" $([ "$MODEL_KEY" = olmoe ] && echo 50000); do_extract rtp; do_extract pile10k ;;
  pursuit)      do_pursuit "$HEADLINE_DATASET"; do_pursuit pile10k ;;
  concepts)     do_concepts ;;
  analysis)     do_analysis pile10k ;;
  circuit)      do_circuit ;;
  downweight)   do_downweight ;;
  rigor)        do_rigor ;;
  report-repro)
    # The minimal set to reproduce this model's report numbers from scratch.
    if [ "$MODEL_KEY" = olmoe ]; then do_extract triviaqa 50000; do_pursuit triviaqa; fi
    do_extract pile10k; do_pursuit pile10k; do_analysis pile10k
    do_extract rtp; do_concepts; do_circuit; do_downweight
    ;;
  all)
    do_extract "$HEADLINE_DATASET" $([ "$MODEL_KEY" = olmoe ] && echo 50000)
    do_extract pile10k; do_extract rtp
    do_pursuit "$HEADLINE_DATASET"; do_pursuit pile10k
    do_concepts; do_analysis pile10k; do_circuit; do_downweight; do_rigor
    ;;
  *) echo "Unknown step '$STEP'"; sed -n '2,40p' "$0"; exit 1 ;;
esac

echo "== done: $MODEL_KEY / $STEP =="
