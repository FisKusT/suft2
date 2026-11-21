#!/bin/bash
# Submit bbf_suft_26games_short_L40s-4h.sh as 4 separate job arrays, one per short term buffer size
# (26 games x 4 runs = 104 tasks each). Each array has its own job name (bbf_suft_26g_s<size>) and sends its own
# email when it ends (--mail-type=END in the array script)
#
# Usage:
#   ./submit_bbf_suft_26games_short.sh                  short term buffer sizes 2000 4000 8000 16000
#   ./submit_bbf_suft_26games_short.sh 8000 16000       only these sizes
#   MAX_PARALLEL=10 ./submit_bbf_suft_26games_short.sh  run at most 10 tasks of each array at a time
#   DRY_RUN=1 ./submit_bbf_suft_26games_short.sh        print the sbatch commands without submitting

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT=$SCRIPT_DIR/bbf_suft_26games_short_L40s-4h.sh

if [ $# -gt 0 ]; then
  SHORT_SIZES=("$@")
else
  SHORT_SIZES=(2000 4000 8000 16000)
fi

for SHORT_SIZE in "${SHORT_SIZES[@]}"; do
  if ! [[ "$SHORT_SIZE" =~ ^[0-9]+$ ]]; then
    echo "Short term buffer size must be a positive integer, got '$SHORT_SIZE'" >&2
    exit 1
  fi
done

for SHORT_SIZE in "${SHORT_SIZES[@]}"; do
  ARRAY_ARG=()
  [ -n "$MAX_PARALLEL" ] && ARRAY_ARG=(--array="0-103%$MAX_PARALLEL")
  # Slurm writes the output files where sbatch is run, so run it from slurm_run/ like the other scripts
  CMD=(sbatch --chdir="$SCRIPT_DIR" --job-name="bbf_suft_26g_s$SHORT_SIZE" "${ARRAY_ARG[@]}" "$SCRIPT" "$SHORT_SIZE")
  if [ -n "$DRY_RUN" ]; then
    echo "${CMD[@]}"
  else
    echo -n "short_term_buffer_size $SHORT_SIZE: "
    "${CMD[@]}" || exit 1
  fi
done
