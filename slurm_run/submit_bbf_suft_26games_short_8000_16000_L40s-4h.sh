#!/bin/bash
# Submit bbf_suft_26games_short_L40s-4h.sh on the L40s-4h partition as 2 separate job arrays, short term buffer
# sizes 8000 and 16000 (26 games x 4 runs = 104 tasks each, tf_lambda 0.1, epsilon_train 0.05, sampling_fraction 0.2).
# Each array has its own job name (bbf_suft_26g_s8000, bbf_suft_26g_s16000) and sends its own email when it ends.
# Finished runs are skipped by the array script, so nothing is overwritten
#
# Usage:
#   ./submit_bbf_suft_26games_short_8000_16000_L40s-4h.sh                  submit both arrays
#   MAX_PARALLEL=10 ./submit_bbf_suft_26games_short_8000_16000_L40s-4h.sh  run at most 10 tasks of each array at a time
#   DRY_RUN=1 ./submit_bbf_suft_26games_short_8000_16000_L40s-4h.sh        print the sbatch commands without submitting

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT=$SCRIPT_DIR/bbf_suft_26games_short_L40s-4h.sh

for SHORT_SIZE in 8000 16000; do
  ARRAY_ARG=()
  [ -n "$MAX_PARALLEL" ] && ARRAY_ARG=(--array="0-103%$MAX_PARALLEL")
  # Slurm writes the output files where sbatch is run, so run it from slurm_run/ like the other scripts
  CMD=(sbatch --chdir="$SCRIPT_DIR" --partition=L40s-4h --time=04:00:00 --job-name="bbf_suft_26g_s$SHORT_SIZE"
       "${ARRAY_ARG[@]}" "$SCRIPT" "$SHORT_SIZE")
  if [ -n "$DRY_RUN" ]; then
    echo "${CMD[@]}"
  else
    echo -n "short_term_buffer_size $SHORT_SIZE: "
    "${CMD[@]}" || exit 1
  fi
done
