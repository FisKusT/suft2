#!/bin/bash
# Submit bbf_suft_26games_short_shaham-L40-fast.sh on the shaham-L40-fast partition as 2 separate job arrays, short
# term buffer sizes 500 and 250 (26 games x 4 runs = 104 tasks each, tf_lambda 0.1, epsilon_train 0.05,
# sampling_fraction 0.2). Same configuration as the 8000 / 16000 runs, only the short term buffer size differs.
# Each array has its own job name (bbf_suft_26g_s500, bbf_suft_26g_s250) and sends its own email when it ends.
#
# Task ranges (the run number varies fastest, see 'bash bbf_suft_26games_short_shaham-L40-fast.sh --list 500'):
#   short 500: tasks 12-103. Tasks 0-11 (Alien, Amidar, Assault, runs 0-3) already finished, so they are not submitted
#   short 250: tasks 0-103, all games
# Override a range with ARRAY_500 / ARRAY_250, e.g. ARRAY_500=12-103 ARRAY_250=0-103
# Finished runs are also skipped by the array script itself, so nothing is overwritten
#
# Usage:
#   ./submit_bbf_suft_26games_short_500_250_shaham-L40-fast.sh                  submit both arrays
#   MAX_PARALLEL=10 ./submit_bbf_suft_26games_short_500_250_shaham-L40-fast.sh  run at most 10 tasks of each array at a time
#   DRY_RUN=1 ./submit_bbf_suft_26games_short_500_250_shaham-L40-fast.sh        print the sbatch commands without submitting

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT=$SCRIPT_DIR/bbf_suft_26games_short_shaham-L40-fast.sh

declare -A ARRAY_RANGE=(
  [500]=${ARRAY_500:-12-103}
  [250]=${ARRAY_250:-0-103}
)

for SHORT_SIZE in 500 250; do
  RANGE=${ARRAY_RANGE[$SHORT_SIZE]}
  [ -n "$MAX_PARALLEL" ] && RANGE="$RANGE%$MAX_PARALLEL"
  # Slurm writes the output files where sbatch is run, so run it from slurm_run/ like the other scripts
  CMD=(sbatch --chdir="$SCRIPT_DIR" --partition=shaham-L40-fast --job-name="bbf_suft_26g_s$SHORT_SIZE"
       --array="$RANGE" "$SCRIPT" "$SHORT_SIZE")
  if [ -n "$DRY_RUN" ]; then
    echo "${CMD[@]}"
  else
    echo -n "short_term_buffer_size $SHORT_SIZE (tasks $RANGE): "
    "${CMD[@]}" || exit 1
  fi
done
