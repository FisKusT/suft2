#!/bin/bash
#SBATCH --job-name=bbf_base
#SBATCH --partition=shaham-L40
#SBATCH --gres=gpu:1
#SBATCH --mem=100G

# Baseline: a single BBF run on one Atari game (Atari 100k)
#
# Usage: sbatch bbf_baseline.sh <Game> <run_number>     e.g. sbatch bbf_baseline.sh Alien 0
# <Game> is the name of the gin config in bbf/configs (Alien, KungFuMaster, ...)
# To submit many games / runs at once as a job array use submit_bbf_baseline.sh

# In a job array the arguments are <Game>:<run_number> pairs and each task runs
# the pair at index $SLURM_ARRAY_TASK_ID
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  TASKS=("$@")
  TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
  if [ -z "$TASK" ]; then
    echo "No task for array index $SLURM_ARRAY_TASK_ID ($# tasks given)" >&2
    exit 1
  fi
  set -- "${TASK%%:*}" "${TASK#*:}"
fi

GAME=$1
RUN_NUMBER=$2
if [ -z "$GAME" ] || [ -z "$RUN_NUMBER" ]; then
  echo "Usage: sbatch bbf_baseline.sh <Game> <run_number>" >&2
  exit 1
fi

BBF_DIR=/home/dsi/fiskustal/talfiskus/suft2/suft2
GIN_FILE=bbf/configs/${GAME}.gin

cd "$BBF_DIR" || exit 1
if [ ! -f "$GIN_FILE" ]; then
  echo "No gin config for game '$GAME': $BBF_DIR/$GIN_FILE" >&2
  exit 1
fi

# Replay ratio in gradient updates per env step, as in the BBF paper: gin replay_ratio / batch_size
# (the game config wins over BBF.gin if it overrides the value)
gin_value() {
  sed -n "s/^BBFAgent\.$1 *= *\([0-9_]\+\).*/\1/p" bbf/configs/BBF.gin "$GIN_FILE" | tail -1 | tr -d _
}
REPLAY_RATIO=$(gin_value replay_ratio)
BATCH_SIZE=$(gin_value batch_size)
if [ -z "$REPLAY_RATIO" ] || [ -z "$BATCH_SIZE" ]; then
  echo "Could not read BBFAgent.replay_ratio / BBFAgent.batch_size from the gin configs" >&2
  exit 1
fi
RR=$(( REPLAY_RATIO / BATCH_SIZE ))
[ "$RR" -lt 1 ] && RR=1

RUN_NAME=baseline_bbf_rr${RR}_run${RUN_NUMBER}_${GAME}_100k

source venv/bin/activate
mkdir -p results

CUDA_VISIBLE_DEVICES=0 python -u -m bbf.train --gin_files="$GIN_FILE" --base_dir="results/$RUN_NAME" --run_number="$RUN_NUMBER" \
  > "results/$RUN_NAME.log" 2>&1
