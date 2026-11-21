#!/bin/bash
#SBATCH --job-name=bbf_suft
#SBATCH --partition=shaham-L40-fast
#SBATCH --gres=gpu:1
#SBATCH --mem=100G

# BBF + SUFT with the single (prioritized) replay buffer: one run on one Atari game (Atari 100k)
# The SUFT term tf_lambda * (Q_current(s, a) - Q_old(s, a))^2 is added for every sample and
# scaled by the prioritized replay importance weights. tf_lambda = 0 is plain BBF
#
# Usage: sbatch bbf_suft.sh <Game> <run_number> <tf_lambda> <epsilon_train>
#   e.g. sbatch bbf_suft.sh Alien 0 1 0.05
# <Game> is the name of the gin config in bbf/configs (Alien, KungFuMaster, ...)
# To submit the 26 games as job arrays use submit_bbf_suft.sh
# Everything else (replay ratio, replay scheme, target_action_selection, ...) comes from bbf/configs/BBF.gin

# In a job array the arguments are <Game>:<run_number>:<tf_lambda>:<epsilon_train> tasks and
# each array task runs the one at index $SLURM_ARRAY_TASK_ID
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  TASKS=("$@")
  TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
  if [ -z "$TASK" ]; then
    echo "No task for array index $SLURM_ARRAY_TASK_ID ($# tasks given)" >&2
    exit 1
  fi
  IFS=: read -r TASK_GAME TASK_RUN TASK_LAMBDA TASK_EPS <<< "$TASK"
  set -- "$TASK_GAME" "$TASK_RUN" "$TASK_LAMBDA" "$TASK_EPS"
fi

GAME=$1
RUN_NUMBER=$2
TF_LAMBDA=$3
EPSILON=$4
if [ -z "$GAME" ] || [ -z "$RUN_NUMBER" ] || [ -z "$TF_LAMBDA" ] || [ -z "$EPSILON" ]; then
  echo "Usage: sbatch bbf_suft.sh <Game> <run_number> <tf_lambda> <epsilon_train>" >&2
  exit 1
fi

# The repo on this server: passed by submit_bbf_suft.sh (the repo it is run from),
# or the default below when the script is submitted directly with sbatch
BBF_DIR=${BBF_DIR:-/home/dsi/fiskustal/talfiskus/suft2/suft2}
GIN_FILE=bbf/configs/${GAME}.gin

cd "$BBF_DIR" || exit 1
if [ ! -f "$GIN_FILE" ]; then
  echo "No gin config for game '$GAME': $BBF_DIR/$GIN_FILE" >&2
  exit 1
fi

# Value of a gin parameter (the game config wins over BBF.gin if it overrides the value)
gin_value() {
  sed -n "s/^$1 *= *\([0-9_.]\+\).*/\1/p" bbf/configs/BBF.gin "$GIN_FILE" | tail -1 | tr -d _
}
# Replay ratio in gradient updates per env step, as in the BBF paper: gin replay_ratio / batch_size
REPLAY_RATIO=$(gin_value 'BBFAgent\.replay_ratio')
BATCH_SIZE=$(gin_value 'BBFAgent\.batch_size')
if [ -z "$REPLAY_RATIO" ] || [ -z "$BATCH_SIZE" ]; then
  echo "Could not read BBFAgent.replay_ratio / BBFAgent.batch_size from the gin configs" >&2
  exit 1
fi
# Formatted as a decimal (0.25, 1, 2, ...) so that RR below 1 gets its own run name
RR=$(awk -v r="$REPLAY_RATIO" -v b="$BATCH_SIZE" 'BEGIN { print r / b }')

GIN_BINDINGS=(
  "--gin_bindings=BBFAgent.tf_lambda = $TF_LAMBDA"
  "--gin_bindings=JaxDQNAgent.epsilon_train = $EPSILON"
)

RUN_NAME=suft_bbf_rr${RR}_eps${EPSILON}_tf_lambda${TF_LAMBDA}_run${RUN_NUMBER}_${GAME}_100k
LOG_FILE=results/$RUN_NAME.log

echo "Task ${SLURM_ARRAY_TASK_ID:-single}: $GAME run $RUN_NUMBER, tf_lambda $TF_LAMBDA, epsilon_train $EPSILON," \
     "RR $RR -> $LOG_FILE"

source venv/bin/activate
mkdir -p results

CUDA_VISIBLE_DEVICES=0 python -u -m bbf.train --gin_files="$GIN_FILE" --base_dir="results/$RUN_NAME" --run_number="$RUN_NUMBER" \
  "${GIN_BINDINGS[@]}" > "$LOG_FILE" 2>&1
