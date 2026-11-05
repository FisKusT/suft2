#!/bin/bash
#SBATCH --job-name=bbf_sl
#SBATCH --partition=shaham-L40
#SBATCH --gres=gpu:1
#SBATCH --mem=100G

# BBF with the short/long term replay buffer: a single run on one Atari game (Atari 100k)
# The short term buffer (most recent transitions) is sampled uniformly and the long term
# buffer (the rest of the replay capacity) is sampled with BBF's prioritized replay
#
# Usage: sbatch bbf_short_long.sh <Game> <run_number> [<short_term_buffer_size>]
#   e.g. sbatch bbf_short_long.sh Alien 0          short term buffer size from the gin configs
#        sbatch bbf_short_long.sh Alien 0 8000     short term buffer of 8000 transitions
# <Game> is the name of the gin config in bbf/configs (Alien, KungFuMaster, ...)
# The replay scheme, the buffer sizes, the sampling fraction and the SUFT weight (tf_lambda) are
# set in bbf/configs/BBF.gin; <short_term_buffer_size> overrides the short term buffer size

# In a job array the arguments are <Game>:<run_number>[:<short_term_buffer_size>] tasks and
# each array task runs the one at index $SLURM_ARRAY_TASK_ID
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  TASKS=("$@")
  TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
  if [ -z "$TASK" ]; then
    echo "No task for array index $SLURM_ARRAY_TASK_ID ($# tasks given)" >&2
    exit 1
  fi
  IFS=: read -r TASK_GAME TASK_RUN TASK_SHORT <<< "$TASK"
  set -- "$TASK_GAME" "$TASK_RUN" $TASK_SHORT
fi

GAME=$1
RUN_NUMBER=$2
SHORT_SIZE_ARG=$3
if [ -z "$GAME" ] || [ -z "$RUN_NUMBER" ]; then
  echo "Usage: sbatch bbf_short_long.sh <Game> <run_number> [<short_term_buffer_size>]" >&2
  exit 1
fi
if [ -n "$SHORT_SIZE_ARG" ] && ! [[ "$SHORT_SIZE_ARG" =~ ^[0-9]+$ ]]; then
  echo "<short_term_buffer_size> must be a positive integer, got '$SHORT_SIZE_ARG'" >&2
  exit 1
fi

# The repo on this server: passed by the submit scripts (the repo they are run from),
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
BUFFER=bbf\.replay_memory\.subsequence_replay_buffer\.ShortLongTermJaxSubsequenceParallelEnvReplayBuffer

# The replay scheme is set in the gin configs (BBFAgent.replay_scheme), it must be 'short_long'
REPLAY_SCHEME=$(sed -n "s/^BBFAgent\.replay_scheme *= *['\"]\([a-z_]*\)['\"].*/\1/p" bbf/configs/BBF.gin "$GIN_FILE" | tail -1)
if [ "$REPLAY_SCHEME" != "short_long" ]; then
  echo "BBFAgent.replay_scheme is '$REPLAY_SCHEME' in the gin configs, set it to 'short_long' in bbf/configs/BBF.gin" >&2
  exit 1
fi

# Replay ratio in gradient updates per env step, as in the BBF paper: gin replay_ratio / batch_size
REPLAY_RATIO=$(gin_value 'BBFAgent\.replay_ratio')
BATCH_SIZE=$(gin_value 'BBFAgent\.batch_size')
SHORT_SIZE=$(gin_value "$BUFFER\.short_term_buffer_size")
SAMPLING_FRACTION=$(gin_value "$BUFFER\.sampling_fraction")
# The short term buffer size given as an argument overrides the gin configs
GIN_BINDINGS=()
if [ -n "$SHORT_SIZE_ARG" ]; then
  SHORT_SIZE=$SHORT_SIZE_ARG
  GIN_BINDINGS+=("--gin_bindings=${BUFFER//\\/}.short_term_buffer_size = $SHORT_SIZE")
fi
if [ -z "$REPLAY_RATIO" ] || [ -z "$BATCH_SIZE" ] ||[ -z "$SHORT_SIZE" ] || [ -z "$SAMPLING_FRACTION" ]; then
  echo "Could not read the replay settings from the gin configs" >&2
  exit 1
fi
# Formatted as a decimal (0.25, 1, 2, ...) so that RR below 1 gets its own run name
RR=$(awk -v r="$REPLAY_RATIO" -v b="$BATCH_SIZE" 'BEGIN { print r / b }')

# SUFT weight, 0 (disabled) when BBFAgent.tf_lambda is not set
TF_LAMBDA=$(gin_value 'BBFAgent\.tf_lambda')
TF_LAMBDA=${TF_LAMBDA:-0}

RUN_NAME=short_long_bbf_rr${RR}_short${SHORT_SIZE}_frac${SAMPLING_FRACTION}_tf_lambda${TF_LAMBDA}_run${RUN_NUMBER}_${GAME}_100k
LOG_FILE=results/$RUN_NAME.log

source venv/bin/activate
mkdir -p results

# bbf/train.py prints the replay buffer configuration at the top of the log
CUDA_VISIBLE_DEVICES=0 python -u -m bbf.train --gin_files="$GIN_FILE" --base_dir="results/$RUN_NAME" --run_number="$RUN_NUMBER" \
  "${GIN_BINDINGS[@]}" > "$LOG_FILE" 2>&1
