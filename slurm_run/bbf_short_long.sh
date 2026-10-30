#!/bin/bash
#SBATCH --job-name=sfq_sl
#SBATCH --partition=shaham-L40
#SBATCH --gres=gpu:1
#SBATCH --mem=100G

# SF-Q with the short/long term replay buffer: a single run on one Atari game (Atari 100k)
# The short term buffer (most recent transitions) is sampled uniformly and the long term
# buffer (the rest of the replay capacity) is sampled with BBF's prioritized replay
# The SUFT loss is added only to the short term buffer samples
#
# Usage: sbatch bbf_short_long.sh <Game> <run_number>     e.g. sbatch bbf_short_long.sh Alien 0
# <Game> is the name of the gin config in bbf/configs (Alien, KungFuMaster, ...)
# The replay scheme, the buffer sizes and the sampling fraction are set in bbf/configs/BBF.gin

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
  echo "Usage: sbatch bbf_short_long.sh <Game> <run_number>" >&2
  exit 1
fi

BBF_DIR=/home/dsi/fiskustal/talfiskus/suft2/suft2
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
if [ -z "$REPLAY_RATIO" ] || [ -z "$BATCH_SIZE" ] ||[ -z "$SHORT_SIZE" ] || [ -z "$SAMPLING_FRACTION" ]; then
  echo "Could not read the replay settings from the gin configs" >&2
  exit 1
fi
RR=$(awk "BEGIN { print $REPLAY_RATIO / $BATCH_SIZE }")

RUN_NAME=short_long_sfq_rr${RR}_short${SHORT_SIZE}_frac${SAMPLING_FRACTION}_run${RUN_NUMBER}_${GAME}_100k
LOG_FILE=results/$RUN_NAME.log

source venv/bin/activate
mkdir -p results

# bbf/train.py prints the replay buffer configuration at the top of the log
CUDA_VISIBLE_DEVICES=0 python -u -m bbf.train --gin_files="$GIN_FILE" --base_dir="results/$RUN_NAME" --run_number="$RUN_NUMBER" \
  > "$LOG_FILE" 2>&1
