#!/bin/bash
#SBATCH --job-name=bbf_suft_sweep
#SBATCH --partition=shaham-L40
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --array=0-719
#SBATCH --mail-type=END
#SBATCH --mail-user=fiskust@biu.ac.il

# SUFT sweep for BBF with the short/long term replay buffer, as one job array (Atari 100k)
#
# Grid (3 x 2 x 3 x 5 = 90 configurations, x 2 games x 4 runs = 720 tasks):
#   tf_lambda (SUFT weight)          0.1 0.5 1
#   epsilon_train                    0 0.05
#   sampling_fraction (short term)   0.1 0.2 0.5
#   short_term_buffer_size           1000 2000 4000 8000 16000
#   games                            CrazyClimber Kangaroo
#   runs                             0 1 2 3
# Fixed: replay_scheme = 'short_long', target_action_selection = True (also the BBF.gin value). Everything else (replay
# ratio, replay capacity, ...) comes from bbf/configs/BBF.gin
#
# Usage:
#   sbatch bbf_suft_sweep.sh                  submit all 720 tasks
#   sbatch --array=0-719%40 bbf_suft_sweep.sh at most 40 tasks at a time
#   sbatch --array=17 bbf_suft_sweep.sh       rerun one task
#   bash bbf_suft_sweep.sh --list             print the task table without submitting
# The repo on the server is BBF_DIR (default below), e.g. BBF_DIR=/path/to/suft2 sbatch bbf_suft_sweep.sh

TF_LAMBDAS=(0.1 0.5 1)
EPSILONS=(0 0.05)
SAMPLING_FRACTIONS=(0.1 0.2 0.5)
SHORT_SIZES=(1000 2000 4000 8000 16000)
GAMES=(CrazyClimber Kangaroo)
RUNS=(0 1 2 3)

N_LAMBDA=${#TF_LAMBDAS[@]}
N_EPS=${#EPSILONS[@]}
N_FRAC=${#SAMPLING_FRACTIONS[@]}
N_SHORT=${#SHORT_SIZES[@]}
N_GAME=${#GAMES[@]}
N_RUN=${#RUNS[@]}
N_TASKS=$(( N_GAME * N_LAMBDA * N_EPS * N_FRAC * N_SHORT * N_RUN ))

# Sets GAME TF_LAMBDA EPSILON SAMPLING_FRACTION SHORT_SIZE RUN_NUMBER for task index $1
# (the run number varies fastest, the game slowest)
decode_task() {
  local i=$1
  RUN_NUMBER=${RUNS[$(( i % N_RUN ))]};                i=$(( i / N_RUN ))
  SHORT_SIZE=${SHORT_SIZES[$(( i % N_SHORT ))]};       i=$(( i / N_SHORT ))
  SAMPLING_FRACTION=${SAMPLING_FRACTIONS[$(( i % N_FRAC ))]}; i=$(( i / N_FRAC ))
  EPSILON=${EPSILONS[$(( i % N_EPS ))]};               i=$(( i / N_EPS ))
  TF_LAMBDA=${TF_LAMBDAS[$(( i % N_LAMBDA ))]};        i=$(( i / N_LAMBDA ))
  GAME=${GAMES[$i]}
}

if [ "$1" = "--list" ]; then
  echo "$N_TASKS tasks (#SBATCH --array must be 0-$(( N_TASKS - 1 )))"
  printf "%5s  %-13s %9s %8s %6s %6s %4s\n" task game tf_lambda epsilon frac short run
  for (( t = 0; t < N_TASKS; t++ )); do
    decode_task $t
    printf "%5d  %-13s %9s %8s %6s %6s %4s\n" $t "$GAME" "$TF_LAMBDA" "$EPSILON" \
      "$SAMPLING_FRACTION" "$SHORT_SIZE" "$RUN_NUMBER"
  done
  exit 0
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "Submit with sbatch (it is a job array), or run 'bash $0 --list' to see the tasks" >&2
  exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_TASKS" ]; then
  echo "No task for array index $SLURM_ARRAY_TASK_ID ($N_TASKS tasks)" >&2
  exit 1
fi
decode_task "$SLURM_ARRAY_TASK_ID"

# The repo on this server (the job runs from a copy of this script in the Slurm spool directory)
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
RR=$(awk -v r="$REPLAY_RATIO" -v b="$BATCH_SIZE" 'BEGIN { print r / b }')

BUFFER=bbf.replay_memory.subsequence_replay_buffer.ShortLongTermJaxSubsequenceParallelEnvReplayBuffer
# String values in double quotes: bbf/train.py strips single quotes from the bindings
GIN_BINDINGS=(
  "--gin_bindings=BBFAgent.replay_scheme = \"short_long\""
  "--gin_bindings=BBFAgent.target_action_selection = True"
  "--gin_bindings=BBFAgent.tf_lambda = $TF_LAMBDA"
  "--gin_bindings=JaxDQNAgent.epsilon_train = $EPSILON"
  "--gin_bindings=$BUFFER.sampling_fraction = $SAMPLING_FRACTION"
  "--gin_bindings=$BUFFER.short_term_buffer_size = $SHORT_SIZE"
)

RUN_NAME=suft_sweep_bbf_rr${RR}_tas_short${SHORT_SIZE}_frac${SAMPLING_FRACTION}_eps${EPSILON}_tf_lambda${TF_LAMBDA}_run${RUN_NUMBER}_${GAME}_100k
LOG_FILE=results/$RUN_NAME.log

echo "Task $SLURM_ARRAY_TASK_ID/$N_TASKS: $GAME run $RUN_NUMBER, tf_lambda $TF_LAMBDA, epsilon_train $EPSILON," \
     "sampling_fraction $SAMPLING_FRACTION, short_term_buffer_size $SHORT_SIZE -> $LOG_FILE"

source venv/bin/activate
mkdir -p results

# bbf/train.py prints the replay buffer configuration at the top of the log
CUDA_VISIBLE_DEVICES=0 python -u -m bbf.train --gin_files="$GIN_FILE" --base_dir="results/$RUN_NAME" --run_number="$RUN_NUMBER" \
  "${GIN_BINDINGS[@]}" > "$LOG_FILE" 2>&1
