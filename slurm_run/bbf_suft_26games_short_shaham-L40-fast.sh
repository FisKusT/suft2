#!/bin/bash
#SBATCH --job-name=bbf_suft_26g_s16000
#SBATCH --partition=shaham-L40-fast
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --array=0-103
#SBATCH --mail-type=END
#SBATCH --mail-user=fiskust@biu.ac.il

# One SUFT configuration of BBF with the short/long term replay buffer on the 26 Atari 100k games, for one short
# term buffer size given as the argument, as one job array. --mail-type=END sends one email when the whole array ends
#
# Configuration:
#   tf_lambda (SUFT weight)          0.1
#   epsilon_train                    0.05
#   sampling_fraction (short term)   0.2
#   short_term_buffer_size           the argument, default 16000 (e.g. 2000 4000 8000 16000)
# Fixed: replay_scheme = 'short_long', target_action_selection = True (also the BBF.gin value). Everything else (replay
# ratio, replay capacity, ...) comes from bbf/configs/BBF.gin
#
# Tasks: 26 games x 4 runs (0 1 2 3) = 104 tasks, the run number varies fastest
# A task whose log already holds all the evaluations is skipped, so finished runs are never overwritten
# (e.g. CrazyClimber with short 2000 and 4000, already run in bbf_suft_sweep.sh)
#
# Usage:
#   sbatch bbf_suft_26games_short_shaham-L40-fast.sh       submit the array, short term buffer size 16000
#   sbatch bbf_suft_26games_short_shaham-L40-fast.sh 8000          submit one array, short term buffer size 8000
#   sbatch --array=17 bbf_suft_26games_short_shaham-L40-fast.sh 8000   rerun one task
#   bash bbf_suft_26games_short_shaham-L40-fast.sh --list 8000     print the task table without submitting
# The repo on the server is BBF_DIR (default below), e.g. BBF_DIR=/path/to/suft2 sbatch bbf_suft_26games_short_shaham-L40-fast.sh 8000
# The run names match bbf_suft_sweep.sh, so the results sit next to the sweep results in results/

TF_LAMBDA=0.1
EPSILON=0.05
SAMPLING_FRACTION=0.2

# The 26 games of the Atari 100k benchmark
GAMES=(Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber
       DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster
       MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown)
RUNS=(0 1 2 3)

N_GAME=${#GAMES[@]}
N_RUN=${#RUNS[@]}
N_TASKS=$(( N_GAME * N_RUN ))

LIST=0
if [ "$1" = "--list" ]; then
  LIST=1
  shift
fi
SHORT_SIZE=${1:-16000}
if ! [[ "$SHORT_SIZE" =~ ^[0-9]+$ ]]; then
  echo "Usage: sbatch $0 <short_term_buffer_size>   (or bash $0 --list <short_term_buffer_size>)" >&2
  exit 1
fi

# Sets GAME RUN_NUMBER for task index $1 (the run number varies fastest)
decode_task() {
  local i=$1
  RUN_NUMBER=${RUNS[$(( i % N_RUN ))]}
  GAME=${GAMES[$(( i / N_RUN ))]}
}

if [ "$LIST" = 1 ]; then
  echo "$N_TASKS tasks (#SBATCH --array must be 0-$(( N_TASKS - 1 )))," \
       "tf_lambda $TF_LAMBDA, epsilon_train $EPSILON, sampling_fraction $SAMPLING_FRACTION, short_term_buffer_size $SHORT_SIZE"
  printf "%5s  %-15s %4s\n" task game run
  for (( t = 0; t < N_TASKS; t++ )); do
    decode_task $t
    printf "%5d  %-15s %4s\n" $t "$GAME" "$RUN_NUMBER"
  done
  exit 0
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "Submit with sbatch (it is a job array), or run 'bash $0 --list $SHORT_SIZE' to see the tasks" >&2
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
NUM_ITERATIONS=$(gin_value 'Runner\.num_iterations')
if [ -z "$REPLAY_RATIO" ] || [ -z "$BATCH_SIZE" ] || [ -z "$NUM_ITERATIONS" ]; then
  echo "Could not read BBFAgent.replay_ratio / BBFAgent.batch_size / Runner.num_iterations from the gin configs" >&2
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

# A finished run (one evaluation per iteration in its log) is not run again
if [ -f "$LOG_FILE" ]; then
  DONE_EVALS=$(grep -c "Average undiscounted return per evaluation episode" "$LOG_FILE")
  if [ "$DONE_EVALS" -ge "$NUM_ITERATIONS" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID/$N_TASKS: $GAME run $RUN_NUMBER, short_term_buffer_size $SHORT_SIZE already finished" \
         "($DONE_EVALS evaluations in $LOG_FILE), skipping"
    exit 0
  fi
fi

echo "Task $SLURM_ARRAY_TASK_ID/$N_TASKS: $GAME run $RUN_NUMBER, tf_lambda $TF_LAMBDA, epsilon_train $EPSILON," \
     "sampling_fraction $SAMPLING_FRACTION, short_term_buffer_size $SHORT_SIZE -> $LOG_FILE"

source venv/bin/activate
mkdir -p results

# bbf/train.py prints the replay buffer configuration at the top of the log
CUDA_VISIBLE_DEVICES=0 python -u -m bbf.train --gin_files="$GIN_FILE" --base_dir="results/$RUN_NAME" --run_number="$RUN_NUMBER" \
  "${GIN_BINDINGS[@]}" > "$LOG_FILE" 2>&1
