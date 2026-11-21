#!/bin/bash
#SBATCH --job-name=bbf_suft_25games_L40s
#SBATCH --partition=L40s-4h
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --array=0-99
#SBATCH --mail-type=END
#SBATCH --mail-user=fiskust@biu.ac.il

# One SUFT configuration of BBF with the short/long term replay buffer on the Atari 100k games,
# except CrazyClimber (already run in the sweep, bbf_suft_sweep.sh tasks 80-83), as one job array
#
# Configuration (from the CrazyClimber sweep):
#   tf_lambda (SUFT weight)          0.1
#   epsilon_train                    0.05
#   sampling_fraction (short term)   0.2
#   short_term_buffer_size           1000
# Fixed: replay_scheme = 'short_long', target_action_selection = True (also the BBF.gin value). Everything else (replay
# ratio, replay capacity, ...) comes from bbf/configs/BBF.gin
#
# Tasks: 25 games x 4 runs (0 1 2 3) = 100 tasks, the run number varies fastest
#
# Usage:
#   sbatch bbf_suft_25games_lam0.1_frac0.2_L40s-4h.sh                  submit all 100 tasks
#   sbatch --array=0-99%20 bbf_suft_25games_lam0.1_frac0.2_L40s-4h.sh  at most 20 tasks at a time
#   sbatch --array=17 bbf_suft_25games_lam0.1_frac0.2_L40s-4h.sh       rerun one task
#   bash bbf_suft_25games_lam0.1_frac0.2_L40s-4h.sh --list             print the task table without submitting
# The repo on the server is BBF_DIR (default below), e.g. BBF_DIR=/path/to/suft2 sbatch bbf_suft_25games_lam0.1_frac0.2_L40s-4h.sh
# The run names match bbf_suft_sweep.sh, so the results sit next to the sweep results in results/

TF_LAMBDA=0.1
EPSILON=0.05
SAMPLING_FRACTION=0.2
SHORT_SIZE=1000

# The 26 games of the Atari 100k benchmark without CrazyClimber
GAMES=(Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand
       DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster
       MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown)
RUNS=(0 1 2 3)

N_GAME=${#GAMES[@]}
N_RUN=${#RUNS[@]}
N_TASKS=$(( N_GAME * N_RUN ))

# Sets GAME RUN_NUMBER for task index $1 (the run number varies fastest)
decode_task() {
  local i=$1
  RUN_NUMBER=${RUNS[$(( i % N_RUN ))]}
  GAME=${GAMES[$(( i / N_RUN ))]}
}

if [ "$1" = "--list" ]; then
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
