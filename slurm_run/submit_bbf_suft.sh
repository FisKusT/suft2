#!/bin/bash
# Submit bbf_suft.sh (BBF + SUFT, single replay buffer) as one job array per tf_lambda, with a
# task for every (game, run) pair. Default: tf_lambda 0 and 1, epsilon_train 0.05, the 26 Atari
# 100k games, runs 0-3 -> 2 job arrays of 104 tasks, 208 runs in total
# Each job array sends one email when the whole array has finished
# Run it directly (not with sbatch), it calls sbatch itself
#
# Usage:
#   ./submit_bbf_suft.sh                          the 2 job arrays
#   ./submit_bbf_suft.sh Alien Pong               only these games
# Options (environment variables):
#   TF_LAMBDAS      SUFT weights, one job array each, default "0 1"
#   EPSILON         epsilon_train, default 0.05
#   RUNS            run numbers, default "0 1 2 3"
#   PARTITION       Slurm partition, default shaham-L40
#   MAX_PARALLEL    run at most this many tasks of each job array at a time
#   MAIL_USER       email address, default fiskust@biu.ac.il
#   DRY_RUN=1       print the sbatch commands and the task lists without submitting
# The other settings (replay ratio, replay scheme, ...) come from bbf/configs/BBF.gin

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# The repo the jobs run in: the one this script is in (it is passed to the jobs, since
# bbf_suft.sh runs from a copy in the Slurm spool directory)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
TF_LAMBDAS=${TF_LAMBDAS:-"0 1"}
EPSILON=${EPSILON:-0.05}
RUNS=${RUNS:-"0 1 2 3"}
PARTITION=${PARTITION:-shaham-L40}
MAIL_USER=${MAIL_USER:-fiskust@biu.ac.il}

# The 26 games of the Atari 100k benchmark
ATARI_100K="Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand
CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster
MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown"

if [ $# -gt 0 ]; then
  GAMES="$*"
else
  GAMES=$ATARI_100K
fi

SUBMITTED=0
TOTAL=0
for TF_LAMBDA in $TF_LAMBDAS; do
  # Task i of this job array runs TASKS[i], passed to bbf_suft.sh as
  # <Game>:<run_number>:<tf_lambda>:<epsilon_train>
  TASKS=()
  for GAME in $GAMES; do
    if [ ! -f "$REPO_DIR/bbf/configs/${GAME}.gin" ]; then
      echo "Skipping '$GAME': no bbf/configs/${GAME}.gin" >&2
      continue
    fi
    for RUN_NUMBER in $RUNS; do
      TASKS+=("$GAME:$RUN_NUMBER:$TF_LAMBDA:$EPSILON")
    done
  done

  if [ ${#TASKS[@]} -eq 0 ]; then
    echo "Nothing to submit for tf_lambda $TF_LAMBDA" >&2
    continue
  fi

  ARRAY="0-$(( ${#TASKS[@]} - 1 ))"
  [ -n "$MAX_PARALLEL" ] && ARRAY="$ARRAY%$MAX_PARALLEL"

  # --mail-type=END on a job array sends a single email when all of its tasks have ended,
  # the job name in its subject tells which array it is
  CMD=(sbatch
       --array="$ARRAY"
       --partition="$PARTITION"
       --job-name="bbf_suft_lambda${TF_LAMBDA}_eps${EPSILON}"
       --mail-type=END
       --mail-user="$MAIL_USER"
       --export="ALL,BBF_DIR=$REPO_DIR"
       "$SCRIPT_DIR/bbf_suft.sh" "${TASKS[@]}")
  echo "tf_lambda $TF_LAMBDA, epsilon_train $EPSILON: ${#TASKS[@]} tasks on partition $PARTITION, repo $REPO_DIR"
  if [ -n "$DRY_RUN" ]; then
    echo "  ${CMD[*]:0:8} <${#TASKS[@]} tasks>"
    for i in "${!TASKS[@]}"; do
      echo "    task $i: ${TASKS[$i]}"
    done
  else
    "${CMD[@]}" || exit 1
  fi
  SUBMITTED=$(( SUBMITTED + 1 ))
  TOTAL=$(( TOTAL + ${#TASKS[@]} ))
done

if [ "$SUBMITTED" -eq 0 ]; then
  echo "Nothing to submit" >&2
  exit 1
fi
echo "$SUBMITTED job arrays, $TOTAL tasks"
