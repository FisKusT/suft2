#!/bin/bash
# Submit bbf_short_long.sh (BBF with the short/long term replay buffer) as one job array per
# short term buffer size, with a task for every (game, run) pair, all on one partition
# Each job array sends one email when the whole array has finished
#
# The per-server entry points set the partition and the buffer sizes and call this script:
#   ./submit_bbf_short_long_shaham.sh     shaham-L40, short term buffer of 2000 and 4000
#   ./submit_bbf_short_long_l40s.sh       L40s-4h,    short term buffer of 8000
# Run them directly (not with sbatch), they call sbatch themselves
#
# Usage:
#   PARTITION=<partition> SHORT_SIZES="<size> ..." ./submit_bbf_short_long.sh [<Game> ...]
# Options (environment variables):
#   PARTITION       Slurm partition of the job arrays (required)
#   SHORT_SIZES     short term buffer sizes, one job array each (required)
#   RUNS            run numbers, default "0 1 2 3"
#   MAX_PARALLEL    run at most this many tasks of each job array at a time
#   MAIL_USER       email address, default fiskust@biu.ac.il
#   DRY_RUN=1       print the sbatch commands and the task lists without submitting
# <Game> ...: only these games, default the 26 Atari 100k games
# The other settings (replay ratio, sampling fraction, tf_lambda, ...) come from bbf/configs/BBF.gin

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# The repo the jobs run in: the one this script is in (it is passed to the jobs, since
# bbf_short_long.sh runs from a copy in the Slurm spool directory)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
RUNS=${RUNS:-"0 1 2 3"}
MAIL_USER=${MAIL_USER:-fiskust@biu.ac.il}

if [ -z "$PARTITION" ] || [ -z "$SHORT_SIZES" ]; then
  echo "Set PARTITION and SHORT_SIZES, or use submit_bbf_short_long_shaham.sh / submit_bbf_short_long_l40s.sh" >&2
  exit 1
fi

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
for SHORT_SIZE in $SHORT_SIZES; do
  # Task i of this job array runs TASKS[i], passed to bbf_short_long.sh as
  # <Game>:<run_number>:<short_term_buffer_size>
  TASKS=()
  for GAME in $GAMES; do
    if [ ! -f "$REPO_DIR/bbf/configs/${GAME}.gin" ]; then
      echo "Skipping '$GAME': no bbf/configs/${GAME}.gin" >&2
      continue
    fi
    for RUN_NUMBER in $RUNS; do
      TASKS+=("$GAME:$RUN_NUMBER:$SHORT_SIZE")
    done
  done

  if [ ${#TASKS[@]} -eq 0 ]; then
    echo "Nothing to submit for short term buffer size $SHORT_SIZE" >&2
    continue
  fi

  ARRAY="0-$(( ${#TASKS[@]} - 1 ))"
  [ -n "$MAX_PARALLEL" ] && ARRAY="$ARRAY%$MAX_PARALLEL"

  # --mail-type=END on a job array sends a single email when all of its tasks have ended,
  # the job name in its subject tells which array it is
  CMD=(sbatch
       --array="$ARRAY"
       --partition="$PARTITION"
       --job-name="bbf_sl_short${SHORT_SIZE}"
       --mail-type=END
       --mail-user="$MAIL_USER"
       --export="ALL,BBF_DIR=$REPO_DIR"
       "$SCRIPT_DIR/bbf_short_long.sh" "${TASKS[@]}")
  echo "Short term buffer $SHORT_SIZE: ${#TASKS[@]} tasks on partition $PARTITION, repo $REPO_DIR"
  if [ -n "$DRY_RUN" ]; then
    echo "  ${CMD[*]:0:8} <${#TASKS[@]} tasks>"
    for i in "${!TASKS[@]}"; do
      echo "    task $i: ${TASKS[$i]}"
    done
  else
    "${CMD[@]}" || exit 1
  fi
  SUBMITTED=$(( SUBMITTED + 1 ))
done

if [ "$SUBMITTED" -eq 0 ]; then
  echo "Nothing to submit" >&2
  exit 1
fi
