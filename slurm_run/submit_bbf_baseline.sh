#!/bin/bash
# Submit bbf_baseline.sh as one job array with a task for every (game, run) pair
#
# Usage:
#   ./submit_bbf_baseline.sh                      the 26 Atari 100k games, runs 0-3
#   ./submit_bbf_baseline.sh Alien Pong           only these games
#   RUNS="0 1" ./submit_bbf_baseline.sh Alien     only these runs
#   MAX_PARALLEL=8 ./submit_bbf_baseline.sh       run at most 8 array tasks at a time
#   DRY_RUN=1 ./submit_bbf_baseline.sh            print the sbatch command and the task list without submitting

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNS=${RUNS:-"0 1 2 3"}

# The 26 games of the Atari 100k benchmark
ATARI_100K="Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand
CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster
MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown"

if [ $# -gt 0 ]; then
  GAMES="$*"
else
  GAMES=$ATARI_100K
fi

# Array task i runs the pair TASKS[i], passed to bbf_baseline.sh as <Game>:<run_number>
TASKS=()
for GAME in $GAMES; do
  if [ ! -f "$SCRIPT_DIR/../bbf/configs/${GAME}.gin" ]; then
    echo "Skipping '$GAME': no bbf/configs/${GAME}.gin" >&2
    continue
  fi
  for RUN_NUMBER in $RUNS; do
    TASKS+=("$GAME:$RUN_NUMBER")
  done
done

if [ ${#TASKS[@]} -eq 0 ]; then
  echo "Nothing to submit" >&2
  exit 1
fi

ARRAY="0-$(( ${#TASKS[@]} - 1 ))"
[ -n "$MAX_PARALLEL" ] && ARRAY="$ARRAY%$MAX_PARALLEL"

CMD=(sbatch --array="$ARRAY" "$SCRIPT_DIR/bbf_baseline.sh" "${TASKS[@]}")
if [ -n "$DRY_RUN" ]; then
  echo "${CMD[@]}"
  for i in "${!TASKS[@]}"; do
    echo "  task $i: ${TASKS[$i]}"
  done
else
  "${CMD[@]}"
fi
