#!/bin/bash
# Submit bbf_baseline.sh for every (game, run) pair
#
# Usage:
#   ./submit_bbf_baseline.sh                      the 26 Atari 100k games, runs 0-7
#   ./submit_bbf_baseline.sh Alien Pong           only these games
#   RUNS="0 1" ./submit_bbf_baseline.sh Alien     only these runs
#   DRY_RUN=1 ./submit_bbf_baseline.sh            print the sbatch commands without submitting

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

for GAME in $GAMES; do
  if [ ! -f "$SCRIPT_DIR/../bbf/configs/${GAME}.gin" ]; then
    echo "Skipping '$GAME': no bbf/configs/${GAME}.gin" >&2
    continue
  fi
  for RUN_NUMBER in $RUNS; do
    CMD=(sbatch --job-name="${GAME,,}${RUN_NUMBER}" "$SCRIPT_DIR/bbf_baseline.sh" "$GAME" "$RUN_NUMBER")
    if [ -n "$DRY_RUN" ]; then
      echo "${CMD[@]}"
    else
      "${CMD[@]}"
    fi
  done
done
