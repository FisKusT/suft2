#!/bin/bash
# Submit the BBF short/long term buffer runs of the L40s-4h server: one job array with a short
# term buffer of 8000, the 26 Atari 100k games, runs 0-3, and one email to fiskust@biu.ac.il
# when the array has finished
# Run it directly (not with sbatch), it calls sbatch itself
#
# Usage:
#   ./submit_bbf_short_long_l40s.sh                      the job array
#   ./submit_bbf_short_long_l40s.sh Alien Pong           only these games
#   RUNS, MAX_PARALLEL, MAIL_USER, DRY_RUN=1             see submit_bbf_short_long.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PARTITION=${PARTITION:-L40s-4h}
export SHORT_SIZES=${SHORT_SIZES:-"8000"}
exec "$SCRIPT_DIR/submit_bbf_short_long.sh" "$@"
