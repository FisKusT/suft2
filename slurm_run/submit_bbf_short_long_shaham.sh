#!/bin/bash
# Submit the BBF short/long term buffer runs of the shaham-L40 server: one job array per short
# term buffer size (2000 and 4000), each with the 26 Atari 100k games, runs 0-3, and one email
# to fiskust@biu.ac.il when the array has finished
# Run it directly (not with sbatch), it calls sbatch itself
#
# Usage:
#   ./submit_bbf_short_long_shaham.sh                    both job arrays
#   ./submit_bbf_short_long_shaham.sh Alien Pong         only these games
#   SHORT_SIZES="2000" ./submit_bbf_short_long_shaham.sh only this size
#   RUNS, MAX_PARALLEL, MAIL_USER, DRY_RUN=1             see submit_bbf_short_long.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PARTITION=${PARTITION:-shaham-L40}
export SHORT_SIZES=${SHORT_SIZES:-"2000 4000"}
exec "$SCRIPT_DIR/submit_bbf_short_long.sh" "$@"
