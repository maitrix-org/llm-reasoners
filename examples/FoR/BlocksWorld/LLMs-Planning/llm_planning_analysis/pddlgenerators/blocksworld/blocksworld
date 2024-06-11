#! /bin/bash

set -euo pipefail

OPS=$1  # Either 3 or 4.
BLOCKS=$2

# Get full directory name of the script (https://stackoverflow.com/a/246128).
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

BS=${SCRIPT_DIR}/bwstates.1/bwstates
TR=${SCRIPT_DIR}/${OPS}ops/2pddl/2pddl

${BS} -s 2 -n ${BLOCKS} > STATES
${TR} -d STATES -n ${BLOCKS}

rm STATES
