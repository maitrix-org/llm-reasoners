#! /bin/bash

# usage: matching-bw-generator.sh <base name> <n>

set -euo pipefail

cd "$(dirname "$0")"

./../bwstates.1/bwstates -n $2 > temp.blocks || true
./../bwstates.1/bwstates -n $2 >> temp.blocks || true

./2pddl-typed -d temp.blocks -n $2 > $1-typed.pddl
./2pddl-untyped -d temp.blocks -n $2 > $1-untyped.pddl

rm -f temp.blocks
