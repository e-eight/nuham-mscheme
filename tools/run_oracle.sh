#!/bin/bash
# Build and run the imsrg++ m-scheme 2B oracle in the imsrg-dev podman image.
# Needs the imsrg++ build from nuham-jax's refs/imsrg/build.sh in $WORK/imsrg (default /tmp/nuham-imsrg).
# usage: [MODE=file|file+trel|trel HW=16 A=4] run_oracle.sh FILE.me2j[.gz] FILE_EMAX FILE_E2MAX QUERIES_IN VALUES_OUT
# QUERIES_IN: one query per line, "2 <4 states>" or "1 <2 states>" (see imsrg_mscheme_2b.cc).
set -euo pipefail
WORK=${WORK:-/tmp/nuham-imsrg}
HERE=$(cd "$(dirname "$0")" && pwd)
F=$(basename "$1")
mkdir -p "$WORK/imsrg/tools" "$WORK/imsrg/data"
cp "$HERE/imsrg_mscheme_2b.cc" "$WORK/imsrg/tools/"
[ "$(realpath "$1")" = "$(realpath -m "$WORK/imsrg/data/$F")" ] || cp "$1" "$WORK/imsrg/data/$F"
cp "$4" "$WORK/imsrg/tools/tuples.in"
podman run --rm -e LD_LIBRARY_PATH=/work/build -v "$WORK/imsrg:/work:Z" -w /work/tools imsrg-dev bash -c "
  S=/work/src
  rm -f oracle2b;  c++ -include cstdint -O2 -DNDEBUG -DNO_HDF -fopenmp -std=gnu++14 \
    -I\$S -I\$S/Commutator -I\$S/imsrg_util -I\$S/ReferenceImplementations -I\$S/ThreeBodyME \
    -I\$S/extern/armadillo -I\$S/extern/half/include \
    imsrg_mscheme_2b.cc -o oracle2b -L/work/build -lIMSRG /work/build/extern/boost_src/libIMSRGBoostZip.a \
    -lz -lgsl -lgslcblas -lopenblas -Wl,-rpath,/work/build
  ./oracle2b /work/data/$F $2 $3 ${MODE:-file} ${HW:-16} ${A:-4} < tuples.in > values.out"
grep '^V ' "$WORK/imsrg/tools/values.out" | cut -c3- > "$5"
