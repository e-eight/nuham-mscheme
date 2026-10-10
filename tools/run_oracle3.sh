#!/bin/bash
# Build and run the imsrg++ m-scheme 3B oracle in the imsrg-dev podman image.
# Needs the imsrg++ build from nuham-jax's refs/imsrg/build.sh in $WORK/imsrg (default /tmp/nuham-imsrg).
# usage: run_oracle3.sh FILE.me3j[.gz] EMAX E2MAX E3MAX QUERIES_IN VALUES_OUT
# EMAX, E2MAX, E3MAX are the limits of the file (the model space is set to them).
# QUERIES_IN: one query per line, six states "n l j2 tz2 m2" (see imsrg_mscheme_3b.cc).
set -euo pipefail
WORK=${WORK:-/tmp/nuham-imsrg}
HERE=$(cd "$(dirname "$0")" && pwd)
F=$(basename "$1")
mkdir -p "$WORK/imsrg/tools" "$WORK/imsrg/data"
cp "$HERE/imsrg_mscheme_3b.cc" "$WORK/imsrg/tools/"
[ "$(realpath "$1")" = "$(realpath -m "$WORK/imsrg/data/$F")" ] || cp "$1" "$WORK/imsrg/data/$F"
cp "$5" "$WORK/imsrg/tools/queries3.in"
podman run --rm -e LD_LIBRARY_PATH=/work/build -v "$WORK/imsrg:/work:Z" -w /work/tools imsrg-dev bash -c "
  S=/work/src
  rm -f oracle3b;  c++ -include cstdint -O2 -DNDEBUG -DNO_HDF -fopenmp -std=gnu++14 \
    -I\$S -I\$S/Commutator -I\$S/imsrg_util -I\$S/ReferenceImplementations -I\$S/ThreeBodyME \
    -I\$S/extern/armadillo -I\$S/extern/half/include \
    imsrg_mscheme_3b.cc -o oracle3b -L/work/build -lIMSRG /work/build/extern/boost_src/libIMSRGBoostZip.a \
    -lz -lgsl -lgslcblas -lopenblas -Wl,-rpath,/work/build
  ./oracle3b /work/data/$F $2 $3 $4 < queries3.in > values3.out"
grep '^V ' "$WORK/imsrg/tools/values3.out" | cut -c3- > "$6"
