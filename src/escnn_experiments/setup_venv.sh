#!/usr/bin/env bash
# Creates the isolated environment these experiments need.
#
# Why an isolated environment at all: escnn imports its SO(3) machinery
# unconditionally, so `import escnn` requires lie_learn -- and lie_learn 0.0.2
# is a Cython extension built against the numpy 1.x C ABI. Under the numpy 2.2.6
# the repo root pins, importing it dies with
#   ImportError: numpy.core.multiarray failed to import
# so this is a genuine ABI incompatibility, not a conservative version pin.
#
# Why "thin": torch + the nvidia CUDA wheels are ~6.3 GB. Reinstalling them here
# just to change numpy's version is wasteful, so this venv installs only the
# numpy-ABI-sensitive packages locally and shares torch/torchvision from the
# repo-root venv through a .pth file. Local site-packages precedes the appended
# shared path, so the local numpy 1.26.4 wins while torch resolves to the shared
# copy. Result is ~285 MB instead of ~6.6 GB.
#
# If you have disk to spare and would rather have a fully standalone env, skip
# this script and just `pip install -r requirements.txt`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_VENV="$HERE/../../.venv"

if [ ! -d "$MAIN_VENV" ]; then
    echo "error: expected the repo-root venv at $MAIN_VENV (this script shares its torch)." >&2
    echo "       create it first, or install requirements.txt standalone instead." >&2
    exit 1
fi

python3 -m venv "$HERE/.venv"
"$HERE/.venv/bin/pip" install --upgrade pip

# Everything except torch/torchvision, which are shared below.
"$HERE/.venv/bin/pip" install \
    'numpy==1.26.4' 'lie_learn==0.0.2' 'py3nj==0.2.1' 'scipy==1.15.3' \
    'autograd==1.8.0' 'pymanopt==2.2.1'
"$HERE/.venv/bin/pip" install --no-deps 'escnn==1.0.11'

# Share torch/torchvision from the root venv. Named zzz_* so it sorts last and
# is appended after the local site-packages, keeping local numpy authoritative.
SITE="$("$HERE/.venv/bin/python3" -c 'import site; print(site.getsitepackages()[0])')"
python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" \
    "$MAIN_VENV/lib/python3.10/site-packages" > "$SITE/zzz_shared_torch.pth"

echo
echo "verifying..."
"$HERE/.venv/bin/python3" - <<'PY'
import numpy, torch, scipy
assert numpy.__version__.startswith("1.26"), f"local numpy should win, got {numpy.__version__}"
import escnn
from escnn import gspaces, nn as enn
enn.FieldType(gspaces.rot2dOnR2(N=8), [gspaces.rot2dOnR2(N=8).regular_repr])
print(f"  ok: numpy {numpy.__version__}, torch {torch.__version__}, "
      f"cuda {torch.cuda.is_available()}, escnn imports")
PY

echo
echo "done. run experiments with:"
echo "  src/escnn_experiments/.venv/bin/python3 src/escnn_experiments/<script>.py"
