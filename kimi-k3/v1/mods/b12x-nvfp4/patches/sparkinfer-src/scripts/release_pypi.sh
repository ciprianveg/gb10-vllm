set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
VENV_DIR="$ROOT_DIR/.venv-release"

if [[ -f .env ]]; then
  set -a
    source .env
  set +a
fi

: "${TWINE_USERNAME:=__token__}"
: "${TWINE_PASSWORD:?TWINE_PASSWORD must be set in .env or the environment}"

rm -rf dist/

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install --upgrade build twine
"$VENV_DIR/bin/python" -m build
"$VENV_DIR/bin/python" -m twine check dist/*
"$VENV_DIR/bin/python" -m twine upload dist/*
