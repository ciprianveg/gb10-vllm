#!/usr/bin/env bash
# fix-b12x-cute-cache-key: Deterministic CuTe DSL cache keys (B12X PR #41)
#
# Root cause: b12x/cute/compiler.py line 2009 uses repr(dsl_compile_options)
# to build a cache key. repr(OptLevel(2)) produces:
#   <cutlass.base_dsl.compiler.OptLevel object at 0xe64d36ebf6e0>
# The memory address differs per process/restart, so cache keys never match,
# causing SparseNSATiledTopkKernel and other CuTeDSL kernels to recompile
# on every startup instead of using the disk cache.
#
# Fix: replace repr() with a deterministic serialization based on the
# object's __dict__ (e.g. {'_value': 2} -> "OptLevel(_value=2)").
#
# This is the runtime equivalent of B12X PR #41 (merged in v19 B12X c7dc733).
# Applies to v18 B12X bc85ef3.

set -euo pipefail

# Auto-detect b12x install path
if [ -f "/opt/venv/lib/python3.12/site-packages/b12x/cute/compiler.py" ]; then
    FILE="/opt/venv/lib/python3.12/site-packages/b12x/cute/compiler.py"
elif [ -f "/usr/local/lib/python3.12/dist-packages/b12x/cute/compiler.py" ]; then
    FILE="/usr/local/lib/python3.12/dist-packages/b12x/cute/compiler.py"
else
    echo "ERROR: b12x/cute/compiler.py not found"
    exit 1
fi
echo "  Using: $FILE"

# Check if already patched
if grep -q "OptLevel(" "$FILE" 2>/dev/null && grep -q "_dsl_compile_options_key_deterministic" "$FILE" 2>/dev/null; then
    echo "  fix-b12x-cute-cache-key: already applied"
    exit 0
fi

python3 -c "
import sys

FILE = '$FILE'

with open(FILE, 'r') as f:
    content = f.read()

OLD = 'kwargs[\"__dsl_compile_options_key\"] = repr(dsl_compile_options)'

NEW = '''kwargs[\"__dsl_compile_options_key\"] = (
            _dsl_compile_options_key_deterministic(dsl_compile_options)
            if hasattr(dsl_compile_options, \"__dict__\")
            else repr(dsl_compile_options)
        )'''

if OLD not in content:
    # Try with different whitespace
    print('ERROR: Could not find repr(dsl_compile_options) pattern')
    sys.exit(1)

content = content.replace(OLD, NEW, 1)

# Add the helper function before the line where it's used
# Find a good insertion point - before the function that uses it
INSERT_AFTER = 'def _short_repr('
HELPER = '''
def _dsl_compile_options_key_deterministic(obj):
    \"\"\"Deterministic cache key for DSL compile options (B12X PR #41).

    repr(OptLevel(2)) includes a process-specific memory address, making
    cache keys non-deterministic across processes and restarts. This
    serializes the object's __dict__ instead.
    \"\"\"
    items = sorted(vars(obj).items())
    parts = [f\"{k}={v!r}\" for k, v in items]
    return f\"{type(obj).__qualname__}({', '.join(parts)})\"

'''

# Insert the helper function before _short_repr
idx = content.find(INSERT_AFTER)
if idx > 0:
    content = content[:idx] + HELPER + content[idx:]
else:
    print('WARNING: Could not find insertion point, prepending helper')
    content = HELPER + content

with open(FILE, 'w') as f:
    f.write(content)

print('  fix-b12x-cute-cache-key: applied deterministic DSL compile options key')
"

# Clear bytecode caches
VLLM_BASE=$(dirname $(dirname $(dirname "$FILE")))
find "$VLLM_BASE/b12x" -name '*.pyc' -delete 2>/dev/null || true
find "$VLLM_BASE/b12x" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

echo "✓ B12X CuTe cache key fix applied"
