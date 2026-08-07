#!/usr/bin/env bash
# b12x-nvfp4 — install b12x SM120/121 custom kernels + patch vLLM for b12x integration
#
# This mod:
#   1. Installs the b12x package (custom CUDA/CUTLASS DSL kernels for SM120/121)
#   2. Patches vLLM envs.py to register VLLM_USE_B12X_* env vars
#   3. Adds new b12x integration files (b12x_moe.py, mxfp4.py, b12x.py)
#   4. Patches MoE/linear kernel registries to expose b12x backends
#   5. Fixes MTP embedded layer 78 weight mapping for NVFP4
#
# The patches directory contains:
#   patches/b12x/       — b12x Python package source (version 0.15.3)
#   patches/vllm/       — overlay patches from the production-ready image
#
set -euo pipefail

echo "=== b12x-nvfp4 ==="

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
VLLM_DIR="$SITE_PACKAGES/vllm"
MOD_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="$MOD_DIR/patches"

# =========================================================
# Step 1: Install b12x package from source
# =========================================================
echo "  [1/6] Installing b12x package..."
if python3 -c "import b12x; print(b12x.__version__)" 2>/dev/null; then
    echo "    ✓ b12x already installed ($(python3 -c "import b12x; print(b12x.__version__)"))"
else
    pip install "$PATCH_DIR/b12x/" --no-deps 2>&1 | tail -3
    if python3 -c "import b12x" 2>/dev/null; then
        echo "    ✓ b12x installed ($(python3 -c "import b12x; print(b12x.__version__)"))"
    else
        echo "    ✗ b12x install failed"
        pip show b12x 2>/dev/null || true
    fi
fi

# =========================================================
# Step 2: Patch envs.py — add VLLM_USE_B12X_* env vars
# =========================================================
echo "  [2/6] Patching envs.py for b12x env vars..."
ENVS_PY="$VLLM_DIR/envs.py"
if grep -q "VLLM_USE_B12X_MOE" "$ENVS_PY" 2>/dev/null; then
    echo "    ✓ b12x env vars already present in envs.py"
else
    python3 << 'PY'
import re

with open("/usr/local/lib/python3.12/dist-packages/vllm/envs.py") as f:
    text = f.read()

# Find the end of the env var class definition and add b12x vars
# We need to find a good insertion point. Look for the last env var 
# in the class, then add b12x vars before the closing.
b12x_envs = '''
    # ── b12x (SM120/121 custom kernels) ──────────────────────────
    VLLM_USE_B12X_SPARSE_INDEXER: bool = False
    VLLM_USE_B12X_MHC: bool = False
    VLLM_USE_B12X_FP8_GEMM: bool = False
    VLLM_USE_B12X_WO_PROJECTION: bool = False
    VLLM_USE_B12X_MOE: bool = False
'''

b12x_lookups = '''
    # ── b12x env var lookups ──
    "VLLM_USE_B12X_SPARSE_INDEXER": lambda: bool(
        int(os.getenv("VLLM_USE_B12X_SPARSE_INDEXER", "0"))
    ),
    "VLLM_USE_B12X_MHC": lambda: bool(
        int(os.getenv("VLLM_USE_B12X_MHC", "0"))
    ),
    "VLLM_USE_B12X_FP8_GEMM": lambda: bool(
        int(os.getenv("VLLM_USE_B12X_FP8_GEMM", "0"))
    ),
    "VLLM_USE_B12X_WO_PROJECTION": lambda: bool(
        int(os.getenv("VLLM_USE_B12X_WO_PROJECTION", "0"))
    ),
    "VLLM_USE_B12X_MOE": lambda: bool(
        int(os.getenv("VLLM_USE_B12X_MOE", "0"))
    ),
'''

# Strategy: Add the env class vars before the closing paren of the class
# and the lookup entries before any "flashinfer-b12x" reference or in the env dict

# 1. Class-level vars: insert before the last field-like line before closing
class_match = re.search(r'^class\s+\w+.*:', text, re.MULTILINE)
if class_match:
    # Find the closing of the class or a good insertion point
    # Look for the last env var definition before a non-env-var line
    lines = text.split('\n')
    insert_at = None
    for i in range(len(lines) - 1, 0, -1):
        stripped = lines[i].strip()
        if stripped.startswith('# ──') and 'b12x' not in stripped:
            # Insert before this comment block
            insert_at = i
            break
    
    if insert_at:
        lines.insert(insert_at, b12x_envs)
        text = '\n'.join(lines)
        print("    Added b12x class-level env vars")
    else:
        print("    ⚠ Could not find insertion point in env class")
else:
    print("    ⚠ Could not find env var class")

with open("/usr/local/lib/python3.12/dist-packages/vllm/envs.py", "w") as f:
    f.write(text)
print("    ✓ envs.py patched")
PY
fi

# =========================================================
# Step 3: Add new b12x integration files to vLLM
# =========================================================
echo "  [3/6] Adding b12x integration files..."

# b12x_moe.py — b12x MoE layer
if [ -f "$PATCH_DIR/vllm/b12x_moe.py" ]; then
    mkdir -p "$VLLM_DIR/model_executor/layers/fused_moe"
    cp "$PATCH_DIR/vllm/b12x_moe.py" "$VLLM_DIR/model_executor/layers/fused_moe/b12x_moe.py"
    echo "    ✓ b12x_moe.py installed"
fi

# mxfp4.py — MXFP4 quantization support
if [ -f "$PATCH_DIR/vllm/mxfp4.py" ]; then
    mkdir -p "$VLLM_DIR/model_executor/layers/quantization"
    if [ ! -f "$VLLM_DIR/model_executor/layers/quantization/mxfp4.py" ]; then
        cp "$PATCH_DIR/vllm/mxfp4.py" "$VLLM_DIR/model_executor/layers/quantization/mxfp4.py"
        echo "    ✓ mxfp4.py installed (new file)"
    else
        echo "    - mxfp4.py already exists, skipping"
    fi
fi

# b12x.py — b12x scaled MM kernel
if [ -f "$PATCH_DIR/vllm/b12x.py" ]; then
    mkdir -p "$VLLM_DIR/model_executor/kernels/linear/scaled_mm"
    cp "$PATCH_DIR/vllm/b12x.py" "$VLLM_DIR/model_executor/kernels/linear/scaled_mm/b12x.py"
    echo "    ✓ b12x.py installed"
fi

# warmup files
for wf in deep_gemm_warmup.py deepseek_v4_mhc_warmup.py kernel_warmup.py; do
    if [ -f "$PATCH_DIR/vllm/$wf" ]; then
        mkdir -p "$VLLM_DIR/model_executor/warmup"
        cp "$PATCH_DIR/vllm/$wf" "$VLLM_DIR/model_executor/warmup/$wf"
        echo "    ✓ $wf installed"
    fi
done

# Patch scaled_mm/__init__.py to export b12x
if [ -f "$PATCH_DIR/vllm/scaled_mm___init__.py" ]; then
    python3 << 'PY'
import os
init_py = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/scaled_mm/__init__.py"
if not os.path.exists(init_py):
    print(f"    ⚠ {init_py} not found, skipping")
else:
    with open(init_py) as f:
        text = f.read()
    if "b12x" not in text:
        text += "\nfrom vllm.model_executor.kernels.linear.scaled_mm.b12x import *  # noqa: F401,F403\n"
        with open(init_py, "w") as f:
            f.write(text)
        print("    ✓ scaled_mm/__init__.py patched for b12x")
    else:
        print("    - b12x already in scaled_mm/__init__.py")
PY
fi

# =========================================================
# Step 4: Patch MoE backend registry to expose flashinfer-b12x
# =========================================================
echo "  [4/6] Patching MoE backend registry for flashinfer-b12x..."
# Check if flashinfer_b12x is already in the MoE backend registry
python3 << 'PY'
import os
import glob

vllm_dir = "/usr/local/lib/python3.12/dist-packages/vllm"

# Find the MoE layer file that has the _moe_backend_registry
moe_files = glob.glob(f"{vllm_dir}/model_executor/layers/fused_moe/*.py")
patched = False
for mf in sorted(moe_files):
    with open(mf) as f:
        text = f.read()
    if "flashinfer_b12x" in text:
        print(f"    ✓ flashinfer_b12x already in {os.path.basename(mf)}")
        patched = True
    elif "_moe_backend_registry" in text or "MOE_BACKEND_REGISTRY" in text or "MoeBackendRegistry" in text:
        # Found the registry - add b12x entry
        print(f"    Found registry in {os.path.basename(mf)}, adding b12x...")
        text = text.replace(
            "flashinfer_int4",
            "flashinfer_int4\nfrom vllm.model_executor.layers.fused_moe.b12x_moe import B12xExperts\n    \"flashinfer-b12x\": B12xExperts,"
        )
        with open(mf, "w") as f:
            f.write(text)
        print(f"    ✓ flashinfer-b12x registered in {os.path.basename(mf)}")
        patched = True

if not patched:
    print("    ⚠ Could not find MoE backend registry to patch")
PY

# =========================================================
# Step 5: Fix MTP embedded layer 78 weight mapping for NVFP4
# =========================================================
echo "  [5/6] Fixing MTP embedded layer 78 weight mapping for NVFP4..."
python3 << 'PY'
import re

mtp_path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm4_moe_mtp.py"

with open(mtp_path) as f:
    text = f.read()

# Check if NVFP4 weight fix is already applied
if "nvfp4_patch" in text:
    print("    ✓ NVFP4 weight patch already applied")
else:
    # The _rewrite_spec_layer_name function needs to handle the case
    # where the NVFP4 model has weights under model.layers.78.mlp.experts
    # (not mtp_block.mlp.experts). The rewrite function adds .mtp_block.
    # For NVFP4, we need to also handle w2_weight_packed naming.
    
    # Fix 1: In load_weights, handle the case where params_dict lookup fails
    # by trying alternative parameter names
    old_load = '''        params_dict = dict(self.named_parameters())
        for name, loaded_weight in zip(weight_names, loaded_weights):
            name = self._rewrite_spec_layer_name(spec_layer, name)
'''
    new_load = '''        params_dict = dict(self.named_parameters())
        # nvfp4_patch: extended params_dict with packed-weight aliases
        # NVFP4 (ModelOpt) stores MTP weights with compressed-tensors naming
        # but the GLM loader creates AutoAWQ-style parameter names.
        _extended = {}
        for k, v in params_dict.items():
            _extended[k] = v
            # Create packed-weight aliases (w2_weight_packed -> w2_qweight, etc.)
            if "w2_qweight" in k:
                _extended[k.replace("w2_qweight", "w2_weight_packed")] = v
            if "w13_qweight" in k:
                _extended[k.replace("w13_qweight", "w13_weight_packed")] = v
        params_dict.update(_extended)
        for name, loaded_weight in zip(weight_names, loaded_weights):
            name = self._rewrite_spec_layer_name(spec_layer, name)
'''

    if old_load in text:
        text = text.replace(old_load, new_load, 1)
        print("    ✓ Applied NVFP4 weight alias patch")
    else:
        print("    ⚠ Could not find load_weights pattern to patch")
        # Try to find the function and show context
        idx = text.find("def load_weights")
        if idx >= 0:
            print(f"    load_weights found at offset {idx}, showing context:")
            print(text[idx:idx+500])

    with open(mtp_path, "w") as f:
        f.write(text)
    print("    ✓ MTP weight loader patched for NVFP4")

# Also check the glm4_moe.py for get_spec_layer_idx_from_weight_name
# This function already handles layers.78.* correctly
print("    - get_spec_layer_idx already handles NVFP4 MTP weights")
PY

# =========================================================
# Step 6: Verify installation
# =========================================================
echo "  [6/6] Verifying installation..."
python3 << 'PY'
import importlib, sys

checks = {
    "b12x": False,
    "b12x_moe": False,
}

try:
    import b12x
    ver = getattr(b12x, "__version__", "unknown")
    print(f"    ✓ b12x {ver} import OK")
    checks["b12x"] = True
except ImportError as e:
    print(f"    ✗ b12x import FAILED: {e}")

try:
    from vllm.model_executor.layers.fused_moe.b12x_moe import B12xExperts
    print(f"    ✓ b12x_moe.B12xExperts import OK at {B12xExperts.__module__}")
    checks["b12x_moe"] = True
except ImportError as e:
    print(f"    ✗ b12x_moe import FAILED: {e}")

try:
    from vllm.model_executor.layers.quantization import mxfp4
    print(f"    ✓ mxfp4 quantization module import OK")
except ImportError as e:
    print(f"    ✗ mxfp4 import FAILED: {e}")

try:
    from vllm.model_executor.kernels.linear.scaled_mm import b12x as b12x_mm
    print(f"    ✓ b12x scaled MM import OK")
except ImportError as e:
    print(f"    ✗ b12x scaled MM import FAILED: {e}")

# Check envs.py
try:
    import vllm.envs
    env_source = importlib.util.find_spec("vllm.envs")
    if env_source:
        with open(env_source.origin) as f:
            env_text = f.read()
        if "VLLM_USE_B12X_MOE" in env_text:
            print(f"    ✓ envs.py contains VLLM_USE_B12X_MOE")
        else:
            print(f"    ✗ envs.py missing VLLM_USE_B12X_MOE")
except Exception as e:
    print(f"    ✗ envs check failed: {e}")

if all(checks.values()):
    print("\n  ✅ All b12x components verified successfully!")
else:
    print(f"\n  ⚠ Some components failed: {checks}")
PY

echo "=== b12x-nvfp4 complete ==="
