#!/usr/bin/env bash
# fix-dsa-block-table-dim: Fix off-by-one in DSA indexer expanded_block_table_buffer
#
# The scheduler's block_table_tensor can have max_num_blocks_per_req+1 columns
# (e.g. 5470 vs 5469 for max_model_len=350000, block_size=64), but the indexer's
# pre-allocated expanded_block_table_buffer only has max_num_blocks_per_req columns.
# This causes a RuntimeError when torch.repeat_interleave produces a tensor wider
# than the buffer during the variable-decode-length flattening path (MTP with
# concurrency > 1).
#
# Fix: add +1 to the buffer column count to accommodate the scheduler's extra block.

set -euo pipefail

# Auto-detect vllm install path (v18 uses /opt/venv, v16 uses /usr/local)
if [ -f "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/indexer.py" ]; then
    INDEXER="/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/indexer.py"
elif [ -f "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py" ]; then
    INDEXER="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py"
else
    echo "ERROR: indexer.py not found in any known location"
    exit 1
fi
echo "  Using: $INDEXER"

# Check if already patched
if grep -q 'max_num_blocks_per_req + 1' "$INDEXER" 2>/dev/null; then
    echo "Already patched: expanded_block_table_buffer uses max_num_blocks_per_req + 1"
    exit 0
fi

# Patch: after the cdiv computation of max_num_blocks_per_req, add +1
# The pattern we're looking for:
#   max_num_blocks_per_req = cdiv(
#       self.vllm_config.model_config.max_model_len,
#       self.kv_cache_spec.block_size * get_total_cp_world_size(),
#   )
#   self.expanded_block_table_buffer = torch.zeros(
#       (
#           scheduler_config.max_num_batched_tokens,
#           max_num_blocks_per_req,
#
# We insert: max_num_blocks_per_req += 1
# between the cdiv and the buffer allocation.

python3 -c "
import re

with open('$INDEXER', 'r') as f:
    content = f.read()

# Pattern: find the max_num_blocks_per_req computation followed by expanded_block_table_buffer allocation
old = '''        max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        )
        self.expanded_block_table_buffer = torch.zeros(
            (
                scheduler_config.max_num_batched_tokens,
                max_num_blocks_per_req,'''

new = '''        max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        )
        # +1: scheduler block_table_tensor may have one extra column beyond
        # cdiv(max_model_len, block_size) to handle alignment/edge cases.
        max_num_blocks_per_req += 1
        self.expanded_block_table_buffer = torch.zeros(
            (
                scheduler_config.max_num_batched_tokens,
                max_num_blocks_per_req,'''

if old not in content:
    # Try alternative whitespace (tabs vs spaces)
    print('WARNING: primary pattern not found, trying alt pattern')
    # Just do a targeted sed-style replacement
    import sys
    sys.exit(1)

content = content.replace(old, new, 1)

with open('$INDEXER', 'w') as f:
    f.write(content)

print('Patched expanded_block_table_buffer allocation: max_num_blocks_per_req += 1')
"
