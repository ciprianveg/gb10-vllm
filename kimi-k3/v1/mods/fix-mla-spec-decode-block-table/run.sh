#!/bin/bash
set -e
echo "--- Applying MLA spec-decode block table expansion fix..."
python3 << 'PYEOF'
with open("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/triton_mla.py", "r") as f:
    lines = f.readlines()

# Find the block to replace
for i, line in enumerate(lines):
    if "if not attn_metadata.causal:" in line and "block_table = attn_metadata.decode.block_table" in lines[i-2]:
        start = i
        # Find end of block (blank line after the if-block)
        for j in range(i, len(lines)):
            if lines[j].strip() == "" and j > i + 5:
                end = j
                break
        else:
            end = i + 10

        new_lines = lines[:start] + [
            "        # Expand block table and seq_lens for multi-token decode (spec decode / DSpark).\n",
            "        # When num_decode_tokens > num_decodes, each request contributes multiple\n",
            "        # query rows; repeat each request's block table row and seq_len accordingly.\n",
            "        query_len = attn_metadata.num_decode_tokens // attn_metadata.num_decodes\n",
            "        if query_len > 1:\n",
            "            block_table = block_table.repeat_interleave(query_len, dim=0)\n",
            "            seq_lens = seq_lens.repeat_interleave(query_len)\n",
            "\n",
        ] + lines[end:]

        with open("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/triton_mla.py", "w") as f:
            f.writelines(new_lines)
        print("Patch applied successfully")
        break
else:
    print("ERROR: Could not find target block")
    exit(1)
PYEOF
echo "=== OK"