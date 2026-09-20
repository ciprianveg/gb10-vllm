from pathlib import Path
p = Path("vllm/v1/worker/gpu/model_states/mamba_hybrid.py")
s = p.read_text()
anchor = "        mamba_attn_metadata = MambaHybridAttnMetadata("
block = (
    "        if self._align_mode:\n"
    "            mamba_group_ids, _ = self._get_mamba_group_info(kv_cache_config)\n"
    "            aligned_index_builders = []\n"
    "            for group_idx, group_id in enumerate(mamba_group_ids):\n"
    "                for group in attn_groups[group_id]:\n"
    "                    builder = group.get_metadata_builder(0)\n"
    "                    if hasattr(builder, \"mamba_aligned_state_indices\"):\n"
    "                        aligned_index_builders.append((group_idx, builder))\n"
    "            if aligned_index_builders:\n"
    "                ctx = self._ensure_align_ctx(\n"
    "                    kv_cache_config, mamba_group_ids, block_tables\n"
    "                )\n"
    "                all_group_indices = ctx.compute_aligned_state_indices(\n"
    "                    input_batch.seq_lens, num_reqs\n"
    "                )\n"
    "                for group_idx, builder in aligned_index_builders:\n"
    "                    builder.mamba_aligned_state_indices = all_group_indices[group_idx]\n"
    "\n"
)
if "_get_mamba_group_info(kv_cache_config)" not in s or "mamba_aligned_state_indices = all_group_indices" not in s:
    assert anchor in s, "fixup2 anchor missing"
    s = s.replace(anchor, block + anchor, 1)
    p.write_text(s)
print("fixup2 ok")
