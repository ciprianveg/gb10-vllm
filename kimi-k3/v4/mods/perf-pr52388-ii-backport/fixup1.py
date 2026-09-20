from pathlib import Path
p = Path("vllm/models/kimi_k3/nvidia/kda_metadata.py")
s = p.read_text()
need = "    mamba_aligned_state_indices: torch.Tensor | None = None"
anchor = "class KimiK3KDAMetadataBuilder(GDNAttentionMetadataBuilder):"
if need not in s:
    assert anchor in s, "fixup1 anchor missing"
    s = s.replace(anchor, anchor + "\n" + need, 1)
    p.write_text(s)
print("fixup1 ok")
