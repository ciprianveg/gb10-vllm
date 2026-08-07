"""Minimal compile test for the single-pass MLA kernel. Uses tiny tensors to avoid OOM."""
import torch
from b12x.attention.mla.kernel import run_sparse_mla_kernel, clear_sparse_mla_kernel_cache

def test_compile():
    # Tiny sizes - just enough to trigger kernel compilation
    # GLM-5.1 MLA: q has 128 heads, v_head_dim=512, packed kv cache = 656 bytes/row
    num_queries = 1
    num_heads = 128  # must be 128 for MLA
    q_head_dim = 576  # 512 nope + 64 rope
    v_head_dim = 512
    kv_bytes = 656  # packed KV cache row width in bytes -> 164 u32
    
    q_all = torch.randn((num_queries, num_heads, q_head_dim), dtype=torch.bfloat16, device="cuda")
    kv_cache = torch.zeros((4, 1, kv_bytes), dtype=torch.uint8, device="cuda")
    page_table_1 = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device="cuda")
    active_token_counts = torch.tensor([4], dtype=torch.int32, device="cuda")
    output = torch.zeros((num_queries, num_heads, v_head_dim), dtype=torch.bfloat16, device="cuda")
    
    try:
        run_sparse_mla_kernel(
            q_all=q_all,
            kv_cache=kv_cache,
            page_table_1=page_table_1,
            active_token_counts=active_token_counts,
            sm_scale=0.0417,  # 1/sqrt(576)
            output=output,
        )
        print("Kernel compiled and ran successfully!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        clear_sparse_mla_kernel_cache()

if __name__ == "__main__":
    test_compile()
