#!/usr/bin/env python3
"""
Fix PP2 MTP embed_tokens sharing.

Root cause: _maybe_share_embeddings in llm_base_proposer.py has a
`if get_pp_group().world_size == 1:` gate that skips ALL embedding sharing
under PP2. The draft model's embed_tokens on rank 1 is never loaded →
garbage → ~8% acceptance.

Fix: Remove the gate. On rank 0 (first PP rank), share target's real
embed_tokens with draft. On rank 1+ (non-first PP ranks), target has
PPMissingLayer → delete draft's embed_tokens (it receives hidden states
from rank 0 via PP communication, so embed_tokens is not needed).
"""
import re, os

CANDIDATES = [
    "/opt/kimi-k3/vllm/vllm/v1/spec_decode/llm_base_proposer.py",
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/spec_decode/llm_base_proposer.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py",
]
FILE_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
if FILE_PATH is None:
    print("ERROR: Could not find llm_base_proposer.py in any known location")
    exit(1)
print(f"Patching: {FILE_PATH}")

with open(FILE_PATH, "r") as f:
    content = f.read()

# === Step 1: Find the _maybe_share_embeddings function ===
# Match from `def _maybe_share_embeddings` to the next `def ` at same indentation
func_pattern = r'(    def _maybe_share_embeddings\(self, target_language_model: nn\.Module\) -> None:\n)(.*?)(\n    def )'
func_match = re.search(func_pattern, content, re.DOTALL)
if not func_match:
    print("ERROR: Could not find _maybe_share_embeddings function")
    exit(1)

old_func_body = func_match.group(2)

# === Step 2: Build the new function body ===
new_func_body = '''        """
        Some draft models may not have their own embedding layers, and some may
        have a duplicate copy of the target model's embedding layers. In these cases,
        we share the target model's embedding layers with the draft model to save
        memory.

        Under pipeline parallelism (PP > 1), the target model's embed_tokens may be
        a PPMissingLayer on non-first ranks. In that case, we skip sharing on those
        ranks and let the draft model use its own embedding (or skip loading).
        """
        from vllm.model_executor.models.utils import PPMissingLayer

        inner_model = getattr(target_language_model, "model", None)
        if inner_model is None:
            raise AttributeError("Target model does not have 'model' attribute")
        if hasattr(inner_model, "embed_tokens"):
            target_embed_tokens = inner_model.embed_tokens
        elif hasattr(inner_model, "embedding"):
            target_embed_tokens = inner_model.embedding
        else:
            raise AttributeError(
                "Target model does not have 'embed_tokens' or 'embedding' attribute"
            )

        # Under PP > 1, non-first ranks have PPMissingLayer for embed_tokens.
        # Skip sharing on those ranks — the draft model receives hidden states
        # from the target model's previous PP rank, so embed_tokens is not needed.
        if isinstance(target_embed_tokens, PPMissingLayer):
            logger.info(
                "Target embed_tokens is PPMissingLayer on this PP rank. "
                "Skipping embedding sharing — draft model will not use "
                "embed_tokens on non-first PP ranks."
            )
            if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
                del self.model.model.embed_tokens
                self.model.model.embed_tokens = None
            return

        share_embeddings = False
        if hasattr(self.model, "has_own_embed_tokens"):
            # EAGLE model
            if not self.model.has_own_embed_tokens:
                share_embeddings = True
                logger.info(
                    "Detected EAGLE model without its own embed_tokens in the"
                    " checkpoint. Sharing target model embedding weights with the"
                    " draft model."
                )
            elif (
                isinstance(target_embed_tokens.weight, torch.Tensor)
                and hasattr(self.model.model, "embed_tokens")
                and isinstance(self.model.model.embed_tokens.weight, torch.Tensor)
                # TODO: Offload to CPU for comparison to avoid extra GPU memory
                # usage in CI testing environments with limited GPU memory
                and torch.equal(
                    target_embed_tokens.weight.cpu(),
                    self.model.model.embed_tokens.weight.cpu(),
                )
            ):
                share_embeddings = True
                logger.info(
                    "Detected EAGLE model with embed_tokens identical to the target"
                    " model. Sharing target model embedding weights with the draft"
                    " model."
                )
            else:
                logger.info(
                    "Detected EAGLE model with distinct embed_tokens weights. "
                    "Keeping separate embedding weights from the target model."
                )
        else:
            # MTP model
            share_embeddings = True
            logger.info(
                "Detected MTP model. "
                "Sharing target model embedding weights with the draft model."
            )

        if share_embeddings:
            draft_embed = self.model.model.embed_tokens
            # Only share when both models use the same embedding width.
            # Guard with isinstance so non-Tensor weights (e.g. in tests)
            # are not affected — mirrors the weight-equality check above.
            if isinstance(target_embed_tokens.weight, torch.Tensor) and isinstance(
                draft_embed.weight, torch.Tensor
            ):
                target_dim = target_embed_tokens.weight.shape[-1]
                draft_dim = draft_embed.weight.shape[-1]
                if target_dim != draft_dim:
                    share_embeddings = False
                    logger.info(
                        "Target embedding dim (%d) differs from draft "
                        "embedding dim (%d). Keeping separate embedding "
                        "weights.",
                        target_dim,
                        draft_dim,
                    )

        if share_embeddings:
            if hasattr(self.model.model, "embed_tokens"):
                del self.model.model.embed_tokens
            self.model.model.embed_tokens = target_embed_tokens'''

# === Step 3: Replace the function body ===
new_content = content[:func_match.start(2)] + new_func_body + content[func_match.end(2):]

with open(FILE_PATH, "w") as f:
    f.write(new_content)

# Clear bytecode cache
vllm_dir = os.path.dirname(os.path.dirname(FILE_PATH))
for root, dirs, files in os.walk(vllm_dir):
    for name in files:
        if name.endswith('.pyc'):
            os.remove(os.path.join(root, name))
    for name in dirs:
        if name == '__pycache__':
            import shutil
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)

print("Done! Patched _maybe_share_embeddings for PP2 MTP support.")
print(f"  - Removed world_size == 1 gate")
print(f"  - Added PPMissingLayer detection for non-first PP ranks")
print(f"  - Added hasattr guard for embed_tokens access")
