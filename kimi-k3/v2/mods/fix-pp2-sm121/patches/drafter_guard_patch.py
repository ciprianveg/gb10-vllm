#!/usr/bin/env python3
"""
Fix V1 model runner PP drafter — comprehensive fix v14:
1. Add self.drafter = None for non-last PP ranks
2. Guard all self.drafter.dummy_run() calls
3. Guard ALL assert isinstance(self.drafter, ...) including multiline union types
4. Guard self.drafter.initialize_cudagraph_keys()
5. In _dummy_run: skip speculative block when self.drafter is None
6. Guard all remaining self.speculative_config conditions that access self.drafter
"""
import re

FILE_PATH = "/opt/kimi-k3/vllm/vllm/v1/worker/gpu_model_runner.py"

with open(FILE_PATH, "r") as f:
    content = f.read()

# === Fix 1: Add else: self.drafter = None before self.num_spec_tokens = 0 ===
pattern1 = r'((?:.*\n)*?)([ \t]*self\.num_spec_tokens = 0)'
match1 = re.search(pattern1, content)
if match1:
    before = match1.group(1)
    if 'else:' in before and 'self.drafter = None' in before:
        print('Fix 1: already applied')
    else:
        num_spec_line = match1.group(2)
        indent = re.match(r'(\s*)', num_spec_line).group(1)
        insert = f'{indent}else:\n{indent}    # On non-last PP ranks, drafter is not created.\n{indent}    self.drafter = None\n\n'
        content = content[:match1.start(2)] + insert + content[match1.start(2):]
        print('Fix 1: applied')
else:
    print('Fix 1: pattern not found')

# === Fix 2: Guard self.drafter.dummy_run(num_tokens=1) ===
for m in reversed(list(re.finditer(r'([ \t]*)self\.drafter\.dummy_run\(num_tokens=1\)', content))):
    indent = m.group(1)
    ls = content.rfind('\n', 0, m.start()) + 1
    if 'if self.drafter is not None:' in content[ls:m.end()]:
        continue
    content = content[:m.start()] + f'{indent}if self.drafter is not None:\n{indent}    self.drafter.dummy_run(num_tokens=1)' + content[m.end():]
    print(f'Fix 2: guarded at {m.start()}')

# === Fix 3: Guard multiline self.drafter.dummy_run(\n  num_tokens, ...) ===
for m in reversed(list(re.finditer(r'([ \t]*)self\.drafter\.dummy_run\(\s*\n\s*num_tokens,', content))):
    indent = m.group(1)
    ls = content.rfind('\n', 0, m.start()) + 1
    if 'if self.drafter is not None:' in content[max(0,ls-200):ls]:
        continue
    content = content[:m.start()] + f'{indent}if self.drafter is not None:\n{indent}    self.drafter.dummy_run(\n                        num_tokens,' + content[m.end():]
    print(f'Fix 3: guarded at {m.start()}')

# === Fix 4a: Guard SINGLE-LINE assert isinstance(self.drafter, TypeName) ===
for m in reversed(list(re.finditer(r'([ \t]*)assert isinstance\(self\.drafter,\s*(\w+)\)', content))):
    indent = m.group(1)
    tn = m.group(2)
    ls = content.rfind('\n', 0, m.start()) + 1
    if 'if self.drafter is not None:' in content[max(0,ls-200):ls]:
        continue
    content = content[:m.start()] + f'{indent}if self.drafter is not None:\n{indent}    assert isinstance(self.drafter, {tn})' + content[m.end():]
    print(f'Fix 4a: guarded ({tn}) at {m.start()}')

# === Fix 4b: Guard MULTILINE assert isinstance(self.drafter, ...) ===
ml_pattern = r'([ \t]*)assert isinstance\(\s*\n(\s*)self\.drafter,\s*\n((?:\s*\|?\s*\w+\s*\n?)+,\s*\n\s*\))'
for m in reversed(list(re.finditer(ml_pattern, content))):
    indent = m.group(1)
    tb = m.group(3)
    ls = content.rfind('\n', 0, m.start()) + 1
    if 'if self.drafter is not None:' in content[max(0,ls-200):ls]:
        continue
    content = content[:m.start()] + f'{indent}if self.drafter is not None:\n{indent}    assert isinstance(\n{indent}        self.drafter,\n{indent}        {tb}' + content[m.end():]
    print(f'Fix 4b: guarded multiline at {m.start()}')

# === Fix 5: Guard self.drafter.initialize_cudagraph_keys ===
for m in reversed(list(re.finditer(r'([ \t]*)self\.drafter\.initialize_cudagraph_keys\(', content))):
    indent = m.group(1)
    ls = content.rfind('\n', 0, m.start()) + 1
    if 'if self.drafter is not None:' in content[max(0,ls-200):ls]:
        continue
    le = content.find('\n', m.start())
    fl = content[m.start():le]
    content = content[:m.start()] + f'{indent}if self.drafter is not None:\n{indent}    {fl.strip()}' + content[le:]
    print(f'Fix 5: guarded at {m.start()}')

# === Fix 6: In _dummy_run, skip speculative block when drafter is None ===
dp = r'([ \t]*)if self\.speculative_config and \(\s*\n(\s*)self\.speculative_config\.use_eagle\(\)\s*\n(\s*)or self\.speculative_config\.uses_draft_model\(\)\s*\n(\s*)or self\.speculative_config\.uses_extract_hidden_states\(\)\s*\n(\s*)\):'
for m in reversed(list(re.finditer(dp, content))):
    bt = content[max(0,m.start()-500):m.start()]
    if 'hidden_states = outputs' in bt and 'if self.drafter is not None' not in bt:
        content = content[:m.start()] + m.group(0).replace('if self.speculative_config and (', 'if self.speculative_config and self.drafter is not None and (') + content[m.end():]
        print(f'Fix 6: added drafter guard in _dummy_run at {m.start()}')

# === Fix 7: Guard ALL remaining self.speculative_config conditions that access self.drafter ===
# Pattern 1: if self.speculative_config and (\n  ...use_eagle() or uses_draft_model()...\n):
# Pattern 2: if (\n  self.speculative_config\n  and self.speculative_config.uses_extract_hidden_states()\n):
# We look for these patterns and add `self.drafter is not None` if not already present.

# 7a: if self.speculative_config and (\n  use_eagle()\n  or uses_draft_model()\n):
spec_pat_7a = r'([ \t]*)if self\.speculative_config and \(\s*\n(\s*)self\.speculative_config\.use_eagle\(\)\s*\n(\s*)or self\.speculative_config\.uses_draft_model\(\)\s*\n(\s*)\):'
for m in reversed(list(re.finditer(spec_pat_7a, content))):
    bt = content[max(0,m.start()-500):m.start()]
    # Skip if this is the _dummy_run one (already fixed by Fix 6)
    if 'hidden_states = outputs' in bt:
        continue
    if 'self.drafter is not None' in m.group(0):
        continue
    content = content[:m.start()] + m.group(0).replace('if self.speculative_config and (', 'if self.speculative_config and self.drafter is not None and (') + content[m.end():]
    print(f'Fix 7a: guarded speculative_config condition at {m.start()}')

# 7b: if self.speculative_config and (\n  use_eagle()\n  or uses_extract_hidden_states()\n):
spec_pat_7b = r'([ \t]*)if self\.speculative_config and \(\s*\n(\s*)self\.speculative_config\.use_eagle\(\)\s*\n(\s*)or self\.speculative_config\.uses_extract_hidden_states\(\)\s*\n(\s*)\):'
for m in reversed(list(re.finditer(spec_pat_7b, content))):
    if 'self.drafter is not None' in m.group(0):
        continue
    content = content[:m.start()] + m.group(0).replace('if self.speculative_config and (', 'if self.speculative_config and self.drafter is not None and (') + content[m.end():]
    print(f'Fix 7b: guarded speculative_config condition at {m.start()}')

# 7c: if (\n  self.speculative_config\n  and self.speculative_config.uses_extract_hidden_states()\n):
spec_pat_7c = r'([ \t]*)if \(\s*\n(\s*)self\.speculative_config\s*\n(\s*)and self\.speculative_config\.uses_extract_hidden_states\(\)\s*\n(\s*)\):'
for m in reversed(list(re.finditer(spec_pat_7c, content))):
    if 'self.drafter is not None' in m.group(0):
        continue
    content = content[:m.start()] + m.group(0).replace('if (', 'if (self.drafter is not None and') + content[m.end():]
    print(f'Fix 7c: guarded speculative_config condition at {m.start()}')

with open(FILE_PATH, "w") as f:
    f.write(content)

print(f"\n✓ All fixes applied ({content.count(chr(10))} lines)")
