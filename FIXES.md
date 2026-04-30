# Fix Log — papers/3 Generated Code

## Fix 1: `algorithms/rank_expander.py` — missing `expand_rank_random` and `expand_rank_qr`
**Error:** `cannot import name 'expand_rank_random' from 'algorithms.rank_expander'`
**Root cause:** `algorithms/__init__.py` exports these two functions, but they were never written into `rank_expander.py`
**Fix:** Added two standalone wrapper functions at end of file

## Fix 2: `algorithms/sparsity_constraint.py` — missing `enforce_sparsity` and `compute_sparsity_penalty` as module-level functions
**Error:** `cannot import name 'enforce_sparsity' from 'algorithms.sparsity_constraint'`
**Root cause:** These exist as *methods* on the `SparsityConstraint` class but `__init__.py` expects them as standalone functions
**Fix:** Added two wrapper functions at end of file that delegate to the class

## Fix 3: `utils/evaluation.py` — missing `compute_rouge_metrics`
**Error:** `cannot import name 'compute_rouge_metrics' from 'utils.evaluation'`
**Root cause:** Function referenced in `trainers/` files but never written
**Fix:** Added implementation using `rouge_score` library

## Fix 4: `datasets/` folder — name collision with HuggingFace `datasets` package
**Error:** `cannot import name 'load_dataset' from 'datasets'` (resolved local package instead of HuggingFace)
**Root cause:** Local folder named `datasets/` shadows the installed `datasets` library on `sys.path`
**Fix:** Renamed folder to `data_loaders/`, updated all internal cross-references

## Fix 5: `models/llama_wrapper.py` — bare module imports without package prefix
**Error:** `No module named 'adapter_base'`, `No module named 'adapter_manager'`, `No module named 'weightlora_layer'`
**Root cause:** Generated imports like `from adapter_base import ...` instead of `from models.adapter_base import ...`
**Fix:** Prefixed all three imports with `models.`

## Fix 6: `rouge_score` — missing dependency
**Error:** `No module named 'rouge_score'`
**Root cause:** Not installed, not in requirements.txt (which also doesn't exist)
**Fix:** `pip install rouge-score`

## Fix 7: `models/adapter_base.py` — wrong matrix multiplication order in forward pass
**Error:** `mat1 and mat2 shapes cannot be multiplied (64x32 and 2x64)` at runtime
**Root cause:** `LoRAAdapter.forward()` computed `torch.matmul(delta_w, x)` where `delta_w=(in, out)` and `x=(batch, in)`. Should be `x @ delta_w`.
**Fix:** Changed to `torch.matmul(x, delta_w)` — correct LoRA forward: `h = x @ (A @ B) * scaling`

## Fix 8: `algorithms/rank_expander.py` — rank expansion concatenates A on wrong dimension
**Error:** After `expand_rank_random`, `A` shape was `(68, 4)` instead of `(64, 8)` — rows added instead of columns
**Root cause:** `torch.cat([old_A, torch.randn(r_new-r_old, old_A.shape[1])], dim=0)` — A is `(in_features, rank)`, new rank columns should extend dim=1, not dim=0
**Fix:** Changed to `torch.cat([old_A, torch.randn(old_A.shape[0], r_new-r_old)], dim=1)`

## Fix 9: `algorithms/rank_expander.py` — QR strategy doesn't extend B
**Error:** After `expand_rank_qr`, `A=(64,8)` but `B=(4,32)` → `A @ B` would fail (incompatible dims)
**Root cause:** Code had comment "B remains the same (or can be extended with zeros)" and kept `B_extended = old_B` — but B must be extended to match new rank
**Fix:** Extended B: `torch.cat([old_B, torch.zeros(r_new-r_old, old_B.shape[1])], dim=0)`

---
## Summary
| Category | Count |
|---|---|
| Missing functions (written in __init__ but not in file) | 2 fixes |
| Wrong import paths (bare names instead of package.module) | 2 fixes |
| Package name collision | 1 fix |
| Missing dependency | 1 fix |
| Wrong matmul operand order (logic error in forward pass) | 1 fix |
| Wrong dimension in rank expansion (A extended on wrong axis) | 1 fix |
| Incomplete rank expansion (B not extended to match new rank) | 1 fix |
| **Total** | **9 fixes** |

## Fix 10: `models/adapter_base.py` — `create_lora_adapter` case mismatch
**Error:** `ValueError: Unsupported layer type: Linear`
**Root cause:** Factory checked `layer_type == 'linear'` but `AdapterManager` passes `type(layer).__name__` = `'Linear'` (capital L)
**Fix:** `lt = layer_type.lower()` before comparison

## Fix 11: `models/adapter_manager.py` + `models/adapter_base.py` — `LoRALinear` missing `.A`/`.B` attributes
**Error:** `AttributeError: 'LoRALinear' object has no attribute 'A'` in `AdapterManager.initialize_adapters`
**Root cause:** `AdapterManager` accesses `adapter.A.shape[0]` but `LoRALinear` stores the adapter as `self.lora_adapter` (a nested `LoRAAdapter`)
**Fix:** Added read-only proxy properties `A`, `B`, `r` to `LoRALinear` that delegate to `self.lora_adapter`

## Fix 12: `algorithms/rank_expander.py` — can't assign `nn.Parameter` through property on `nn.Module`
**Error:** `KeyError: "attribute 'A' already exists"` when `rank_expander` does `adapter.A = nn.Parameter(...)`
**Root cause:** `nn.Module.__setattr__` intercepts the assignment before Python's property descriptor can. Since `A` is not in `_parameters` of `LoRALinear` (it's a property), PyTorch refuses the re-registration
**Fix:** Added `update_lora_params(A, B, r)` method to `LoRALinear` that sets on the inner adapter. Added `_set_adapter_params()` helper in `rank_expander.py` that uses this method when available

---
## Summary
| Category | Count |
|---|---|
| Missing functions (written in __init__ but not in file) | 2 fixes |
| Wrong import paths (bare names instead of package.module) | 2 fixes |
| Package name collision | 1 fix |
| Missing dependency | 1 fix |
| Wrong matmul operand order (logic error in forward pass) | 1 fix |
| Wrong dimension in rank expansion (A extended on wrong axis) | 1 fix |
| Incomplete rank expansion (B not extended to match new rank) | 1 fix |
| Case mismatch in factory function | 1 fix |
| Missing attribute proxy (LoRALinear vs LoRAAdapter interface) | 1 fix |
| nn.Module.__setattr__ conflict with property setter | 1 fix |
| **Total** | **12 fixes** |

**Import-level:** 6 fixes — all 22 modules import cleanly
**Runtime/logic-level:** 6 fixes — forward pass math, rank expansion shapes, integration between classes
