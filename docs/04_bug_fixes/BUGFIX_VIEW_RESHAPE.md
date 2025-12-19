# Bug Fix: RuntimeError with .view() in Stage 2

## Error Message
```
RuntimeError: view size is not compatible with input tensor's size and stride
(at least one dimension spans across two contiguous subspaces).
Use .reshape(...) instead.
```

## Root Cause

In [adapter_cross_attention.py:463](model/adapter_cross_attention.py#L463), the code was using `.view(-1, 768)` on a tensor that may not be contiguous in memory:

```python
class_text_context = self.text_context_dict[class_name].to(device)
flattened = class_text_context.view(-1, 768)  # ❌ Fails if not contiguous
```

### Why This Happens

When tensors are saved and loaded with `torch.save()`/`torch.load()`, or after certain operations (like transpose, permute, indexing), they may become **non-contiguous** in memory. PyTorch's `.view()` requires the tensor to be contiguous because it creates a new view without copying data.

## Solution

Changed `.view()` to `.reshape()` on line 463:

```python
# Before (line 463)
flattened = class_text_context.view(-1, 768)  # ❌ RuntimeError

# After (line 464)
flattened = class_text_context.reshape(-1, 768)  # ✅ Works always
```

### Why `.reshape()` Works

- `.reshape()` automatically handles non-contiguous tensors
- If the tensor is contiguous, it acts like `.view()` (zero-copy)
- If not contiguous, it makes a copy and then reshapes (slightly slower but safe)

### Alternative Fix (not used)

Could also use `.contiguous().view()`:
```python
flattened = class_text_context.contiguous().view(-1, 768)
```

But `.reshape()` is cleaner and handles both cases automatically.

## Other `.view()` Calls in the File

Checked all other `.view()` usages in [adapter_cross_attention.py](model/adapter_cross_attention.py):

| Line | Code | Safe? | Reason |
|------|------|-------|--------|
| 135-137 | `Q.view(...)`, `K.view(...)`, `V.view(...)` | ✅ | Fresh linear projections (always contiguous) |
| 148 | `output.transpose(...).contiguous().view(...)` | ✅ | Explicitly calls `.contiguous()` first |
| 256-258 | `Q.view(...)`, `K.view(...)`, `V.view(...)` | ✅ | Fresh linear projections (always contiguous) |
| 269 | `output.transpose(...).contiguous().view(...)` | ✅ | Explicitly calls `.contiguous()` first |

**All other `.view()` calls are safe!** ✅

## Testing

After this fix, Stage 2 training should proceed without errors:

```bash
python train_cross_attention.py --dataset MVTec --save_path ckpt/cross_attn
```

Expected output:
```
STAGE 2: IMAGE ADAPTER TRAINING WITH CROSS-ATTENTION
======================================================================
Epoch 0:   0%|                                    | 0/863 [00:00<?, ?it/s]
BATCH 0 DEBUG
  image: torch.Size([2, 3, 518, 518]), class_names: ['tile', 'cable']
[PREPARE TEXT CONTEXT for Stage 2]
  class_names: ['tile', 'cable']
  text_context shape: torch.Size([2, 1232, 768])  # ✅ Should work now
```

## Lessons Learned

### When to use `.view()` vs `.reshape()`

**Use `.view()`:**
- When you know the tensor is contiguous
- In performance-critical code where you want to avoid potential copies
- When working with freshly created tensors (linear layers, etc.)

**Use `.reshape()`:**
- When you're unsure about contiguity
- When working with loaded tensors (torch.load)
- When working with tensors after transpose/permute/indexing operations
- Default choice for safety

**Use `.contiguous().view()`:**
- When you want to explicitly handle non-contiguous cases
- When you want to signal to readers that contiguity matters
- In the middle of operation chains with `.transpose()`, etc.

### Best Practice Pattern

For multi-head attention reshaping:
```python
# ✅ Good pattern (used in lines 148, 269)
output = output.transpose(1, 2).contiguous().view(B, L, D)

# ✅ Alternative (safer, slightly slower if copy needed)
output = output.transpose(1, 2).reshape(B, L, D)
```

## Status

- ✅ **Bug fixed** in [adapter_cross_attention.py:464](model/adapter_cross_attention.py#L464)
- ✅ **All other `.view()` calls verified safe**
- ✅ **Ready for Stage 2 training**

---

**Fix applied:** 2024
**File modified:** `model/adapter_cross_attention.py`
**Lines changed:** 463-464
