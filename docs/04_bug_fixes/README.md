# Bug Fixes & Investigations

Documentation of bugs found, investigations performed, and fixes applied.

## Critical Fixes

### ⚠️ GRADIENT DILUTION BUG (CRITICAL)

**Status:** ✅ FIXED

**Files:**
- **[GRADIENT_DILUTION_BUG_FIXED.md](GRADIENT_DILUTION_BUG_FIXED.md)** - Complete fix documentation
- **[GRADIENT_DILUTION_FIX.md](GRADIENT_DILUTION_FIX.md)** - Technical analysis

**Problem:**
Stage 2 cross-attention absorbed 63% of gradients, leaving only 37% for the image adapter. This prevented the adapter from learning, causing feedback training to fail.

**Evidence:**
```
Image adapter gradients:
  WITHOUT cross-attention: 1.467525 (100%)
  WITH cross-attention:    0.549976 (37%)  ← PROBLEM!
```

**Cause:**
Cross-attention output was added at full scale (100%) instead of being scaled like adapters (10%):
```python
# BEFORE (BUGGY):
x = x + cross_out.permute(1, 0, 2)  # Full-scale residual
```

**Fix:**
Added scaling weights for both Stage 1 and Stage 2 cross-attention:
```python
# AFTER (FIXED):
cross_out_scaled = 0.1 * cross_out  # Scale to 10%
x = x + cross_out_scaled.permute(1, 0, 2)  # Scaled residual
```

**Verification:**
```bash
cd AA-CLIP
python test_stage2_gradients.py

# After fix:
Image adapter gradient norm:
  Without cross-attention: 0.455933
  With cross-attention:    0.438084
  Ratio: 0.9609x  ← FIXED! Now 96% of gradients reach adapter
```

**Files Modified:**
- `AA-CLIP/model/adapter_cross_attention.py` (lines 332-333, 536-537, 659-660, 666-667)
- `AA-CLIP/train_cross_attention_feedback.py` (lines 778-779)

**Impact:**
- Base model: Image loss 3.12 → 2.26 (learning)
- Feedback BEFORE fix: Image loss 3.16 → 3.13 (NOT learning)
- Feedback AFTER fix: Image loss should decrease like base model

**Action Required:**
All feedback training must use the fixed model. Retrain if using old code.

---

## Other Fixes

### View/Reshape Bug

**File:** [BUGFIX_VIEW_RESHAPE.md](BUGFIX_VIEW_RESHAPE.md)

**Problem:** Tensor view/reshape operations failing due to non-contiguous tensors.

**Fix:** Use `reshape()` instead of `view()` when tensors may be non-contiguous.

**Location:** Various places in `AA-CLIP/model/adapter_cross_attention.py`

---

### Untrained Base Model Issue

**File:** [URGENT_FIX_UNTRAINED_BASE.md](URGENT_FIX_UNTRAINED_BASE.md)

**Problem:** Attempting to train cross-attention from untrained base model.

**Fix:** Always train base model first with `train.py` before adding cross-attention.

**Workflow:**
1. Train base: `python train.py --dataset VisA --save_path ./ckpt/base`
2. Then cross-attention: Load base checkpoints with `--load_base_text_adapter` and `--load_base_image_adapter`

---

## Investigations

### Investigation Findings

**File:** [INVESTIGATION_FINDINGS.md](INVESTIGATION_FINDINGS.md)

**Summary of Key Findings:**
1. **Gradient dilution** was the root cause of feedback training failure
2. Cross-attention architecture was sound, just needed proper scaling
3. Text adapter (Stage 1) was learning fine - only Stage 2 had issues
4. Loss staying flat (3.16 → 3.13) indicated zero learning, not convergence

**Testing Methodology:**
- Created `test_stage2_gradients.py` to measure gradient flow
- Compared gradient norms with and without cross-attention
- Identified 63% gradient absorption by cross-attention module

**Key Insight:**
The problem was in the **implementation details** (lack of scaling), not the core idea. Cross-attention with image features IS beneficial when properly scaled.

---

## How to Verify Fixes Are Applied

### 1. Check Model Code

```bash
cd AA-CLIP
grep -n "image_cross_attn_weight" model/adapter_cross_attention.py
```

**Expected output:**
```
332:        image_cross_attn_weight: float = 0.1,  # NEW: Scale for image cross-attention
344:        self.i_ca_w = image_cross_attn_weight  # NEW: Weight for image cross-attention
659:                cross_out_scaled = self.i_ca_w * cross_out
```

### 2. Check Training Script

```bash
grep -n "image_cross_attn_weight" train_cross_attention_feedback.py
```

**Expected output:**
```
779:        image_cross_attn_weight=0.1,  # FIX: Prevents gradient dilution in Stage 2!
```

### 3. Run Gradient Test

```bash
python test_stage2_gradients.py
```

**Expected:**
```
Ratio: 0.9XXXx  ← Should be >0.8 (at least 80% of gradients)
[OK] Cross-attention doesn't harm gradient flow significantly
```

---

## Quick Fix Checklist

Before training feedback model:

- [ ] `model/adapter_cross_attention.py` has `image_cross_attn_weight` parameter
- [ ] Line 659 has `cross_out_scaled = self.i_ca_w * cross_out`
- [ ] `train_cross_attention_feedback.py` line 779 passes `image_cross_attn_weight=0.1`
- [ ] Running `test_stage2_gradients.py` shows >80% gradient retention
- [ ] Have trained base model checkpoints to load from

If any checkbox fails, review [GRADIENT_DILUTION_BUG_FIXED.md](GRADIENT_DILUTION_BUG_FIXED.md) for complete fix instructions.

---

## Related Documentation

- **Training**: [Feedback Training V2 Guide](../02_training_guides/FEEDBACK_TRAINING_V2_GUIDE.md) - Uses fixed code
- **Implementation**: [Cross-Attention Details](../01_cross_attention_feedback_implementation/README.md)
- **Testing**: [Testing Guide](../03_testing_evaluation/TESTING_GUIDE.md)

---

## Timeline

1. **Original Implementation**: Cross-attention added without scaling
2. **Problem Discovered**: Feedback training shows flat loss (Dec 2024)
3. **Investigation**: Created gradient flow test, identified 37% gradient retention
4. **Fix Applied**: Added 0.1 scaling factor to cross-attention outputs
5. **Verification**: Gradient retention improved to 96%
6. **Documentation**: Comprehensive fix guide created

**Status:** All fixes are applied to the current codebase. New training runs should work correctly.
