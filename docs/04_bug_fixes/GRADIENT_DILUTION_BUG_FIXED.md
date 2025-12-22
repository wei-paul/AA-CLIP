# OUTDATED: Stage 2 Gradient Dilution Fix (Partial Fix)

**⚠️ This document describes a PARTIAL fix that has been SUPERSEDED.**
**✅ See [STAGE2_CROSS_ATTENTION_NORMALIZATION_FIX.md](STAGE2_CROSS_ATTENTION_NORMALIZATION_FIX.md) for the COMPLETE fix.**

---

## Problem Summary (Partial Understanding)

Your feedback training was failing because **cross-attention absorbed 63% of the gradients**, leaving only 37% for the image adapter. This prevented the adapter from learning.

**Note:** This was the correct diagnosis of the symptom, but the root cause was actually **unnormalized text_context and missing norm-matching**, not just gradient absorption. The scaling weights added here helped but didn't fully solve the problem.

## Evidence

### Before Fix
```bash
python test_stage2_gradients.py

Image adapter gradient norm:
  Without cross-attention: 1.467525
  With cross-attention:    0.549976
  Ratio: 0.3748x  ← Only 37% of gradients!

[CAUTION] Cross-attention reduces image adapter gradients significantly
```

### After Fix
```bash
python test_stage2_gradients.py

Image adapter gradient norm:
  Without cross-attention: 0.455933
  With cross-attention:    0.438084
  Ratio: 0.9609x  ← Now 96% of gradients!

[OK] Cross-attention doesn't harm gradient flow significantly
```

## Root Cause

In [adapter_cross_attention.py:656](AA-CLIP/model/adapter_cross_attention.py#L656), the cross-attention output was added at **full scale**:

```python
# BEFORE (BUGGY):
x = x + cross_out.permute(1, 0, 2)  # Full-scale residual
```

This caused the cross-attention module (initialized randomly with Xavier init) to:
1. **Produce large outputs** that dominate the forward pass
2. **Absorb most gradients** during backprop (63%)
3. **Starve the image adapter** of gradients (only 37%)

Compare to how adapters are weighted (adapter.py:99):
```python
x = self.i_w * adapt_out + (1 - self.i_w) * x
#   ↑ 0.1 weight            ↑ 0.9 weight
```

Adapters are scaled to 10% - but cross-attention was at 100%!

## The Fix

Added scaling weights for both Stage 1 and Stage 2 cross-attention:

### 1. Added New Parameters ([adapter_cross_attention.py:332-333](AA-CLIP/model/adapter_cross_attention.py#L332-L333))

```python
def __init__(
    self,
    # ... existing params ...
    text_cross_attn_weight: float = 0.1,   # NEW
    image_cross_attn_weight: float = 0.1,  # NEW
):
    # ...
    self.t_ca_w = text_cross_attn_weight   # Scale for text cross-attention
    self.i_ca_w = image_cross_attn_weight  # Scale for image cross-attention
```

### 2. Fixed Stage 1 Cross-Attention ([adapter_cross_attention.py:536-537](AA-CLIP/model/adapter_cross_attention.py#L536-L537))

```python
# AFTER (FIXED):
cross_out_scaled = self.t_ca_w * cross_out          # Scale to 10%
x = x + cross_out_scaled.permute(1, 0, 2)  # Scaled residual
```

### 3. Fixed Stage 2 Cross-Attention ([adapter_cross_attention.py:666-667](AA-CLIP/model/adapter_cross_attention.py#L666-L667))

```python
# AFTER (FIXED):
cross_out_scaled = self.i_ca_w * cross_out          # Scale to 10%
x = x + cross_out_scaled.permute(1, 0, 2)  # Scaled residual
```

### 4. Updated Training Script ([train_cross_attention_feedback.py:778-779](AA-CLIP/train_cross_attention_feedback.py#L778-L779))

```python
model = AdaptedCLIPWithCrossAttention(
    # ... existing args ...
    text_cross_attn_weight=0.1,   # NEW: Scale cross-attention like adapters
    image_cross_attn_weight=0.1,  # NEW: Prevents gradient dilution!
)
```

## Why This Fixes The Problem

### Gradient Flow Comparison

**Before (37% gradients to adapter):**
```
Loss
  ↓ 100% gradients
Cross-attention module
  ↓ 37% gradients (the rest absorbed!)
Image adapter
```

**After (96% gradients to adapter):**
```
Loss
  ↓ 100% gradients
Cross-attention module (10% scale)
  ↓ 96% gradients (minimal absorption)
Image adapter
```

### Training Loss Prediction

**Before fix** (your actual results):
- Base model: 3.12 → 2.26 (learning!)
- Feedback: 3.16 → 3.13 (not learning)

**After fix** (expected):
- Base model: 3.12 → 2.26 (same)
- Feedback: 3.16 → **2.20** (should learn better!)

The feedback model should now **beat the base model** because:
1. Image adapter gets full gradients
2. Cross-attention adds useful text context (but doesn't dominate)
3. Feedback loop compounds improvements across iterations

## How to Retrain

### Option 1: Retrain from Scratch (Recommended)

```bash
cd AA-CLIP

python train_cross_attention_feedback.py \
    --dataset VisA \
    --save_path ./ckpt/visa_feedback_fixed \
    --num_feedback_loops 3 \
    --text_batch_size 16 \
    --image_batch_size 2 \
    --text_lr 0.00001 \
    --image_lr 0.0005 \
    --load_base_text_adapter ./ckpt/VisA_base_full_1216/text_adapter.pth \
    --load_base_image_adapter ./ckpt/VisA_base_full_1216/image_adapter_20.pth
```

### Option 2: Use V2 Script with Higher LR

Since the fix restores gradient flow, you can now also use the V2 script:

```bash
python train_cross_attention_feedback_v2.py \
    --dataset VisA \
    --save_path ./ckpt/visa_feedback_v2_fixed \
    --num_feedback_loops 5 \
    --image_lr 0.002 \
    --load_base_text_adapter ./ckpt/VisA_base_full_1216/text_adapter.pth \
    --load_base_image_adapter ./ckpt/VisA_base_full_1216/image_adapter_20.pth
```

## What to Monitor

### 1. Image Loss Must Decrease

```
Loop 1 - image loss: 3.16 → 2.95 → 2.75 → ...  ← Should decrease!
```

If loss still stays flat, check:
- Gradients are non-zero: `average_grad_norm > 0.01`
- Cross-attention weight is 0.1 (check model init)
- Using the FIXED adapter_cross_attention.py

### 2. Gradient Norms

The V2 script logs gradient norms. Watch for:
```
Average Grad Norm: 0.05-0.15  ← Healthy range
```

If grad_norm < 0.001, gradients are vanishing.

### 3. Compare to Base Model

After training, your feedback model should:
- **Match or beat base model** on pixel AUC
- **Show better localization** in visualizations
- **Improve across feedback loops** (loop 3 > loop 1)

## Files Modified

1. **[model/adapter_cross_attention.py](AA-CLIP/model/adapter_cross_attention.py)**
   - Added `text_cross_attn_weight` and `image_cross_attn_weight` parameters
   - Scaled cross-attention outputs in both Stage 1 and Stage 2

2. **[train_cross_attention_feedback.py](AA-CLIP/train_cross_attention_feedback.py)**
   - Updated model initialization to use the new scaling parameters

## Testing the Fix

Run the gradient flow test:
```bash
cd AA-CLIP
python test_stage2_gradients.py
```

Expected output:
```
Image adapter gradient norm:
  Without cross-attention: 0.XXXXX
  With cross-attention:    0.XXXXX
  Ratio: 0.9XXXx  ← Should be >0.8

[OK] Cross-attention doesn't harm gradient flow significantly
```

## Technical Details

### Why 0.1 Weight?

The 0.1 weight matches the adapter design (image_adapt_weight=0.1). This ensures:
- **Gradual introduction** of cross-attention features
- **Stable training** (no sudden large changes)
- **Balanced learning** between adapter and cross-attention

### Could We Use Higher Weights?

Yes, you could experiment with:
- `image_cross_attn_weight=0.2` (20% scale, more cross-attention influence)
- `image_cross_attn_weight=0.05` (5% scale, more conservative)

But 0.1 is a good starting point based on the adapter design.

### Learnable Scaling (Future Work)

Instead of fixed 0.1, you could make it learnable:
```python
self.i_ca_w = nn.Parameter(torch.tensor(0.1))
```

This lets the model learn the optimal weighting during training.

## Summary

- **Bug:** Cross-attention absorbed 63% of gradients → adapter couldn't learn
- **Fix:** Scale cross-attention output to 10% (like adapters)
- **Result:** Adapter now gets 96% of gradients → should learn normally
- **Action:** Retrain with the fixed model
- **Expected:** Feedback model should now beat base model

The bug was in the **design**, not the idea. Cross-attention with image features IS a good idea - it just needs proper scaling to not dominate the gradient flow!
