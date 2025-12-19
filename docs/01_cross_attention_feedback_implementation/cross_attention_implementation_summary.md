# Cross-Attention Implementation Summary

## What Was Planned

This document summarizes the cross-attention enhancement planned for AA-CLIP's Stage 1 text adapter training.

---

## Overview

**Objective:** Add cross-attention mechanism to Stage 1 to create **vision-aware text embeddings**.

**Design Philosophy:** Follow the original paper's approach by using residual addition of CLS_24 to patches, maintaining consistency with the existing codebase.

---

## Key Design Decisions

### 1. Cross-Attention Placement

**Location:** After Transformer Layer 0, before text_adapter[0]

```
Text Transformer Layer 0 (frozen)
        ↓
    [77, B, 768]
        ↓
╔════════════════════════════╗
║ Cross-Attention (NEW!)     ║
║ Q = text [B, 77, 768]      ║
║ K = V = image [B, 1369, 768]║
╚════════════════════════════╝
        ↓
    [77, B, 768]
        ↓
text_adapter[0] (trainable)
```

**Rationale:**
- Early integration allows cross-attention features to propagate through all subsequent text layers
- Placed before adapters to provide vision-aware input to the adaptation mechanism
- Layer 0 provides initial text representations that are still close to token embeddings

---

### 2. Image Context Representation

**Approach:** Use original paper's residual addition (NOT concatenation)

```python
# Get layer 6 patches (spatial features)
v1_patches = layer6_output[:, 1:, :]  # [B, 1369, 1024]

# Project to 768
v1_patches = v1_patches @ visual.proj  # [B, 1369, 768]

# Get layer 24 CLS (global semantics)
cls_token_24 = layer24_cls_output  # [B, 768]
cls_token_24 = cls_token_24.unsqueeze(1)  # [B, 1, 768]

# ADD as residual (original approach)
image_context = v1_patches + cls_token_24  # [B, 1369, 768]
```

**Result:** `[B, 1369, 768]` where each patch contains both:
- Spatial features from layer 6 (edges, textures, local structures)
- Global semantic features from layer 24 CLS (object-level understanding)

**Why Not Concatenation?**
- Residual addition matches original Stage 1 design (train.py:85)
- No architectural changes needed
- Simpler implementation
- Maintains 1369 patches (no dimension increase)

---

### 3. Cross-Attention Dimensions

| Component | Shape | Description |
|-----------|-------|-------------|
| **Query (Q)** | `[B, 77, 768]` | Text features after layer 0 |
| **Key (K)** | `[B, 1369, 768]` | Image patches with CLS_24 residual |
| **Value (V)** | `[B, 1369, 768]` | Image patches with CLS_24 residual |
| **Attention Scores** | `[B, 77, 1369]` | 77 text tokens × 1369 image patches |
| **Output** | `[B, 77, 768]` | Vision-aware text features |

**Attention Formula:**
```
Attention(Q, K, V) = softmax(QK^T / sqrt(768)) @ V

QK^T: [B, 77, 768] @ [B, 768, 1369] = [B, 77, 1369]
Output: [B, 77, 1369] @ [B, 1369, 768] = [B, 77, 768]
```

---

## Implementation Architecture

### New Module: CrossAttentionModule

```python
class CrossAttentionModule(nn.Module):
    """
    Multi-head cross-attention for text-to-image attention.

    Parameters:
        - dim: 768 (embedding dimension)
        - num_heads: 8 (multi-head attention)
        - dropout: 0.1

    Trainable Parameters:
        - q_proj: Linear(768, 768)
        - k_proj: Linear(768, 768)
        - v_proj: Linear(768, 768)
        - out_proj: Linear(768, 768)
    """
```

---

## Training Modifications

### Processing Order Change

**Original Stage 1:**
1. Load batch (16 images)
2. Generate text embeddings
3. Process images (frozen ViT)
4. Calculate loss
5. Backprop (update text adapters)

**With Cross-Attention:**
1. Load batch (16 images)
2. **PRE-COMPUTE image context** (frozen ViT) ← NEW!
3. **Store image_context in model** ← NEW!
4. Generate text embeddings (with cross-attention)
5. Process images for loss (same as original)
6. Calculate loss
7. Backprop (update text adapters + cross-attention)

---

### Trainable Parameters

**Original Stage 1:**
- text_adapter[0-2]: 3 SimpleAdapter modules
- text_adapter[3]: 1 SimpleProj module
- **Total: 4 modules**

**With Cross-Attention:**
- text_adapter[0-3]: Same 4 modules
- text_cross_attn: 1 CrossAttentionModule (4 linear layers)
- **Total: 5 modules**

---

## Files to Create

**New Files (do NOT modify originals):**

1. **model/adapter_cross_attention.py**
   - `CrossAttentionModule` class
   - `AdaptedCLIPWithCrossAttention` class (extends original)
   - All original functionality preserved

2. **train_cross_attention.py**
   - Modified `train_text_adapter_with_cross_attention()` function
   - Image pre-computation before text forward pass
   - Updated optimizer to include cross-attention parameters

3. **forward_utils_cross_attention.py** (optional)
   - Copy of original with any necessary modifications

**Files NOT Modified:**
- Original `train.py`
- Original `model/adapter.py`
- Original `forward_utils.py`

This allows direct comparison between original and cross-attention versions.

---

## Expected Improvements

### What Cross-Attention Enables

1. **Vision-Aware Text Embeddings**
   - Text tokens can "see" what's in the image
   - "damaged bottle" produces different embeddings based on actual damage patterns present

2. **Spatial Grounding**
   - Text like "bottle neck" can attend to neck-region patches
   - Text like "damaged" can attend to anomaly-containing patches

3. **Better Text-Image Alignment**
   - Normal vs abnormal text embeddings become more discriminative
   - Text learns to focus on relevant visual features

4. **Improved Anomaly Localization**
   - Text embeddings understand spatial structure
   - Can better guide segmentation in downstream tasks

---

## Technical Validation

### Dimension Checks

```python
# Verify image context shape
assert image_context.shape == (16, 1369, 768)

# Verify matches original approach
original_context = v1_patches + cls_token_24
assert torch.allclose(image_context, original_context)

# Verify cross-attention output
assert cross_attn_output.shape == (16, 77, 768)
```

### Gradient Flow Verification

```python
# Check all modules receive gradients
for name, param in model.text_cross_attn.named_parameters():
    assert param.grad is not None, f"{name} has no gradient!"
    assert param.grad.norm() > 0, f"{name} gradient is zero!"
```

---

## Checkpoint Format

```python
# New checkpoint structure
{
    "epoch": epoch + 1,
    "text_adapter": model.text_adapter.state_dict(),     # Original
    "text_cross_attn": model.text_cross_attn.state_dict(),  # NEW!
    "text_optimizer": optimizer.state_dict(),
}
```

**File name:** `text_adapter_cross_attn.pth`

---

## Stage 2 Compatibility

**No changes needed for Stage 2:**

1. After Stage 1 completes, text embeddings are pre-computed:
   ```python
   with torch.no_grad():
       text_embeddings = get_adapted_text_embedding(model, dataset, device)
   ```

2. These embeddings are frozen for Stage 2 (same as original)

3. Stage 2 proceeds unchanged:
   - Uses frozen text embeddings
   - Trains image adapters only
   - No dependency on cross-attention mechanism

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Cross-attention heads | 8 | Multi-head attention |
| Dropout | 0.1 | Regularization |
| Learning rate | 1e-5 | Same as original text training |
| Batch size | 16 | Same as original Stage 1 |
| Epochs | 5 | Same as original Stage 1 |

---

## Summary of Changes

### Minimal Modifications
- ✅ No changes to frozen CLIP encoders
- ✅ No changes to original adapter modules
- ✅ No changes to Stage 2 training
- ✅ No changes to loss functions
- ✅ Follows original paper's CLS residual approach

### Additions
- ✅ 1 new module: `CrossAttentionModule`
- ✅ Pre-computation of image context
- ✅ Cross-attention after text layer 0
- ✅ Updated optimizer and checkpoint format

### Benefits
- ✅ Vision-aware text embeddings
- ✅ Better text-image alignment
- ✅ Improved anomaly detection (expected)
- ✅ Maintains architectural consistency
- ✅ Easy to compare with original (separate files)

---

## Next Steps for Implementation

1. **Create new files** (adapter_cross_attention.py, train_cross_attention.py)
2. **Implement CrossAttentionModule** with dimension assertions
3. **Modify encode_text()** to include cross-attention after layer 0
4. **Update training loop** to pre-compute image context
5. **Test dimension flow** with assertions
6. **Train and compare** with original baseline
7. **Analyze attention patterns** to verify expected behavior

---

## References

- Original code: `AA-CLIP/train.py` lines 74-85 (CLS residual addition)
- Adapter modules: `AA-CLIP/model/adapter.py`
- Text encoding: `AA-CLIP/model/adapter.py` lines 114-145
- Image encoding: `AA-CLIP/model/transformer.py` lines 490-551

---

**End of Summary**
