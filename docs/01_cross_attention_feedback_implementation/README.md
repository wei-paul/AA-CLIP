# Cross-Attention & Feedback Loop Implementation

This directory contains documentation for the core cross-attention mechanism and feedback loop implementation.

## Files

### Core Implementation

- **[cross_attention_implementation_stage1.md](cross_attention_implementation_stage1.md)**
  - Stage 1: Text-to-Image cross-attention
  - How text tokens attend to image patches
  - Implementation details and code references

- **[cross_attention_implementation_summary.md](cross_attention_implementation_summary.md)**
  - Overview of both Stage 1 and Stage 2 cross-attention
  - Architecture diagrams
  - Data flow and dimensions

- **[feedback_implementation.md](feedback_implementation.md)**
  - Feedback loop architecture (Loops 1, 2, 3)
  - How trained image features improve text embeddings
  - Training strategy across loops

## Key Concepts

### Cross-Attention Mechanism

**Stage 1 (Text → Image):**
```
Text tokens [B, 77, 768]
     ↓ Query
Image patches [B, 1369, 768]
     ↓ Key/Value
Cross-attended text [B, 77, 768]
```

**Stage 2 (Image → Text):**
```
Image patches [B, 1370, 1024]
     ↓ Query
Text features [B, 1232, 768]
     ↓ Key/Value
Cross-attended image [B, 1370, 1024]
```

### Feedback Loop

```
Loop 1: Text adapter → Image adapter (frozen image features)
        ↓
Loop 2: Text adapter uses TRAINED image features
        ↓
Loop 3: Further refinement with better features
```

## Related Documentation

- **Bug Fix**: [Gradient Dilution Fix](../04_bug_fixes/GRADIENT_DILUTION_BUG_FIXED.md) - CRITICAL for Stage 2
- **Training**: [Feedback Training V2 Guide](../02_training_guides/FEEDBACK_TRAINING_V2_GUIDE.md)
- **Architecture**: [Complete Architecture Reference](../05_architecture_reference/ARCHITECTURE_REFERENCE.md)

## Quick Reference

### Important Code Locations

- **Cross-Attention Modules**: `AA-CLIP/model/adapter_cross_attention.py`
  - Lines 29-156: `CrossAttentionModule` (Stage 1)
  - Lines 158-277: `ImageCrossAttentionModule` (Stage 2)
  - Lines 520-540: Stage 1 forward pass (with scaling fix)
  - Lines 646-670: Stage 2 forward pass (with scaling fix)

- **Training Script**: `AA-CLIP/train_cross_attention_feedback.py`
  - Lines 312-457: Stage 1 training function
  - Lines 507-642: Stage 2 training function
  - Lines 842-980: Feedback loop orchestration

### Scaling Weights (IMPORTANT)

Both cross-attention modules use **0.1 scaling** to prevent gradient dilution:
```python
text_cross_attn_weight = 0.1   # Stage 1
image_cross_attn_weight = 0.1  # Stage 2
```

Without this scaling, Stage 2 adapter receives only 37% of gradients!
