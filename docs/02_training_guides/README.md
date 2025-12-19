# Training Guides

Step-by-step guides for training AA-CLIP models with cross-attention and feedback loops.

## Files

### Quick Start

- **[QUICK_START_FAIR_COMPARISON.md](QUICK_START_FAIR_COMPARISON.md)**
  - Fast track to training and comparing models
  - Minimal steps to reproduce results
  - Fair comparison setup

### Training Workflows

- **[TRAINING_CROSS_ATTENTION_FROM_BASE.md](TRAINING_CROSS_ATTENTION_FROM_BASE.md)**
  - How to train cross-attention model starting from base model
  - Stage 1 and Stage 2 training details
  - Checkpoint loading and transfer

- **[FEEDBACK_TRAINING_V2_GUIDE.md](FEEDBACK_TRAINING_V2_GUIDE.md)** ⭐ **RECOMMENDED**
  - Improved feedback training with fixes
  - Supports 5+ loops (configurable)
  - Higher learning rates and better optimizers
  - Addresses gradient dilution issue

## Training Order

### 1. Base Model (Required First)
```bash
cd AA-CLIP
python train.py --dataset VisA --save_path ./ckpt/VisA_base
```
- Trains adapters without cross-attention
- Creates baseline for comparison
- Takes ~2 hours on RTX 3090

### 2. Cross-Attention Model (Optional)
```bash
python train_cross_attention.py \
    --dataset VisA \
    --save_path ./ckpt/visa_cross_attn \
    --load_base_text_adapter ./ckpt/VisA_base/text_adapter.pth \
    --load_base_image_adapter ./ckpt/VisA_base/image_adapter_20.pth
```
- Trains with cross-attention but NO feedback
- Good for ablation studies

### 3. Feedback Model (Main Goal)
```bash
python train_cross_attention_feedback_v2.py \
    --dataset VisA \
    --save_path ./ckpt/visa_feedback_v2 \
    --num_feedback_loops 5 \
    --load_base_text_adapter ./ckpt/VisA_base/text_adapter.pth \
    --load_base_image_adapter ./ckpt/VisA_base/image_adapter_20.pth
```
- **Use V2 script** (includes gradient dilution fix!)
- Trains with cross-attention AND feedback loops
- Should outperform base model

## Important Notes

### ⚠️ Critical Fix Required

**You MUST use models with the gradient dilution fix!**

The original implementation had a bug where Stage 2 cross-attention absorbed 63% of gradients, preventing the image adapter from learning.

**Fixed in:**
- `AA-CLIP/model/adapter_cross_attention.py` (lines 659-660, 666-667)
- `AA-CLIP/train_cross_attention_feedback.py` (lines 778-779)
- `AA-CLIP/train_cross_attention_feedback_v2.py` (built-in)

**Check:** Look for `image_cross_attn_weight=0.1` in model initialization

### Training Tips

**Watch These Metrics:**
- Image loss should DECREASE (not stay flat!)
- Gradient norms should be > 0.01
- Loop 3 should beat Loop 1

**If Training Fails:**
- Check gradient norms (should be 0.05-0.15)
- Try higher learning rate: `--image_lr 0.002`
- Disable cross-attention first loop: `--cross_attn_start_loop 2`

## Related Documentation

- **Bug Fix**: [Gradient Dilution Bug](../04_bug_fixes/GRADIENT_DILUTION_BUG_FIXED.md) - Read this first!
- **Testing**: [Testing Guide](../03_testing_evaluation/TESTING_GUIDE.md)
- **Implementation**: [Cross-Attention Details](../01_cross_attention_feedback_implementation/README.md)

## Expected Training Times

| Model | GPU | Time |
|-------|-----|------|
| Base (20 epochs) | RTX 3090 | ~2 hours |
| Cross-attention | RTX 3090 | ~2.5 hours |
| Feedback (3 loops) | RTX 3090 | ~3 hours |
| Feedback (5 loops) | RTX 3090 | ~4 hours |

## Quick Command Reference

```bash
# Base model
python train.py --dataset VisA --save_path ./ckpt/base

# Cross-attention (no feedback)
python train_cross_attention.py \
    --dataset VisA \
    --save_path ./ckpt/cross \
    --load_base_text_adapter ./ckpt/base/text_adapter.pth \
    --load_base_image_adapter ./ckpt/base/image_adapter_20.pth

# Feedback V2 (RECOMMENDED)
python train_cross_attention_feedback_v2.py \
    --dataset VisA \
    --save_path ./ckpt/feedback \
    --num_feedback_loops 5 \
    --load_base_text_adapter ./ckpt/base/text_adapter.pth \
    --load_base_image_adapter ./ckpt/base/image_adapter_20.pth
```
