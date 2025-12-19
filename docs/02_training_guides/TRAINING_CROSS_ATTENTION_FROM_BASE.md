# Training Cross-Attention Model from Pre-trained Base Model

## Overview

To fairly compare your cross-attention model with the base model, you need to:
1. Train the base model first (Stage 1 + Stage 2)
2. Load those weights into your cross-attention model
3. Train ONLY the new cross-attention modules
4. Compare performance

---

## Step 1: Train the Base Model (if not already done)

```bash
cd AA-CLIP

# Train base model (both stages)
python train.py \
    --dataset MVTec \
    --training_mode few_shot \
    --shot 32 \
    --text_epoch 5 \
    --image_epoch 20 \
    --text_batch_size 16 \
    --image_batch_size 2 \
    --save_path ckpt/baseline_mvtec
```

**This creates:**
- `ckpt/baseline_mvtec/text_adapter.pth` (Stage 1 checkpoint)
- `ckpt/baseline_mvtec/image_adapter.pth` (Stage 2 checkpoint)

---

## Step 2: Initialize Cross-Attention Model with Base Weights

### Option A: Using Command-Line Arguments (Recommended)

Add these arguments to `train_cross_attention.py`:

```bash
python train_cross_attention.py \
    --dataset MVTec \
    --training_mode few_shot \
    --shot 32 \
    --text_epoch 5 \
    --image_epoch 20 \
    --save_path ckpt/cross_attn_mvtec \
    --load_base_text_adapter ckpt/baseline_mvtec/text_adapter.pth \
    --load_base_image_adapter ckpt/baseline_mvtec/image_adapter.pth
```

### Option B: Manual Loading in Code

If you want to load manually, use this pattern:

```python
# In train_cross_attention.py main() function

# Create model
model = AdaptedCLIPWithCrossAttention(...)

# Load base model weights BEFORE training
base_text_ckpt = torch.load("ckpt/baseline_mvtec/text_adapter.pth")
model.text_adapter.load_state_dict(base_text_ckpt["text_adapter"])

base_image_ckpt = torch.load("ckpt/baseline_mvtec/image_adapter.pth")
model.image_adapter.load_state_dict(base_image_ckpt["image_adapter"])

# Now text_adapter and image_adapter have pre-trained weights
# Only text_cross_attn and image_cross_attn are randomly initialized
```

---

## Step 3: What Gets Trained?

### Stage 1 (Cross-Attention Training)
| Component | Status | Why |
|-----------|--------|-----|
| `text_adapter[0-3]` | **Fine-tuned** | Already trained by base, now adapts with cross-attention |
| `text_cross_attn` | **Trained from scratch** | NEW component |
| CLIP text encoder | **Frozen** | Never trained |

### Stage 2 (Cross-Attention Training)
| Component | Status | Why |
|-----------|--------|-----|
| `image_adapter["layer_adapters"]` | **Fine-tuned** | Already trained by base |
| `image_adapter["seg_proj"]` | **Fine-tuned** | Already trained by base |
| `image_adapter["det_proj"]` | **Fine-tuned** | Already trained by base |
| `image_cross_attn` | **Trained from scratch** | NEW component |
| CLIP image encoder | **Frozen** | Never trained |

---

## Step 4: Implementation - Add Loading Logic

### Modify `train_cross_attention.py`

Add command-line arguments:

```python
# In main() function, add these arguments:
parser.add_argument(
    "--load_base_text_adapter",
    type=str,
    default=None,
    help="Path to pre-trained base text_adapter.pth"
)
parser.add_argument(
    "--load_base_image_adapter",
    type=str,
    default=None,
    help="Path to pre-trained base image_adapter.pth"
)
```

Add loading logic AFTER model creation:

```python
# After: model = AdaptedCLIPWithCrossAttention(...).to(device)

# Load pre-trained base model weights if provided
if args.load_base_text_adapter is not None:
    logger.info(f"Loading base text adapter from: {args.load_base_text_adapter}")
    base_ckpt = torch.load(args.load_base_text_adapter, map_location=device)
    model.text_adapter.load_state_dict(base_ckpt["text_adapter"])
    logger.info("✓ Base text adapter loaded successfully!")

if args.load_base_image_adapter is not None:
    logger.info(f"Loading base image adapter from: {args.load_base_image_adapter}")
    base_ckpt = torch.load(args.load_base_image_adapter, map_location=device)
    model.image_adapter.load_state_dict(base_ckpt["image_adapter"])
    logger.info("✓ Base image adapter loaded successfully!")
```

---

## Step 5: Verify Loading

Add verification code to check what was loaded:

```python
def verify_weight_loading(model, logger):
    """Verify which components have been initialized"""
    logger.info("\n" + "="*60)
    logger.info("WEIGHT INITIALIZATION VERIFICATION")
    logger.info("="*60)

    # Check text adapter
    text_mean = sum(p.abs().mean().item() for p in model.text_adapter.parameters()) / len(list(model.text_adapter.parameters()))
    logger.info(f"text_adapter mean weight magnitude: {text_mean:.6f}")

    # Check image adapter
    image_mean = sum(p.abs().mean().item() for p in model.image_adapter.parameters()) / len(list(model.image_adapter.parameters()))
    logger.info(f"image_adapter mean weight magnitude: {image_mean:.6f}")

    # Check cross-attention (should be small - Xavier init)
    text_cross_mean = sum(p.abs().mean().item() for p in model.text_cross_attn.parameters()) / len(list(model.text_cross_attn.parameters()))
    logger.info(f"text_cross_attn mean weight magnitude: {text_cross_mean:.6f} (should be small)")

    image_cross_mean = sum(p.abs().mean().item() for p in model.image_cross_attn.parameters()) / len(list(model.image_cross_attn.parameters()))
    logger.info(f"image_cross_attn mean weight magnitude: {image_cross_mean:.6f} (should be small)")

    logger.info("="*60 + "\n")

# Call after loading weights
verify_weight_loading(model, logger)
```

Expected output:
```
============================================================
WEIGHT INITIALIZATION VERIFICATION
============================================================
text_adapter mean weight magnitude: 0.234567 (pre-trained, should be > 0.1)
image_adapter mean weight magnitude: 0.198765 (pre-trained, should be > 0.1)
text_cross_attn mean weight magnitude: 0.012345 (Xavier init, should be small)
image_cross_attn mean weight magnitude: 0.013456 (Xavier init, should be small)
============================================================
```

---

## Step 6: Training Schedule Recommendations

### Conservative Approach (Recommended for Fair Comparison)
```bash
python train_cross_attention.py \
    --dataset MVTec \
    --load_base_text_adapter ckpt/baseline_mvtec/text_adapter.pth \
    --load_base_image_adapter ckpt/baseline_mvtec/image_adapter.pth \
    --text_epoch 5 \
    --image_epoch 20 \
    --text_lr 0.00001 \    # Same as base
    --image_lr 0.0005 \    # Same as base
    --save_path ckpt/cross_attn_from_base
```

### Aggressive Approach (Faster Convergence)
```bash
python train_cross_attention.py \
    --dataset MVTec \
    --load_base_text_adapter ckpt/baseline_mvtec/text_adapter.pth \
    --load_base_image_adapter ckpt/baseline_mvtec/image_adapter.pth \
    --text_epoch 3 \       # Fewer epochs since adapters already trained
    --image_epoch 10 \     # Fewer epochs
    --text_lr 0.000005 \   # Lower LR for fine-tuning
    --image_lr 0.00025 \   # Lower LR for fine-tuning
    --save_path ckpt/cross_attn_from_base_fast
```

---

## Step 7: What NOT to Do (Common Mistakes)

### ❌ DON'T: Train from scratch then compare
```python
# Wrong - unfair comparison
model = AdaptedCLIPWithCrossAttention(...)  # Random init
# Train for same epochs as base
# Compare performance  # Cross-attn has advantage of more parameters!
```

### ❌ DON'T: Load checkpoint but reset optimizer
```python
# Wrong - loses training momentum
model.text_adapter.load_state_dict(...)
optimizer = torch.optim.Adam(...)  # Fresh optimizer state
# This can hurt performance in early epochs
```

### ✅ DO: Load weights, use same hyperparameters
```python
# Correct - fair comparison
model.text_adapter.load_state_dict(base_ckpt["text_adapter"])
model.image_adapter.load_state_dict(base_ckpt["image_adapter"])
# Use SAME lr, epochs, batch_size as base model
# Only cross-attention is new
```

---

## Step 8: Expected Behavior

### During Stage 1 Training
```
Epoch 0: loss: 0.123  # Should start lower than base (already trained adapters)
Epoch 1: loss: 0.089  # Cross-attention learns quickly
Epoch 2: loss: 0.067
Epoch 3: loss: 0.054
Epoch 4: loss: 0.049  # Should converge to similar or better than base
```

### During Stage 2 Training
```
Epoch 0: loss: 0.456  # Should start lower than base
...
Epoch 19: loss: 0.234  # Should match or beat base model
```

---

## Step 9: Fair Comparison Checklist

Before claiming cross-attention is better:

- [ ] Base model fully trained (5 text epochs + 20 image epochs)
- [ ] Cross-attention model initialized with base weights
- [ ] Same hyperparameters (lr, batch_size, epochs)
- [ ] Same dataset split (same random seed for train/test)
- [ ] Same evaluation metrics
- [ ] Tested on same test set
- [ ] Statistical significance testing (multiple seeds)

---

## Step 10: Alternative - Two-Stage Comparison

### Stage 1: Train Cross-Attention on Top of Base
```bash
# Train base model first
python train.py --save_path ckpt/base

# Add cross-attention on top
python train_cross_attention.py \
    --load_base_text_adapter ckpt/base/text_adapter.pth \
    --load_base_image_adapter ckpt/base/image_adapter.pth \
    --save_path ckpt/base_plus_cross_attn
```

### Stage 2: Compare Three Models
1. **Baseline**: Base model only
2. **Cross-Attn (from scratch)**: Your current training
3. **Cross-Attn (from base)**: Initialized with base weights ← **This is the fairest**

---

## Quick Start Commands

```bash
# Step 1: Train base model (if not done)
python train.py \
    --dataset MVTec \
    --shot 32 \
    --save_path ckpt/baseline_mvtec

# Step 2: Train cross-attention from base
python train_cross_attention.py \
    --dataset MVTec \
    --shot 32 \
    --load_base_text_adapter ckpt/baseline_mvtec/text_adapter.pth \
    --load_base_image_adapter ckpt/baseline_mvtec/image_adapter.pth \
    --save_path ckpt/cross_attn_mvtec

# Step 3: Evaluate both
python inference.py --checkpoint ckpt/baseline_mvtec/image_adapter.pth
python inference.py --checkpoint ckpt/cross_attn_mvtec/image_adapter.pth
```

---

## Troubleshooting

### Error: "size mismatch"
```
RuntimeError: Error(s) in loading state_dict for AdaptedCLIPWithCrossAttention:
    size mismatch for text_adapter.0.fc1.weight: copying a param with shape torch.Size([384, 768])
```

**Cause**: Base checkpoint uses different adapter architecture.

**Solution**: Verify your `AdaptedCLIPWithCrossAttention` uses same adapter dimensions as base model.

### Error: "unexpected key"
```
Unexpected key(s) in state_dict: "text_cross_attn.q_proj.weight"
```

**Cause**: Trying to load cross-attention checkpoint as base model.

**Solution**: Use `strict=False` when loading:
```python
model.text_adapter.load_state_dict(base_ckpt["text_adapter"], strict=False)
```

---

**END OF GUIDE**
