# Quick Start: Fair Comparison Training

## TL;DR

```bash
# Assuming you already have trained base model checkpoints at:
# - ckpt/baseline/text_adapter.pth
# - ckpt/baseline/image_adapter.pth

cd AA-CLIP

python train_cross_attention.py \
    --dataset MVTec \
    --shot 32 \
    --text_epoch 5 \
    --image_epoch 20 \
    --load_base_text_adapter ckpt/baseline/text_adapter.pth \
    --load_base_image_adapter ckpt/baseline/image_adapter.pth \
    --save_path ckpt/cross_attn_from_base
```

That's it! Your cross-attention model will now:
1. Load pre-trained adapters from the base model
2. Initialize cross-attention modules randomly (Xavier init)
3. Train both together (adapters fine-tune, cross-attn learns from scratch)

---

## What Happens During Training

### Stage 1 Output Example
```
======================================================================
LOADING PRE-TRAINED BASE TEXT ADAPTER
======================================================================
  Path: ckpt/baseline/text_adapter.pth
  ✓ Base text adapter loaded successfully!
  Mean weight magnitude: 0.234567 (should be > 0.1 if trained)

[NOTE] Training image_adapter from scratch (random initialization)

======================================================================
WEIGHT INITIALIZATION SUMMARY
======================================================================
  text_adapter:       0.234567 (pre-trained)
  text_cross_attn:    0.012345 (random init - NEW)
  image_adapter:      0.087654 (random init)
  image_cross_attn:   0.013456 (random init - NEW)
======================================================================

STAGE 1: TEXT ADAPTER TRAINING WITH CROSS-ATTENTION
======================================================================
```

### Stage 2 Output Example
```
======================================================================
LOADING PRE-TRAINED BASE IMAGE ADAPTER
======================================================================
  Path: ckpt/baseline/image_adapter.pth
  ✓ Base image adapter loaded successfully!
  Mean weight magnitude: 0.198765 (should be > 0.1 if trained)

======================================================================
WEIGHT INITIALIZATION SUMMARY
======================================================================
  text_adapter:       0.256789 (updated from Stage 1)
  text_cross_attn:    0.089012 (trained in Stage 1)
  image_adapter:      0.198765 (pre-trained)
  image_cross_attn:   0.013456 (random init - NEW)
======================================================================

STAGE 2: IMAGE ADAPTER TRAINING WITH CROSS-ATTENTION
======================================================================
```

---

## Alternative: Train From Scratch (Not Recommended for Comparison)

```bash
python train_cross_attention.py \
    --dataset MVTec \
    --shot 32 \
    --save_path ckpt/cross_attn_scratch
```

This trains everything from random initialization. **Not fair for comparison** because:
- Cross-attn model has more parameters (adapters + cross-attention)
- Base model doesn't get the benefit of cross-attention
- Hard to tell if improvements are from architecture or just more parameters

---

## Training Schedule Recommendations

### Conservative (Same as Base)
Best for fair comparison - same epochs, same learning rates:
```bash
python train_cross_attention.py \
    --load_base_text_adapter ckpt/baseline/text_adapter.pth \
    --load_base_image_adapter ckpt/baseline/image_adapter.pth \
    --text_epoch 5 \
    --image_epoch 20 \
    --text_lr 0.00001 \
    --image_lr 0.0005
```

### Aggressive (Faster Convergence)
Since adapters already trained, can use lower LR and fewer epochs:
```bash
python train_cross_attention.py \
    --load_base_text_adapter ckpt/baseline/text_adapter.pth \
    --load_base_image_adapter ckpt/baseline/image_adapter.pth \
    --text_epoch 3 \
    --image_epoch 10 \
    --text_lr 0.000005 \
    --image_lr 0.00025
```

---

## Expected Training Behavior

### Stage 1 Loss (with pre-trained adapters)
```
Epoch 0: loss: 0.089  # Lower than base (~0.15) because adapters already good
Epoch 1: loss: 0.067
Epoch 2: loss: 0.054
Epoch 3: loss: 0.048
Epoch 4: loss: 0.045  # Should match or beat base
```

### Stage 2 Loss (with pre-trained adapters)
```
Epoch 0: loss: 0.345  # Lower than base (~0.5) because adapters already good
...
Epoch 19: loss: 0.198  # Should beat base
```

If your losses start HIGHER than base, something went wrong with loading!

---

## Troubleshooting

### "Mean weight magnitude: 0.012345"
**Problem**: Weight magnitude too small (< 0.05) after loading.
**Cause**: Checkpoint didn't load or wrong path.
**Fix**: Verify checkpoint path exists and contains trained weights.

### "size mismatch for text_adapter.0.fc1.weight"
**Problem**: Base model has different architecture.
**Cause**: Base model trained with different `text_adapt_until` or adapter dimensions.
**Fix**: Train new base model with same hyperparameters, or adjust your model config.

### Loss starts high (> 1.0) in Stage 1
**Problem**: Adapters not actually loaded.
**Cause**: Path error or checkpoint loading failed silently.
**Fix**: Check console output for "✓ Base adapter loaded successfully!" message.

---

## Files Created

After training completes:

```
ckpt/cross_attn_from_base/
├── text_adapter_cross_attn.pth        # Stage 1 checkpoint
├── image_adapter_cross_attn.pth       # Stage 2 checkpoint
├── image_adapter_cross_attn_1.pth     # Intermediate checkpoints
├── image_adapter_cross_attn_2.pth
├── ...
└── train_cross_attention.log          # Training log
```

These checkpoints contain BOTH base adapters AND cross-attention modules.

---

## Next Steps

After training:
1. Evaluate on test set (see EVALUATION.md)
2. Compare metrics with base model
3. Visualize attention maps
4. Ablation studies (with/without cross-attention)

---

**Last updated**: Now
