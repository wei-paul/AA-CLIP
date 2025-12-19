# Feedback Training V2 Guide

## Problem Analysis: Why Original Feedback Training Didn't Work

Looking at your training logs, I identified the **root cause** of why your feedback model looks similar to the base model:

### Training Loss Comparison

| Model | Image Loss Start | Image Loss End | Change |
|-------|-----------------|----------------|--------|
| **Base Model** | 3.12 | 2.26 | **-28%** (Learning!) |
| **Feedback Loop 1** | 3.16 | 3.13 | -1% (NOT learning) |
| **Feedback Loop 2** | 3.13 | 3.13 | 0% (NOT learning) |
| **Feedback Loop 3** | 3.13 | 3.13 | 0% (NOT learning) |

**The image adapter was NOT learning at all in the feedback training!** This explains why:
1. Your feedback model looks identical to the base model
2. The loss is higher (3.13 vs 2.26) - because it never improved

### Root Causes Identified

1. **Learning rate too low**: The base `image_lr=0.0005` works for original training but may be too conservative when combined with cross-attention
2. **Cross-attention interference**: The cross-attention module may be interfering with gradient flow to the image adapter
3. **Optimizer momentum reset**: Creating new optimizers each loop loses momentum history
4. **LR decay too aggressive**: `lr_decay_per_loop=0.5` means by loop 3, LR is only 12.5% of original

---

## V2 Training Script Improvements

The new `train_cross_attention_feedback_v2.py` includes these fixes:

### 1. Higher Base Learning Rate
```python
--image_lr 0.005  # 10x higher than original (was 0.0005)
--text_lr 0.00002  # 2x higher than original (was 0.00001)
```

### 2. AdamW Optimizer with Weight Decay
```python
image_optimizer = torch.optim.AdamW(
    parameters,
    lr=current_image_lr,
    betas=(0.9, 0.999),  # Standard Adam betas
    weight_decay=0.01,   # Regularization
)
```

### 3. Learning Rate Warmup
```python
--warmup_epochs 2  # Warmup prevents early instability
```

### 4. Gradient Clipping
```python
--grad_clip 1.0  # Prevents exploding gradients
```

### 5. Cross-Attention Control
```python
--cross_attn_start_loop 1  # Can disable cross-attention for first N loops
                           # Set to 2 to let adapters learn first
```

### 6. Configurable Loop Count
```python
--num_feedback_loops 5  # Now supports 1-10 loops
```

### 7. Better Epoch Scheduling
```python
--text_epochs_per_loop 5
--image_epochs_per_loop 15
--epochs_decay_rate 0.8  # More gradual decay
```

---

## Recommended Training Commands

### Option 1: Standard 5-Loop Training (Recommended)
```bash
cd AA-CLIP

python train_cross_attention_feedback_v2.py \
    --dataset VisA \
    --save_path ./ckpt/visa_feedback_v2 \
    --num_feedback_loops 5 \
    --text_epochs_per_loop 5 \
    --image_epochs_per_loop 15 \
    --text_lr 0.00002 \
    --image_lr 0.005 \
    --lr_decay_per_loop 0.8 \
    --grad_clip 1.0 \
    --load_base_text_adapter ./ckpt/VisA_base_full_1216/text_adapter.pth \
    --load_base_image_adapter ./ckpt/VisA_base_full_1216/image_adapter_20.pth
```

### Option 2: Cross-Attention Delayed Start
This lets the adapters learn first without cross-attention interference:
```bash
python train_cross_attention_feedback_v2.py \
    --dataset VisA \
    --save_path ./ckpt/visa_feedback_v2_delayed \
    --num_feedback_loops 5 \
    --cross_attn_start_loop 2 \
    --text_epochs_per_loop 5 \
    --image_epochs_per_loop 15 \
    --text_lr 0.00002 \
    --image_lr 0.005 \
    --load_base_text_adapter ./ckpt/VisA_base_full_1216/text_adapter.pth \
    --load_base_image_adapter ./ckpt/VisA_base_full_1216/image_adapter_20.pth
```

### Option 3: Train from Scratch (No Base Loading)
```bash
python train_cross_attention_feedback_v2.py \
    --dataset VisA \
    --save_path ./ckpt/visa_feedback_v2_scratch \
    --num_feedback_loops 5 \
    --text_epochs_per_loop 7 \
    --image_epochs_per_loop 20 \
    --text_lr 0.00003 \
    --image_lr 0.008 \
    --grad_clip 1.0
```

### Option 4: MVTec Training
```bash
python train_cross_attention_feedback_v2.py \
    --dataset MVTec \
    --save_path ./ckpt/mvtec_feedback_v2 \
    --num_feedback_loops 5 \
    --text_epochs_per_loop 5 \
    --image_epochs_per_loop 15 \
    --text_lr 0.00002 \
    --image_lr 0.005
```

---

## What to Monitor During Training

### 1. Image Loss Should DECREASE
Look for this pattern in logs:
```
Loop 1 - image loss: 3.15  # Start
Loop 1 - image loss: 3.05  # Should decrease!
Loop 1 - image loss: 2.95
...
Loop 5 - image loss: 2.40  # Target similar to base model
```

If loss stays flat (like 3.13 -> 3.13), something is still wrong.

### 2. Gradient Norms Should Be Non-Zero
The V2 script logs gradient norms:
```
Average Grad Norm: 0.0523  # Good - gradients are flowing
Average Grad Norm: 0.0001  # Bad - vanishing gradients
```

### 3. Text Loss Should Decrease Per Loop
```
Loop 1 - text loss: 0.95 -> 0.87
Loop 2 - text loss: 0.50 -> 0.42  # Starts lower each loop
Loop 3 - text loss: 0.42 -> 0.41
```

---

## Troubleshooting

### Loss Not Decreasing
1. **Increase learning rate**: Try `--image_lr 0.01` (20x original)
2. **Disable cross-attention**: Set `--cross_attn_start_loop 3`
3. **More epochs**: Increase `--image_epochs_per_loop 25`

### Loss Exploding (NaN or Very High)
1. **Lower learning rate**: Try `--image_lr 0.002`
2. **Stronger gradient clipping**: `--grad_clip 0.5`
3. **More warmup**: `--warmup_epochs 3`

### Training Too Slow
1. **Fewer loops**: `--num_feedback_loops 3`
2. **Fewer epochs**: `--image_epochs_per_loop 10`
3. **Larger batch**: `--image_batch_size 4` (if GPU memory allows)

---

## Expected Training Time

With default settings (5 loops, 15 image epochs per loop):

| GPU | Time per Loop | Total Time |
|-----|---------------|------------|
| RTX 3090 | ~30 min | ~2.5 hours |
| RTX 4090 | ~20 min | ~1.5 hours |
| A100 | ~15 min | ~1.25 hours |

---

## After Training: Evaluation

```bash
# Compare all feedback loops
python compare_feedback_loops.py \
    --dataset VisA \
    --save_path ./ckpt/visa_feedback_v2

# Visualize specific images
python visualize_comparison.py \
    --class_name candle \
    --image_idx 0 \
    --base_path ./ckpt/VisA_base_full_1216 \
    --cross_path ./ckpt/visa_cross_attn_from_base1217 \
    --feedback_path ./ckpt/visa_feedback_v2 \
    --feedback_loop 5
```

---

## Key Differences: V1 vs V2

| Feature | V1 | V2 |
|---------|----|----|
| Max loops | 3 (hardcoded) | 10 (configurable) |
| Image LR | 0.0005 | 0.005 (10x) |
| Text LR | 0.00001 | 0.00002 (2x) |
| Optimizer | Adam | AdamW |
| LR Schedule | MultiStepLR | Warmup + Cosine |
| Gradient Clip | None | 1.0 |
| Cross-attn control | Always on | Configurable start loop |
| Gradient monitoring | None | Logged |
| Epoch decay | Fixed per loop | Configurable rate |

---

## Summary

The original feedback training failed because the image adapter wasn't learning (loss stayed flat at ~3.13). The V2 script addresses this with:

1. **10x higher image learning rate**
2. **Warmup to prevent early instability**
3. **Gradient clipping to prevent explosions**
4. **Option to delay cross-attention** until adapters stabilize
5. **Better optimizer (AdamW)** with weight decay
6. **Gradient norm monitoring** to detect problems early

Try the recommended training command and watch for decreasing image loss!
