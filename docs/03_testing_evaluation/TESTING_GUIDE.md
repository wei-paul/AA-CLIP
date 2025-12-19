# Testing Guide for Cross-Attention Feedback AA-CLIP

## Overview

After training with the feedback loop, you need to evaluate your model to see if it works better than the original. This guide explains how to test and visualize your results.

---

## Understanding the Evaluation

### What the Author Uses

The original AA-CLIP paper evaluates models using:

1. **Quantitative Metrics** (most important for comparison):
   - **Pixel-level AUC**: How well the model detects anomalous pixels (0-1, higher is better)
   - **Pixel-level AP**: Average precision for pixel classification
   - **Image-level AUC**: How well the model classifies abnormal images
   - **Image-level AP**: Average precision for image classification

2. **Visual Heatmaps** (helps understand what the model learned):
   - Stacked visualization showing:
     - Original image (top)
     - Ground truth mask (middle)
     - Predicted anomaly map (bottom)

### Why Training Loss Doesn't Tell You Everything

You're right that loss values during training don't tell you if the model works better! That's because:
- Loss measures how well the model fits the **training data**
- Performance metrics measure how well it **generalizes to test data**
- A model can have low loss but poor test performance (overfitting)
- Or vice versa - high loss but good test performance

**You MUST test on held-out test data to know if your model is good!**

---

## Quick Start: Test Your Feedback Model

### 1. Test the Final Loop (Loop 3)

```bash
cd AA-CLIP

python test_cross_attention_feedback.py \
    --dataset VisA \
    --save_path ./ckpt/visa_cross_feedback \
    --loop 3 \
    --visualize
```

**What this does:**
- Loads your Loop 3 checkpoint (`loop_3_final.pth`)
- Tests on all VisA classes
- Prints metrics for each class + average
- Saves visualizations to `./ckpt/visa_cross_feedback/visualization/VisA/`
- Saves detailed results to `./ckpt/visa_cross_feedback/test.log`

**Output example:**
```
Testing class: candle
  Pixel AUC: 0.9823
  Pixel AP:  0.9456
  Image AUC: 0.9912
  Image AP:  0.9788

Testing class: pcb3
  Pixel AUC: 0.9654
  ...

================================================================================
FINAL RESULTS
================================================================================
class name    pixel AUC    pixel AP    image AUC    image AP
--------------------------------------------------------------
candle          0.9823      0.9456       0.9912      0.9788
pcb3            0.9654      0.9321       0.9856      0.9623
...
Average         0.9742      0.9401       0.9881      0.9701
================================================================================
```

### 2. Compare All Feedback Loops

To see if feedback training actually improved performance:

```bash
python compare_feedback_loops.py \
    --dataset VisA \
    --save_path ./ckpt/visa_cross_feedback
```

**What this does:**
- Tests Loop 1, Loop 2, and Loop 3 checkpoints
- Compares their performance side-by-side
- Shows which loop performed best
- Calculates improvement from Loop 1 → Loop 3

**Output example:**
```
================================================================================
COMPARISON SUMMARY
================================================================================
Loop     Pixel AUC     Pixel AP      Image AUC     Image AP
--------------------------------------------------------------------------------
1        0.9650        0.9301        0.9823        0.9612
2        0.9712        0.9378        0.9867        0.9688
3        0.9742        0.9401        0.9881        0.9701
================================================================================

Best Pixel AUC: Loop 3 (0.9742)
Best Image AUC: Loop 3 (0.9881)

Improvement from Loop 1 to Loop 3:
  Pixel AUC: +0.92%
  Image AUC: +0.58%
```

---

## Comparing to Baselines

### Compare to Original AA-CLIP (No Cross-Attention)

1. **Test your base model** (trained with `train.py`):
```bash
python test_original.py \
    --dataset VisA \
    --save_path ./ckpt/VisA_base_full_1216
```

2. **Test your feedback model**:
```bash
python test_cross_attention_feedback.py \
    --dataset VisA \
    --save_path ./ckpt/visa_cross_feedback \
    --loop 3
```

3. **Compare the "Average" rows** from both test logs

### Compare to Cross-Attention (No Feedback)

If you trained cross-attention without feedback:

1. **Test cross-attention model**:
```bash
python test_cross_attention_feedback.py \
    --dataset VisA \
    --save_path ./ckpt/visa_cross_attn_from_base1217 \
    --loop 1  # Only loop 1 if no feedback
```

2. **Compare to feedback model** (loop 3)

---

## Visualizing Results

### Enable Visualization

Add `--visualize` flag to generate heatmaps:

```bash
python test_cross_attention_feedback.py \
    --dataset VisA \
    --save_path ./ckpt/visa_cross_feedback \
    --loop 3 \
    --visualize
```

### Where Visualizations are Saved

```
./ckpt/visa_cross_feedback/visualization/VisA/
├── candle/
│   ├── bad_001.JPG
│   ├── bad_002.JPG
│   └── ...
├── pcb3/
│   ├── defect_001.JPG
│   └── ...
└── ...
```

### What the Visualizations Show

Each image is a vertical stack:

```
┌─────────────────┐
│ Original Image  │ ← The test image
├─────────────────┤
│ Ground Truth    │ ← Actual anomaly mask (from dataset)
│  (red overlay)  │
├─────────────────┤
│ Prediction Map  │ ← Model's anomaly heatmap
│  (red overlay)  │    (bright red = high anomaly score)
└─────────────────┘
```

**Good predictions**: Predicted map matches ground truth
**Bad predictions**: Predicted map differs from ground truth

---

## Understanding Your Results

### What is "Good" Performance?

For anomaly detection on MVTec/VisA:

| Metric | Poor | Fair | Good | Excellent |
|--------|------|------|------|-----------|
| Pixel AUC | <0.90 | 0.90-0.95 | 0.95-0.98 | >0.98 |
| Image AUC | <0.85 | 0.85-0.92 | 0.92-0.97 | >0.97 |

### Interpreting Your Training Log

Looking at your `train_feedback.log`:

```
Loop 1 - text loss: 0.962 → 0.871 (decreasing ✓)
Loop 1 - image loss: 3.157 → 3.132 (slightly decreasing)

Loop 2 - text loss: 0.503 → 0.422 (decreasing ✓)
Loop 2 - image loss: 3.131 → 3.131 (stable)

Loop 3 - text loss: 0.418 → 0.415 (stable)
Loop 3 - image loss: 3.125 → 3.130 (stable)
```

**Observations:**
- Text loss decreases across loops → model is learning
- Image loss stays around 3.13 → might be converged or need tuning
- **BUT** you can't know if this is good until you test!

### Questions to Answer After Testing

1. **Did feedback improve performance?**
   - Compare Loop 1 vs Loop 3 metrics
   - Look for improvement in Pixel AUC and Image AUC

2. **Which loop performed best?**
   - Use `compare_feedback_loops.py` to find optimal loop
   - Loop 3 isn't always best (might overfit)

3. **Did cross-attention help?**
   - Compare to base AA-CLIP (no cross-attention)
   - Look at specific classes that improved/degraded

4. **What did the model learn?**
   - Look at visualizations
   - Check if predictions match ground truth

---

## Common Issues

### Issue: "Checkpoint not found"

**Problem**: Script can't find `loop_3_final.pth`

**Solution**: Check available checkpoints:
```bash
ls ./ckpt/visa_cross_feedback/*.pth
```

Expected files:
- `loop_1_final.pth`
- `loop_2_final.pth`
- `loop_3_final.pth`

If missing, training didn't complete. Check training logs.

### Issue: Low Performance

**Problem**: Metrics are much worse than expected

**Possible causes:**
1. **Model didn't load correctly**: Check checkpoint loading messages
2. **Wrong dataset**: Make sure `--dataset` matches training
3. **Hyperparameters mismatch**: Ensure test script uses same config as training
4. **Training didn't converge**: Try training longer or adjusting learning rate

### Issue: Visualizations Don't Match

**Problem**: Predicted maps don't look like ground truth

**Diagnosis:**
- If ALL predictions are blank → model might not be loaded correctly
- If predictions are noisy → might need more training
- If predictions are inverted → check normalization

---

## Full Testing Workflow

Here's the complete workflow to evaluate your feedback model:

```bash
# 1. Test final feedback model
python test_cross_attention_feedback.py \
    --dataset VisA \
    --save_path ./ckpt/visa_cross_feedback \
    --loop 3 \
    --visualize

# 2. Compare all feedback loops
python compare_feedback_loops.py \
    --dataset VisA \
    --save_path ./ckpt/visa_cross_feedback

# 3. Test base model for comparison
python test_original.py \
    --dataset VisA \
    --save_path ./ckpt/VisA_base_full_1216

# 4. Compare cross-attention (no feedback)
python test_cross_attention_feedback.py \
    --dataset VisA \
    --save_path ./ckpt/visa_cross_attn_from_base1217 \
    --loop 1
```

Then review:
1. `./ckpt/visa_cross_feedback/test.log` - Your feedback results
2. `./ckpt/visa_cross_feedback/comparison.csv` - Loop comparison
3. `./ckpt/VisA_base_full_1216/test.log` - Base model results
4. `./ckpt/visa_cross_feedback/visualization/` - Visual heatmaps

---

## Next Steps

After testing, you should be able to answer:

✓ **Does feedback training improve performance?**
  - Compare Loop 1 vs Loop 3 metrics

✓ **How much improvement did you get?**
  - Calculate percentage improvement in AUC/AP

✓ **Is the improvement significant?**
  - Look at per-class results
  - Check if improvement is consistent across classes

✓ **What did the model learn?**
  - Examine visualizations
  - Understand failure cases

Then you can:
- Write up your findings
- Tune hyperparameters if needed
- Compare to other methods
- Publish your results!

---

## Quick Reference Commands

```bash
# Test single loop
python test_cross_attention_feedback.py --dataset VisA --save_path ./ckpt/visa_cross_feedback --loop 3 --visualize

# Compare all loops
python compare_feedback_loops.py --dataset VisA --save_path ./ckpt/visa_cross_feedback

# Test base model
python test_original.py --dataset VisA --save_path ./ckpt/VisA_base_full_1216

# View results
cat ./ckpt/visa_cross_feedback/test.log
cat ./ckpt/visa_cross_feedback/comparison.csv
```

Good luck with testing! 🚀
