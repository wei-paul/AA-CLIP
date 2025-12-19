# Testing & Evaluation

Guides for testing models, comparing performance, and visualizing results.

## Files

- **[TESTING_GUIDE.md](TESTING_GUIDE.md)**
  - How to evaluate trained models
  - Quantitative metrics (AUC, AP)
  - Comparing models (base vs cross-attention vs feedback)
  - Understanding performance benchmarks

- **[VISUALIZATION_QUICKSTART.md](VISUALIZATION_QUICKSTART.md)**
  - Visual comparison of anomaly localization
  - Side-by-side heatmap generation
  - How to interpret visualization results
  - Command reference for different classes

## Quick Start

### 1. Visualize Specific Examples

Compare all three models on a single anomalous image:

```bash
cd AA-CLIP

python visualize_comparison.py \
    --class_name candle \
    --image_idx 0 \
    --base_path ./ckpt/VisA_base_full_1216 \
    --cross_path ./ckpt/visa_cross_attn_from_base1217 \
    --feedback_path ./ckpt/visa_feedback_v2 \
    --feedback_loop 5
```

**Output:**
- Side-by-side comparison showing all 3 models
- Individual stacked views (original → ground truth → prediction)
- Saved to `AA-CLIP/comparison_results/`

### 2. Quantitative Evaluation

Test a single feedback loop:

```bash
python test_cross_attention_feedback.py \
    --dataset VisA \
    --save_path ./ckpt/visa_feedback_v2 \
    --loop 5 \
    --visualize
```

### 3. Compare All Loops

See which feedback loop performed best:

```bash
python compare_feedback_loops.py \
    --dataset VisA \
    --save_path ./ckpt/visa_feedback_v2
```

**Output:**
```
COMPARISON SUMMARY
================================================================
Loop     Pixel AUC     Pixel AP      Image AUC     Image AP
----------------------------------------------------------------
1        0.9650        0.9301        0.9823        0.9612
2        0.9712        0.9378        0.9867        0.9688
3        0.9742        0.9401        0.9881        0.9701
5        0.9780        0.9450        0.9895        0.9725
================================================================

Best Pixel AUC: Loop 5 (0.9780)
Best Image AUC: Loop 5 (0.9895)
```

## Evaluation Workflow

### Step 1: Visual Check
Start with visualizations to understand what models are learning:

```bash
# Pick a few representative examples
python visualize_comparison.py --class_name candle --image_idx 0
python visualize_comparison.py --class_name pcb3 --image_idx 5
python visualize_comparison.py --class_name capsules --image_idx 10
```

**Look for:**
- Does feedback model match ground truth better?
- Fewer false positives (red in normal areas)?
- Better localization precision?

### Step 2: Quantitative Metrics
Get objective numbers:

```bash
# Test feedback model
python test_cross_attention_feedback.py \
    --dataset VisA \
    --save_path ./ckpt/visa_feedback_v2 \
    --loop 5

# Test base model for comparison
python test_original.py \
    --dataset VisA \
    --save_path ./ckpt/VisA_base_full_1216
```

### Step 3: Compare Loops
Find the optimal loop:

```bash
python compare_feedback_loops.py \
    --dataset VisA \
    --save_path ./ckpt/visa_feedback_v2
```

## Understanding Metrics

### Pixel-Level Metrics
- **Pixel AUC**: How well the model detects anomalous pixels (0-1, higher is better)
- **Pixel AP**: Average precision for pixel classification

**What's good?**
- \>0.98 = Excellent
- 0.95-0.98 = Good
- 0.90-0.95 = Fair
- <0.90 = Poor

### Image-Level Metrics
- **Image AUC**: How well the model classifies abnormal images
- **Image AP**: Average precision for image classification

**What's good?**
- \>0.97 = Excellent
- 0.92-0.97 = Good
- 0.85-0.92 = Fair
- <0.85 = Poor

## Visualization Interpretation

### Heatmap Colors (JET colormap)
- **Dark Blue**: Very normal (low anomaly score)
- **Green**: Uncertain / boundary
- **Yellow**: Slightly anomalous
- **Red**: Highly anomalous (strongest detection)

### What to Look For

**Good Predictions:**
- Red areas overlap with ground truth
- Blue areas match normal regions
- Tight, precise localization

**Bad Predictions:**
- Red in normal areas (false positives)
- Blue where anomalies exist (missed detections)
- Spread-out red (imprecise localization)

## Available Classes

### VisA Dataset (12 classes)
```
candle, pcb1, pcb2, pcb3, pcb4
capsules, cashew, chewinggum
fryum, pipe_fryum
macaroni1, macaroni2
```

### MVTec Dataset (15 classes)
```
bottle, cable, capsule, carpet, grid
hazelnut, leather, metal_nut, pill, screw
tile, transistor, toothbrush, wood, zipper
```

## Zero-Shot Evaluation

Test on a different dataset than training:

```bash
# Trained on VisA, test on MVTec
python test_cross_attention_feedback.py \
    --dataset MVTec \
    --save_path ./ckpt/visa_feedback_v2 \
    --loop 5
```

**Expected:** Performance will be lower but should still work reasonably well.

## Troubleshooting

### Visualizations Look Identical
- Models may not have converged differently
- Check training logs - did image loss decrease?
- Verify you're loading different checkpoints

### Low Performance (<0.90 AUC)
- Model didn't train properly
- Check gradient norms in training logs
- Verify gradient dilution fix is applied

### Missing Checkpoints
```bash
# List available checkpoints
ls ./ckpt/visa_feedback_v2/*.pth

# Expected:
# loop_1_final.pth
# loop_2_final.pth
# loop_3_final.pth
# ...
```

## Related Documentation

- **Training**: [Feedback Training V2 Guide](../02_training_guides/FEEDBACK_TRAINING_V2_GUIDE.md)
- **Bug Fix**: [Gradient Dilution Fix](../04_bug_fixes/GRADIENT_DILUTION_BUG_FIXED.md)
- **Implementation**: [Cross-Attention Details](../01_cross_attention_feedback_implementation/README.md)
