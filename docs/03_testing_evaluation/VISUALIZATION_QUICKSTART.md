# Visualization Quick Start Guide

## ✅ SUCCESS! Your visualizations are working!

The script has successfully generated visualizations comparing your three models.

## What Was Created

Check the `AA-CLIP/comparison_results/` directory:

```
AA-CLIP/comparison_results/
├── candle_idx0_comparison.png          ← Side-by-side comparison (all models)
├── candle_idx0_Base_Model.png          ← Base model stacked view
├── candle_idx0_Cross-Attention.png     ← Cross-attention stacked view
└── candle_idx0_Feedback_Loop_3.png     ← Feedback loop 3 stacked view
```

### Main Comparison Image

Open `candle_idx0_comparison.png` to see:

```
┌──────────┬─────────────┬────────────┬─────────────────┬──────────────────┐
│ Original │ Ground Truth│ Base Model │ Cross-Attention │ Feedback (Loop 3)│
│  Image   │   (Where    │ Prediction │   Prediction    │    Prediction    │
│          │  anomaly    │            │                 │                  │
│          │   actually  │            │                 │                  │
│          │     is)     │            │                 │                  │
└──────────┴─────────────┴────────────┴─────────────────┴──────────────────┘
```

**Red areas = Detected anomaly**
**Blue areas = Normal**

### Individual Stacked Views

Each `candle_idx0_*.png` file shows:
```
┌──────────────┐
│   Original   │  ← The test image
├──────────────┤
│ Ground Truth │  ← Where the anomaly actually is
├──────────────┤
│  Prediction  │  ← Where this model thinks the anomaly is
└──────────────┘
```

## How to Compare Models

Look at the comparison image and ask:

1. **Which model's prediction matches the ground truth best?**
   - Red areas should overlap with ground truth
   - Blue areas should match normal regions

2. **Which model has fewer false positives?**
   - Red in normal areas = false alarm

3. **Which model has better localization precision?**
   - Tight red spot = precise localization
   - Spread-out red = imprecise

4. **Which model misses fewer anomalies?**
   - Blue where there should be red = missed detection

---

## Try More Examples

### Different Images for Same Class

```bash
cd AA-CLIP

# Try candle image 5
python visualize_comparison.py --class_name candle --image_idx 5

# Try candle image 10
python visualize_comparison.py --class_name candle --image_idx 10

# Try candle image 25
python visualize_comparison.py --class_name candle --image_idx 25
```

### Different Classes

```bash
# List PCB anomalies
python visualize_comparison.py --class_name pcb3 --list_images

# Visualize PCB anomaly
python visualize_comparison.py --class_name pcb3 --image_idx 0

# Try capsules
python visualize_comparison.py --class_name capsules --image_idx 0

# Try cashew
python visualize_comparison.py --class_name cashew --image_idx 0

# Try fryum
python visualize_comparison.py --class_name fryum --image_idx 0
```

### Compare Different Feedback Loops

```bash
# Compare Loop 1 vs Loop 3
python visualize_comparison.py \
    --class_name candle \
    --image_idx 0 \
    --feedback_loop 1  # Try loop 1

python visualize_comparison.py \
    --class_name candle \
    --image_idx 0 \
    --feedback_loop 2  # Try loop 2
```

---

## Understanding the Heatmap Colors

**JET Colormap** (blue → green → yellow → red):
- **Dark Blue**: Very normal (low anomaly score)
- **Light Blue/Cyan**: Slightly normal
- **Green**: Uncertain / boundary
- **Yellow**: Slightly anomalous
- **Orange**: Moderately anomalous
- **Red**: Highly anomalous (strongest detection)

---

## Quick Evaluation Workflow

1. **Pick a few representative examples** from each class:
   ```bash
   # For each class, try images 0, 5, 10
   python visualize_comparison.py --class_name candle --image_idx 0
   python visualize_comparison.py --class_name candle --image_idx 5
   python visualize_comparison.py --class_name candle --image_idx 10

   python visualize_comparison.py --class_name pcb3 --image_idx 0
   python visualize_comparison.py --class_name pcb3 --image_idx 5
   # ... etc
   ```

2. **Look for patterns:**
   - Does one model consistently match ground truth better?
   - Does one model have more false positives?
   - Are there specific types of anomalies where one model excels?

3. **Document your findings:**
   - Take screenshots of good/bad examples
   - Note which model performs best on which class
   - Identify failure cases

---

## Available Classes

VisA dataset has 12 classes:

```
candle        - Candle defects
pcb1          - Dual ultrasonic distance sensor PCB
pcb2          - Integrated circuits board
pcb3          - Infrared sensor PCB module
pcb4          - Battery charging PCB module
capsules      - Capsule defects
cashew        - Cashew nut defects
chewinggum    - Chewing gum defects
fryum         - Wheel-shaped fryum snack defects
pipe_fryum    - Pipe-shaped fryum defects
macaroni1     - Orange macaroni defects
macaroni2     - Scattered yellow macaroni defects
```

---

## Command Reference

```bash
# List available images for a class
python visualize_comparison.py --class_name <CLASS> --list_images

# Visualize specific image
python visualize_comparison.py \
    --class_name <CLASS> \
    --image_idx <N> \
    --base_path ./ckpt/VisA_base_full_1216 \
    --cross_path ./ckpt/visa_cross_attn_from_base1217 \
    --feedback_path ./ckpt/visa_cross_feedback \
    --feedback_loop 3

# Visualize with different feedback loop
python visualize_comparison.py \
    --class_name <CLASS> \
    --image_idx <N> \
    --feedback_loop 1  # Or 2, 3

# Change output directory
python visualize_comparison.py \
    --class_name <CLASS> \
    --image_idx <N> \
    --output_dir ./my_results
```

---

## Next Steps

Now that you can visualize individual examples:

1. ✅ **Visual comparison complete** - You can see how models localize anomalies
2. ⏭️ **Quantitative evaluation** - Run full test to get AUC/AP metrics (use `test_cross_attention_feedback.py`)
3. ⏭️ **Compare loops** - Test all 3 feedback loops to see which is best (use `compare_feedback_loops.py`)

**Your workflow:**
1. Visual check (this script) - Understand what models are doing
2. Quantitative metrics (test scripts) - Measure performance objectively
3. Iterate - Improve training based on findings

---

## Troubleshooting

### No models loaded

Check that checkpoint paths exist:
```bash
ls ./ckpt/VisA_base_full_1216/*.pth
ls ./ckpt/visa_cross_attn_from_base1217/*.pth
ls ./ckpt/visa_cross_feedback/*.pth
```

### Class not found

List available classes:
```python
python -c "from dataset.constants import CLASS_NAMES; print(CLASS_NAMES['VisA'])"
```

### Image index out of range

List images first:
```bash
python visualize_comparison.py --class_name <CLASS> --list_images
```

---

**You now have visual evidence of how your models are performing!** 🎉

Compare the images side-by-side to see if feedback training actually improves anomaly localization.
