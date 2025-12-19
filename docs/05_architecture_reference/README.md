# Architecture Reference

Comprehensive architectural documentation and design decisions for AA-CLIP with cross-attention.

## Files

- **[ARCHITECTURE_REFERENCE.md](ARCHITECTURE_REFERENCE.md)**
  - Complete system architecture overview
  - Model components and their interactions
  - Data flow diagrams
  - Dimension tracking throughout pipeline

- **[claude.md](claude.md)**
  - Detailed walkthrough of the codebase
  - Step-by-step execution flow
  - Line-by-line code explanations
  - Educational reference for understanding implementation

## Architecture Overview

### High-Level System

```
                    AA-CLIP with Cross-Attention & Feedback
                    =====================================

┌─────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: Text Encoding                      │
│                                                                       │
│  Input: Text prompts (normal/abnormal)                              │
│         ↓                                                            │
│  CLIP Text Encoder (frozen)                                         │
│         ↓                                                            │
│  Cross-Attention (text → image)  ← Image context from Stage 2/frozen│
│         ↓                                                            │
│  Text Adapters (trainable, 0.1 weight)                             │
│         ↓                                                            │
│  Output: Text embeddings [768, 2] (normal, abnormal)               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        STAGE 2: Image Encoding                       │
│                                                                       │
│  Input: Images [B, 3, 518, 518]                                     │
│         ↓                                                            │
│  CLIP ViT Encoder (frozen)                                          │
│         ↓                                                            │
│  Cross-Attention (image → text)  ← Text context from Stage 1        │
│         ↓                                                            │
│  Image Adapters (trainable, 0.1 weight)                            │
│         ↓                                                            │
│  Output: Patch features [B, 1369, 768] × 4 scales                  │
│          Detection features [B, 768]                                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         Anomaly Detection                            │
│                                                                       │
│  Similarity Calculation: patches × text embeddings                  │
│         ↓                                                            │
│  Pixel-level anomaly maps [B, 2, 518, 518]                         │
│  Image-level anomaly scores [B, 2]                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Feedback Loop Architecture

```
Loop 1:
  Stage 1 (Text) uses FROZEN image features from clip_surgery
       ↓
  Stage 2 (Image) trains adapters
       ↓
  Save checkpoint: loop_1_final.pth

Loop 2:
  Stage 1 (Text) uses TRAINED image features from Loop 1 (frozen)
       ↓
  Stage 2 (Image) continues training adapters
       ↓
  Save checkpoint: loop_2_final.pth

Loop 3+:
  Stage 1 (Text) uses TRAINED image features from Loop 2 (frozen)
       ↓
  Stage 2 (Image) continues training adapters
       ↓
  Save checkpoint: loop_N_final.pth
```

### Critical Design Decisions

#### 1. Adapter Scaling (0.1 weight)

Both text and image adapters use **0.1 residual weight**:

```python
x = 0.1 * adapted + 0.9 * original
```

**Rationale:**
- Gradual adaptation of frozen CLIP features
- Preserves pre-trained knowledge
- Stable training

#### 2. Cross-Attention Scaling (0.1 weight) ⚠️ CRITICAL

Cross-attention outputs are also scaled to **0.1**:

```python
cross_out_scaled = 0.1 * cross_out
x = x + cross_out_scaled
```

**Rationale:**
- Prevents gradient dilution (63% absorption without scaling)
- Matches adapter design pattern
- Allows both modules to learn together

**Without this scaling:** Image adapter receives only 37% of gradients!

#### 3. Placement of Cross-Attention

**Stage 1:** After text layer 0, before text_adapter[0]
**Stage 2:** After ViT layer 0, before image_adapter[0]

**Rationale:**
- Early layers capture low-level features
- Cross-attention adds context before adaptation
- Later layers have already specialized

#### 4. Multi-Scale Features

Extract features at layers **[6, 12, 18, 24]**:

**Rationale:**
- Layer 6: Edges, textures (early features)
- Layer 12: Parts, structures (mid-level)
- Layer 18: Objects (high-level)
- Layer 24: Global semantics (final)

Different anomalies are visible at different scales.

## Key Components

### 1. CrossAttentionModule (Stage 1)
- **Input:** Text features [B, 77, 768], Image context [B, 1369, 768]
- **Output:** Cross-attended text [B, 77, 768]
- **Parameters:** 8 heads, 0.1 dropout
- **Location:** `model/adapter_cross_attention.py:29-156`

### 2. ImageCrossAttentionModule (Stage 2)
- **Input:** Image features [B, 1370, 1024], Text context [B, 1232, 768]
- **Output:** Cross-attended image [B, 1370, 1024]
- **Challenge:** Dimension mismatch (1024 vs 768)
- **Solution:** Projection layers Q:1024→768, Out:768→1024
- **Location:** `model/adapter_cross_attention.py:158-277`

### 3. SimpleAdapter
- **Architecture:** Linear(d, d) → LeakyReLU → Linear(d, d)
- **Normalization:** Norm-preserving scaling after output
- **Used by:** Both text (768d) and image (1024d) adapters

### 4. SimpleProj
- **Architecture:** Linear(d_in, d_out) → [optional ReLU]
- **Used for:** Final text projection, image seg/det projections

## Dimension Reference

### Text Pipeline
```
Tokens [B, 77]
  ↓ Token Embedding
[B, 77, 768]
  ↓ Positional Embedding
[B, 77, 768]
  ↓ Transformer Layers 0-11 (with adapters at 0-2)
[B, 77, 768]
  ↓ Extract EOS token
[B, 768]
  ↓ Final Projection Adapter
[B, 768]
  ↓ Average normal/abnormal
[768, 2]
```

### Image Pipeline
```
Image [B, 3, 518, 518]
  ↓ Patch Embedding (conv1)
[B, 1369, 1024]
  ↓ Add CLS token
[B, 1370, 1024]
  ↓ ViT Layers 0-23 (with adapters at 0-5)
[B, 1370, 1024]
  ↓ Extract patches (remove CLS)
[B, 1369, 1024]
  ↓ Seg Projections (for each scale)
[B, 1369, 768] × 4 scales
  ↓ Det Projection (last scale only)
[B, 768]
```

### Cross-Attention Dimensions

**Stage 1 (Text → Image):**
- Q: [B, 77, 768] (from text)
- K, V: [B, 1369, 768] (from image)
- Output: [B, 77, 768]

**Stage 2 (Image → Text):**
- Q: [B, 1370, 1024] (from image) → project to 768
- K, V: [B, 1232, 768] (from text, flattened 16×77)
- Output: [B, 1370, 768] → project back to 1024

## Code Structure

```
AA-CLIP/
├── model/
│   ├── adapter.py                    # Base adapter (no cross-attention)
│   ├── adapter_cross_attention.py    # Cross-attention model (MAIN)
│   ├── adapter_modules.py            # SimpleAdapter, SimpleProj
│   └── clip.py                       # CLIP model loader
├── train.py                          # Base training (no cross-attention)
├── train_cross_attention.py          # Cross-attention (no feedback)
├── train_cross_attention_feedback.py # Feedback training (MAIN)
└── train_cross_attention_feedback_v2.py # Improved feedback training
```

## Related Documentation

- **Implementation**: [Cross-Attention Implementation](../01_cross_attention_feedback_implementation/README.md)
- **Training**: [Training Guides](../02_training_guides/README.md)
- **Bug Fixes**: [Gradient Dilution Fix](../04_bug_fixes/GRADIENT_DILUTION_BUG_FIXED.md)

## Design Philosophy

1. **Minimal Changes to CLIP**: Keep CLIP frozen, only add lightweight adapters
2. **Bidirectional Context**: Text and image should inform each other
3. **Iterative Refinement**: Feedback loops compound improvements
4. **Gradient Balance**: All trainable modules must receive adequate gradients
5. **Multi-Scale Features**: Anomalies visible at different semantic levels
