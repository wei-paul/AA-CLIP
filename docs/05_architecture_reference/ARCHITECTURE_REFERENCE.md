# AA-CLIP Architecture Reference
## Quick Context for Future Sessions

**Purpose:** This document provides a complete technical reference of the AA-CLIP (Anomaly-Aware CLIP) architecture for zero-shot anomaly detection. Use this to quickly understand the system before discussing improvements or modifications.

---

## Table of Contents
1. [High-Level Overview](#high-level-overview)
2. [Model Architecture](#model-architecture)
3. [Two-Stage Training Process](#two-stage-training-process)
4. [Key Design Decisions](#key-design-decisions)
5. [Loss Functions](#loss-functions)
6. [Important Implementation Details](#important-implementation-details)
7. [Data Flow Summary](#data-flow-summary)
8. [Potential Improvement Areas](#potential-improvement-areas)

---

## High-Level Overview

**Problem:** Zero-shot anomaly detection using vision-language models
**Base Model:** Pretrained CLIP (ViT-L/14-336)
**Approach:** Lightweight adapter-based fine-tuning in two stages
**Key Insight:** Keep CLIP frozen, only train small adapter modules to preserve zero-shot capability

### Core Components
```
Pretrained CLIP ViT-L/14-336 (FROZEN)
├── Text Encoder: 12 transformer layers, 768-dim embeddings
├── Image Encoder: 24 transformer layers, 1024-dim hidden, 768-dim output
└── Adapters: Lightweight trainable modules (parameter-efficient)
```

---

## Model Architecture

### 1. Text Encoder with Adapters

**Base:** CLIP Text Transformer (12 layers, 768-dim)
**Trainable Components:** 4 modules (Stage 1 only)

```python
# Location: adapter.py:41-44
self.text_adapter = nn.ModuleList([
    SimpleAdapter(768, 768),        # Adapter 0 (after Layer 0)
    SimpleAdapter(768, 768),        # Adapter 1 (after Layer 1)
    SimpleAdapter(768, 768),        # Adapter 2 (after Layer 2)
    SimpleProj(768, 768, relu=True) # Final projection (after EOS token)
])
```

**SimpleAdapter Architecture:**
```python
# 768 → 384 → 768 (bottleneck)
fc1: Linear(768, 384)
act: LeakyReLU(0.2)
fc2: Linear(384, 768)
```

**Integration Pattern:**
```
Token Embedding [B, 77, 768]
    ↓
Positional Embedding
    ↓
┌─────────────────────┐
│ Transformer Layer 0 │ (frozen)
└──────┬──────────────┘
       ↓
[text_adapter[0]] ← TRAINABLE
       ↓
Weighted Residual: x = 0.1 × adapted + 0.9 × original
       ↓
... (repeat for layers 1-2)
       ↓
Transformer Layers 3-11 (frozen, no adapters)
       ↓
Layer Norm (frozen)
       ↓
Extract EOS token [B, 768]
       ↓
[text_adapter[3]] ← TRAINABLE final projection
       ↓
Output [B, 768]
```

**Text Prompt Engineering:**
- Normal prompts: 6 variations (e.g., "flawless {object}", "perfect {object}")
- Abnormal prompts: 10 variations (e.g., "damaged {object}", "{object} with flaw")
- Templates: 7 templates (e.g., "a photo of a {}", "a cropped photo of the {}")
- **Total per class:** 6×7 + 10×7 = 112 text prompts → averaged to 2 embeddings (normal, abnormal)

---

### 2. Image Encoder with Adapters

**Base:** CLIP Vision Transformer (ViT-L/14, 24 layers, 1024-dim hidden)
**Trainable Components:** 11 modules (Stage 2 only)

```python
# Location: adapter.py:27-39
self.image_adapter = nn.ModuleDict({
    "layer_adapters": nn.ModuleList([
        SimpleAdapter(1024, 1024)  # 6 adapters for ViT layers 0-5
        for _ in range(6)
    ]),
    "seg_proj": nn.ModuleList([
        SimpleProj(1024, 768, relu=True)  # 4 projections for multi-scale features
        for _ in range(4)  # levels = [6, 12, 18, 24]
    ]),
    "det_proj": SimpleProj(1024, 768, relu=True)  # 1 projection for classification
})
```

**SimpleAdapter for Images:**
```python
# 1024 → 512 → 1024 (bottleneck)
fc1: Linear(1024, 512)
act: LeakyReLU(0.2)
fc2: Linear(512, 1024)
```

**SimpleProj for Projections:**
```python
# 1024 → 768 + LeakyReLU
fc: Linear(1024, 768)
act: LeakyReLU(0.2)
```

**Integration Pattern:**
```
Input Image [B, 3, 518, 518]
    ↓
Patch Embedding (Conv2d, stride=14) → [B, 1024, 37, 37]
    ↓
Reshape + Permute → [B, 1369, 1024]  (37×37 = 1369 patches)
    ↓
Add CLS Token → [B, 1370, 1024]
    ↓
Add Positional Embeddings (FROZEN, from CLIP pretraining)
    ↓
LayerNorm + Permute → [1370, B, 1024]
    ↓
┌──────────────────────┐
│ ViT Layer 0 (frozen) │
└────┬─────────────────┘
     ↓
[layer_adapters[0]] ← TRAINABLE
     ↓
Weighted Residual: x = 0.1 × adapted + 0.9 × original
     ↓
... (repeat for layers 1-5)
     ↓
┌──────────────────────┐
│ ViT Layer 6 (frozen) │ ← Save patches (remove CLS)
└────┬─────────────────┘
     ↓
ViT Layers 7-11 (frozen, no adapters)
     ↓
┌──────────────────────┐
│ ViT Layer 12         │ ← Save patches
└────┬─────────────────┘
     ↓
... (continue to layer 24)
     ↓
Multi-scale Features:
- tokens[0]: Layer 6  [B, 1369, 1024]
- tokens[1]: Layer 12 [B, 1369, 1024]
- tokens[2]: Layer 18 [B, 1369, 1024]
- tokens[3]: Layer 24 [B, 1369, 1024]
```

**Post-processing:**
```python
# Apply layer norm (frozen)
tokens = [ln_post(t) for t in tokens]

# Apply trainable projections (1024 → 768)
seg_tokens = [seg_proj[i](t) for i, t in enumerate(tokens)]

# L2 normalization
seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens]

# Detection token (for classification)
det_token = det_proj(tokens[-1])  # [B, 1369, 768]
det_token = F.normalize(det_token, dim=-1).mean(1)  # [B, 768]

return seg_tokens, det_token
```

---

## Two-Stage Training Process

### Stage 1: Text Adapter Training

**Objective:** Learn to generate anomaly-aware text embeddings
**Duration:** 5 epochs
**Batch Size:** 16
**Learning Rate:** 1e-5
**Trainable:** 4 text adapter modules ONLY

**Training Loop (train.py:38-114):**
```python
for epoch in range(5):
    for batch in dataloader:  # batch_size = 16
        image = batch["image"]        # [16, 3, 518, 518]
        mask = batch["mask"]          # [16, 1, 518, 518]
        class_names = batch["class_name"]  # ["bottle", "cable", ...]

        # ========================================
        # TEXT FORWARD PASS (TRAINABLE)
        # ========================================
        text_features = {}
        for class_name in set(class_names):
            # Generate 112 prompts per class
            prompts = generate_prompts(class_name)  # 6 normal + 10 abnormal
            tokens = tokenize(prompts)  # [112, 77]

            # Forward through text encoder WITH adapters
            embeddings = model.encode_text(tokens)  # [112, 768]
            embeddings = F.normalize(embeddings, dim=-1)

            # Average: 56 normal prompts → 1 vector, 56 abnormal → 1 vector
            normal_emb = embeddings[:56].mean(0)    # [768]
            abnormal_emb = embeddings[56:].mean(0)  # [768]

            text_features[class_name] = torch.stack([normal_emb, abnormal_emb], dim=1)
            # Shape: [768, 2]

        # Batch lookup
        epoch_text_feature = torch.stack(
            [text_features[cn] for cn in class_names], dim=0
        )  # [16, 768, 2]

        # ========================================
        # IMAGE FORWARD PASS (FROZEN)
        # ========================================
        with torch.no_grad():
            _, patch_features = clip_surgery.encode_image(image, [6, 12, 18, 24])
            # patch_features: 4 tensors, each [16, 1369, 768]

            # Add CLS token to each scale (residual connection)
            cls_token, _ = model.clipmodel.encode_image(image, [])
            cls_token = F.normalize(cls_token, dim=-1)  # [16, 768]
            patch_features = [f + cls_token.unsqueeze(1) for f in patch_features]

        # ========================================
        # LOSS CALCULATION (BUG: only last scale!)
        # ========================================
        for f in patch_features:  # Overwrites loss!
            # Calculate similarity map
            patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size=518)
            # [16, 1369, 768] × [16, 768, 2] → [16, 2, 518, 518]

            # Segmentation loss
            loss = calculate_seg_loss(patch_preds, mask)
            # FocalLoss + 2×DiceLoss

            # Orthogonal loss (encourage normal ⊥ abnormal)
            orthogonal_loss = (
                (epoch_text_feature[:, :, 0] * epoch_text_feature[:, :, 1])
                .sum(1).mean()
            ) ** 2
            loss += orthogonal_loss * 0.1

        # ========================================
        # BACKPROP (only text adapters)
        # ========================================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save checkpoint
torch.save({
    'epoch': 5,
    'text_adapter': model.text_adapter.state_dict(),
    'text_optimizer': optimizer.state_dict(),
}, "text_adapter.pth")
```

**Stage 1 Output:**
- Trained text adapters saved in `text_adapter.pth`
- Text encoder can now generate orthogonal normal/abnormal embeddings

---

### Stage 2: Image Adapter Training

**Objective:** Learn to extract anomaly-aware visual features
**Duration:** 20 epochs
**Batch Size:** 2 (reduced due to memory)
**Learning Rate:** 5e-4 (50× higher than Stage 1)
**Trainable:** 11 image adapter modules ONLY

**Pre-computation (BEFORE training loop):**
```python
# train.py:338-344
del text_dataloader, text_dataset, clip_surgery, text_optimizer
torch.cuda.empty_cache()

with torch.no_grad():  # FROZEN!
    text_embeddings = get_adapted_text_embedding(model, dataset, device)
    # Result: Dictionary mapping each class → [768, 2] tensor
    # Example: {
    #     "bottle": [768, 2],
    #     "cable": [768, 2],
    #     ...
    # }
```

**Training Loop (train.py:117-174):**
```python
for epoch in range(20):
    for batch in dataloader:  # batch_size = 2
        image = batch["image"]    # [2, 3, 518, 518]
        mask = batch["mask"]      # [2, 1, 518, 518]
        label = batch["label"]    # [2] (0=normal, 1=abnormal)
        class_names = batch["class_name"]  # ["bottle", "cable"]

        # ========================================
        # TEXT EMBEDDINGS: DICTIONARY LOOKUP (FROZEN)
        # ========================================
        epoch_text_feature = torch.stack(
            [text_embeddings[cn] for cn in class_names], dim=0
        )  # [2, 768, 2]

        # ========================================
        # IMAGE FORWARD PASS (WITH ADAPTERS!)
        # ========================================
        patch_features, det_feature = model(image)
        # patch_features: 4 tensors, each [2, 1369, 768]
        # det_feature: [2, 768]

        # ========================================
        # LOSS CALCULATION
        # ========================================
        loss = 0.0

        # 1) Classification Loss (Lcls)
        det_feature_expanded = det_feature.unsqueeze(1)  # [2, 1, 768]
        cls_preds = torch.matmul(det_feature_expanded, epoch_text_feature)[:, 0]
        # [2, 1, 768] × [2, 768, 2] → [2, 1, 2] → [2, 2]
        loss += F.cross_entropy(cls_preds, label)

        # 2) Segmentation Loss (Lseg) - Multi-scale
        for f in patch_features:  # ACCUMULATES across 4 scales!
            patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size=518)
            # [2, 1369, 768] × [2, 768, 2] → [2, 2, 518, 518]
            loss += calculate_seg_loss(patch_preds, mask)

        # Total: Lcls + 4×Lseg

        # ========================================
        # BACKPROP (only image adapters)
        # ========================================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # MultiStepLR: decay at steps [16000, 32000]

# Save checkpoint
torch.save({
    'epoch': 20,
    'image_adapter': model.image_adapter.state_dict(),
    'image_optimizer': optimizer.state_dict(),
}, "image_adapter.pth")
```

**Stage 2 Output:**
- Trained image adapters saved in `image_adapter.pth`
- Image encoder can now extract multi-scale anomaly-aware features

---

## Key Design Decisions

### 1. Why Adapters Instead of Full Fine-tuning?

**Parameter Efficiency:**
- CLIP ViT-L/14: ~300M parameters (FROZEN)
- Text adapters: ~2M parameters (0.67% of CLIP)
- Image adapters: ~8M parameters (2.67% of CLIP)
- **Total trainable: ~10M (3.3% of base model)**

**Benefits:**
- Preserves CLIP's zero-shot capability
- Prevents catastrophic forgetting
- Faster training, less overfitting
- Easy to swap adapters for different tasks

### 2. Why Two-Stage Training?

**Stage 1 (Text):**
- Learns domain-specific anomaly concepts
- Small batch size (16) sufficient for text
- Orthogonal loss ensures normal ⊥ abnormal

**Stage 2 (Image):**
- Learns to extract visual features aligned with text concepts
- Larger batch size would be better but limited by memory
- Text frozen → no drift in text space

**Alternative (not used):** Joint training could cause instability

### 3. Why Weighted Residual (0.1×adapted + 0.9×original)?

**Gradual Adaptation:**
- 90% of original CLIP preserved
- Prevents drastic feature space distortion
- Ensures smooth gradient flow (prevents vanishing gradients)

**Empirical Choice:**
- 0.1 is a hyperparameter (can be tuned)
- Similar to LoRA's scaling factor concept

### 4. Why Multi-scale Features [6, 12, 18, 24]?

**Semantic Hierarchy:**
- Layer 6: Low-level (edges, textures) → small defects
- Layer 12: Mid-level (parts, structures) → structural anomalies
- Layer 18: High-level (objects) → object-level issues
- Layer 24: Global semantic → context understanding

**Pyramid Fusion:**
- Combines coarse and fine-grained information
- Similar to FPN (Feature Pyramid Network) in object detection

### 5. Why Adapters Only on First 6 ViT Layers?

**Empirical Finding:**
- Later layers encode more semantic information
- Early layers more flexible for adaptation
- Keeps high-level semantics frozen

**Trade-off:**
- More adapters = more expressiveness but also more parameters
- 6 layers is a sweet spot (found empirically)

---

## Loss Functions

### 1. Segmentation Loss (Lseg)

**Components:**
```python
def calculate_seg_loss(patch_preds, mask):
    # patch_preds: [B, 2, H, W] (normal/abnormal probabilities)
    # mask: [B, 1, H, W] (binary ground truth)

    loss = FocalLoss(patch_preds, mask)
    loss += BinaryDiceLoss(patch_preds[:, 0, :, :], 1 - mask)  # Normal channel
    loss += BinaryDiceLoss(patch_preds[:, 1, :, :], mask)      # Abnormal channel
    return loss
```

**FocalLoss** (forward_utils.py:21-109):
- Addresses class imbalance (normal pixels >> abnormal pixels)
- Formula: `FL = -α(1-pt)^γ log(pt)`
- Default: γ=2, α=dynamic

**BinaryDiceLoss** (forward_utils.py:112-126):
- Measures overlap between prediction and ground truth
- Formula: `Dice = 1 - (2|X∩Y| + smooth) / (|X| + |Y| + smooth)`
- Smooth term prevents division by zero

**Why 3 terms?**
- FocalLoss: Pixel-wise classification
- DiceLoss (normal): Encourages high normal scores on normal regions
- DiceLoss (abnormal): Encourages high abnormal scores on defect regions

### 2. Orthogonal Loss (Lorth) - Stage 1 Only

```python
orthogonal_loss = (
    (text_normal * text_abnormal).sum(dim=1).mean()
) ** 2
```

**Purpose:**
- Encourages normal and abnormal embeddings to be perpendicular
- Maximizes semantic separation in embedding space
- Weight: 0.1 (text_norm_weight)

**Why orthogonal?**
- If normal · abnormal = 0, they are maximally different
- Easier for classifier to distinguish

### 3. Classification Loss (Lcls) - Stage 2 Only

```python
loss = F.cross_entropy(cls_preds, label)
# cls_preds: [B, 2] (normal/abnormal logits)
# label: [B] (0=normal, 1=abnormal)
```

**Purpose:**
- Image-level classification (is image defective?)
- Complements pixel-level segmentation
- Uses global feature (det_token)

---

## Important Implementation Details

### 1. Stage 1 Multi-scale Loss Bug

**Current Code (train.py:87-96):**
```python
for f in patch_features:  # 4 scales
    patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
    loss = calculate_seg_loss(patch_preds, mask)  # OVERWRITES!
    orthogonal_loss = (...)
    loss += orthogonal_loss * text_norm_weight
```

**Problem:**
- Loop overwrites `loss` variable
- Only the LAST scale (layer 24) contributes to gradient
- Scales [6, 12, 18] are computed but discarded

**Intended Behavior (Stage 2 does this correctly):**
```python
loss = 0.0
for f in patch_features:
    patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
    loss += calculate_seg_loss(patch_preds, mask)  # ACCUMULATE
```

**Impact:**
- May underutilize multi-scale features in Stage 1
- Stage 2 accumulates correctly, so final model still benefits

### 2. Dataset Augmentation Differences

**Stage 1 (text=True):**
```python
# dataset/base_dataset.py
transforms.RandomRotation(5),
transforms.RandomCrop(img_size),
transforms.RandomHorizontalFlip(0.5),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# NO ColorJitter!
```

**Stage 2 (text=False):**
```python
# Same as above PLUS:
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
# Applied with 70% probability
```

**Rationale:**
- Stage 1: Stable appearance for text alignment
- Stage 2: More augmentation for robustness

### 3. Positional Embeddings are Frozen

**Source:** Pretrained CLIP checkpoint (OpenAI, 2021)
**Shape:** [1370, 1024] for ViT-L/14
**Status:** NEVER trained during AA-CLIP

**Common Misconception:**
- ❌ "Positional embeddings learn during AA-CLIP training"
- ✅ They were learned by OpenAI during CLIP pretraining and are now frozen

### 4. Text Embedding Pre-computation

**Stage 1:** Text embeddings generated dynamically each batch
**Stage 2:** Text embeddings computed ONCE before training, then cached

**Why pre-compute?**
- Stage 2 freezes text encoder
- Same text features used for all 20 epochs
- Saves ~30% training time

**Storage:**
```python
text_embeddings = {
    "bottle": torch.tensor([768, 2]),
    "cable": torch.tensor([768, 2]),
    # ... for all 15 classes (MVTec) or 12 classes (VisA)
}
```

### 5. Similarity Map Calculation

**Code (forward_utils.py:196-216):**
```python
def calculate_similarity_map(patch_features, text_features, img_size, test=False, domain="Medical"):
    # patch_features: [B, L, 768] where L=1369 patches
    # text_features: [B, 768, 2] (normal/abnormal)

    # Compute similarity
    patch_scores = 100.0 * torch.matmul(patch_features, text_features)
    # [B, 1369, 768] × [B, 768, 2] → [B, 1369, 2]

    # Reshape to spatial grid
    B, L, C = patch_scores.shape  # B, 1369, 2
    H = int(np.sqrt(L))  # 37
    patch_pred = patch_scores.permute(0, 2, 1).view(B, C, H, H)
    # [B, 1369, 2] → [B, 2, 37, 37]

    # Test-time: Apply Gaussian blur + combine channels
    if test:
        sigma = 1 if domain == "Industrial" else 1.5
        kernel_size = 7 if domain == "Industrial" else 9
        patch_pred = (patch_pred[:, 1] + 1 - patch_pred[:, 0]) / 2
        patch_pred = gaussian_blur2d(patch_pred.unsqueeze(1), (kernel_size, kernel_size), (sigma, sigma))

    # Upsample to image size
    patch_preds = F.interpolate(patch_pred, size=img_size, mode='bilinear', align_corners=True)
    # [B, 2, 37, 37] → [B, 2, 518, 518]

    # Training: Apply softmax
    if not test and C > 1:
        patch_preds = torch.softmax(patch_preds, dim=1)

    return patch_preds
```

**Key Points:**
- Multiply by 100: Scale up cosine similarity for sharper predictions
- Gaussian blur (test only): Smooth predictions, reduce noise
- Bilinear interpolation: Upsample 37×37 → 518×518

---

## Data Flow Summary

### Complete Forward Pass (Stage 2)

```
INPUT: Image [2, 3, 518, 518]
    ↓
──────────────────────────────
IMAGE PREPROCESSING
──────────────────────────────
    ↓
Patch Embedding (Conv2d) → [2, 1024, 37, 37]
Reshape → [2, 1369, 1024]
Add CLS Token → [2, 1370, 1024]
Add Positional Embeddings (frozen)
LayerNorm + Permute → [1370, 2, 1024]
    ↓
──────────────────────────────
ViT LAYERS 0-5 (WITH ADAPTERS)
──────────────────────────────
    ↓
FOR i in [0, 1, 2, 3, 4, 5]:
    Transformer Layer i (frozen) → [1370, 2, 1024]
    ↓
    Adapter i (trainable) → [1370, 2, 1024]
    ↓
    Norm-preserving scale
    ↓
    Weighted residual: 0.1×adapted + 0.9×original
    ↓
    ↓
──────────────────────────────
ViT LAYER 6 (NO ADAPTER)
──────────────────────────────
    ↓
Transformer Layer 6 (frozen) → [1370, 2, 1024]
SAVE: tokens[0] = patches [1369, 2, 1024]
    ↓
──────────────────────────────
ViT LAYERS 7-11 (NO ADAPTERS)
──────────────────────────────
    ↓
Transformer Layers 7-11 (frozen)
    ↓
──────────────────────────────
ViT LAYER 12 (NO ADAPTER)
──────────────────────────────
    ↓
Transformer Layer 12 (frozen) → [1370, 2, 1024]
SAVE: tokens[1] = patches [1369, 2, 1024]
    ↓
... (same for layers 13-17)
    ↓
──────────────────────────────
ViT LAYER 18 (NO ADAPTER)
──────────────────────────────
    ↓
Transformer Layer 18 (frozen) → [1370, 2, 1024]
SAVE: tokens[2] = patches [1369, 2, 1024]
    ↓
... (same for layers 19-23)
    ↓
──────────────────────────────
ViT LAYER 24 (NO ADAPTER)
──────────────────────────────
    ↓
Transformer Layer 24 (frozen) → [1370, 2, 1024]
SAVE: tokens[3] = patches [1369, 2, 1024]
    ↓
──────────────────────────────
POST-PROCESSING
──────────────────────────────
    ↓
Permute to batch-first: [2, 1369, 1024] each
    ↓
Layer Norm (frozen): [2, 1369, 1024] each
    ↓
Trainable Projections (1024 → 768):
    seg_tokens[0] = seg_proj[0](tokens[0]) → [2, 1369, 768]
    seg_tokens[1] = seg_proj[1](tokens[1]) → [2, 1369, 768]
    seg_tokens[2] = seg_proj[2](tokens[2]) → [2, 1369, 768]
    seg_tokens[3] = seg_proj[3](tokens[3]) → [2, 1369, 768]
    ↓
L2 Normalize
    ↓
Detection Token:
    det_token = det_proj(tokens[3]) → [2, 1369, 768]
    det_token = normalize + mean(dim=1) → [2, 768]
    ↓
──────────────────────────────
OUTPUT
──────────────────────────────
    ↓
seg_tokens: List of 4 tensors [2, 1369, 768]
det_token: [2, 768]
    ↓
──────────────────────────────
LOSS CALCULATION
──────────────────────────────
    ↓
TEXT LOOKUP (frozen):
    epoch_text_feature = [2, 768, 2]
    ↓
1) Classification Loss:
    cls_preds = det_token @ epoch_text_feature → [2, 2]
    Lcls = CrossEntropy(cls_preds, label)
    ↓
2) Segmentation Loss (4 scales):
    FOR each seg_token in seg_tokens:
        similarity_map = seg_token @ epoch_text_feature
        → [2, 1369, 768] × [2, 768, 2] → [2, 2, 518, 518]
        Lseg += FocalLoss + 2×DiceLoss
    ↓
Total Loss = Lcls + 4×Lseg
    ↓
──────────────────────────────
BACKPROPAGATION
──────────────────────────────
    ↓
Gradients flow through:
    ✓ det_proj (1 module)
    ✓ seg_proj (4 modules)
    ✓ layer_adapters (6 modules)
    ✗ ViT transformer layers (frozen)
    ✗ Text encoder (frozen)
    ✗ Text adapters (frozen)
```

---

## Potential Improvement Areas

### 1. Stage 1 Multi-scale Loss Bug
**Current:** Only layer 24 contributes to gradient
**Fix:** Accumulate loss across all scales
```python
loss = 0.0
for f in patch_features:
    patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
    loss += calculate_seg_loss(patch_preds, mask)
orthogonal_loss = (...)
loss += orthogonal_loss * text_norm_weight
```

### 2. Batch Size in Stage 2
**Current:** batch_size=2 (memory limitation)
**Issue:** Small batch size → noisy gradients, slower convergence
**Potential Solutions:**
- Gradient accumulation
- Mixed precision training (FP16)
- Model parallelism
- Smaller image size (trade-off with accuracy)

### 3. Adapter Placement Strategy
**Current:** Adapters on first 6 ViT layers (empirical choice)
**Questions:**
- Why not layers 18-24 (high-level features)?
- Why not alternate layers (0, 2, 4, ...)?
- Why not adaptive selection based on layer importance?

**Potential Exploration:**
- Layer-wise adapter importance analysis
- Prune unnecessary adapters
- Dynamic adapter routing

### 4. Text Prompt Engineering
**Current:** Fixed templates + state descriptions
**Limitations:**
- Generic templates may not capture domain-specific anomalies
- 112 prompts per class is computationally expensive

**Potential Improvements:**
- Learnable prompt tokens (soft prompts)
- Meta-learning for prompt optimization
- Class-specific templates

### 5. Multi-scale Feature Fusion
**Current:** Simple weighted sum in loss (equal weight for all scales)
**Question:** Should different scales have different importance?

**Potential Improvements:**
- Learnable scale weights
- Attention-based fusion across scales
- Adaptive scale selection per image

### 6. Orthogonal Loss Formulation
**Current:** Encourages normal ⊥ abnormal via (normal · abnormal)²
**Limitation:** Only controls angle, not magnitude

**Alternatives:**
- Contrastive loss (push normal/abnormal apart)
- Triplet loss (anchor-positive-negative)
- InfoNCE loss

### 7. Detection Token Design
**Current:** Mean pooling of all patches from layer 24
**Question:** Are all patches equally important for classification?

**Potential Improvements:**
- Attention-weighted pooling
- Use CLS token instead of mean pooling
- Combine features from multiple layers

### 8. Norm-Preserving Scaling
**Current:** Rescales adapted features to match original norm
**Question:** Is this necessary? Does it limit expressiveness?

**Potential Exploration:**
- Remove norm rescaling
- Use LayerNorm instead
- Learnable scale factor

### 9. Adapter Architecture
**Current:** Bottleneck (1024 → 512 → 1024)
**Question:** Is bottleneck optimal?

**Alternatives:**
- Parallel adapters (à la Compacter)
- Low-rank adapters (à la LoRA)
- Convolution-based adapters (spatial inductive bias)

### 10. Two-Stage vs Joint Training
**Current:** Stage 1 (text) → Stage 2 (image)
**Question:** Would joint training improve alignment?

**Potential Exploration:**
- Joint training with curriculum learning
- Alternating optimization (text → image → text → ...)
- End-to-end training with careful learning rate scheduling

---

## Quick Reference: File Locations

### Core Architecture
- `AA-CLIP/model/adapter.py`: AdaptedCLIP model definition
- `AA-CLIP/model/adapter_modules.py`: SimpleAdapter, SimpleProj
- `AA-CLIP/model/clip.py`: CLIP model wrapper

### Training
- `AA-CLIP/train.py`: Main training script (Stages 1 & 2)
- `AA-CLIP/forward_utils.py`: Loss functions, text embedding generation

### Dataset
- `AA-CLIP/dataset/__init__.py`: Dataset loading, augmentation
- `AA-CLIP/dataset/constants.py`: Class names, prompts, paths

### Inference
- `AA-CLIP/inference.py`: Zero-shot inference script
- `AA-CLIP/segment.py`: Segmentation-specific inference

### Documentation
- `claude.md`: Stage-by-stage detailed walkthrough
- `ARCHITECTURE_REFERENCE.md`: This file (quick context)

---

## Training Command Examples

**Stage 1 (Text Adapter):**
```bash
python train.py \
    --dataset MVTec \
    --training_mode few_shot \
    --shot 32 \
    --text_epoch 5 \
    --image_epoch 0 \
    --text_batch_size 16 \
    --text_lr 0.00001 \
    --save_path ckpt/mvtec_stage1
```

**Stage 2 (Image Adapter):**
```bash
# Assumes Stage 1 checkpoint exists at ckpt/mvtec_stage1/text_adapter.pth
python train.py \
    --dataset MVTec \
    --training_mode few_shot \
    --shot 32 \
    --text_epoch 5 \
    --image_epoch 20 \
    --image_batch_size 2 \
    --image_lr 0.0005 \
    --save_path ckpt/mvtec_stage1  # Same path!
```

**Full Pipeline:**
```bash
# Both stages (text_epoch > 0 and image_epoch > 0)
python train.py \
    --dataset MVTec \
    --training_mode few_shot \
    --shot 32 \
    --text_epoch 5 \
    --image_epoch 20 \
    --text_batch_size 16 \
    --image_batch_size 2 \
    --text_lr 0.00001 \
    --image_lr 0.0005 \
    --save_path ckpt/mvtec_full
```

---

## Key Hyperparameters

| Parameter | Stage 1 | Stage 2 | Notes |
|-----------|---------|---------|-------|
| **Learning Rate** | 1e-5 | 5e-4 | Stage 2 is 50× higher |
| **Batch Size** | 16 | 2 | Stage 2 limited by memory |
| **Epochs** | 5 | 20 | Stage 1 converges faster |
| **Optimizer** | Adam (β1=0.5, β2=0.999) | Adam (same) | |
| **Scheduler** | None | MultiStepLR [16k, 32k] γ=0.5 | Step decay |
| **text_adapt_weight** | 0.1 | - | Weighted residual |
| **image_adapt_weight** | - | 0.1 | Weighted residual |
| **text_norm_weight** | 0.1 | - | Orthogonal loss weight |
| **text_adapt_until** | 3 | - | First 3 text layers |
| **image_adapt_until** | - | 6 | First 6 ViT layers |
| **levels** | - | [6, 12, 18, 24] | Multi-scale extraction |
| **img_size** | 518 | 518 | Input image size |

---

## Context Loading Checklist

When starting a new session to discuss improvements:

1. ✓ Understand the two-stage training paradigm
2. ✓ Know which components are trainable vs frozen
3. ✓ Recognize the multi-scale feature extraction strategy
4. ✓ Understand the adapter architecture (bottleneck design)
5. ✓ Know the loss functions (Lseg, Lorth, Lcls)
6. ✓ Be aware of the Stage 1 multi-scale loss bug
7. ✓ Understand text embedding pre-computation in Stage 2
8. ✓ Know the batch size differences (16 vs 2)
9. ✓ Understand the weighted residual connections (0.1 + 0.9)
10. ✓ Recognize that CLIP weights are ALWAYS frozen

---

## Summary for Quick Recall

**What is AA-CLIP?**
- Zero-shot anomaly detection using adapted CLIP
- 15 trainable modules (4 text + 11 image)
- 97% of parameters frozen

**How does it work?**
- Stage 1: Learn anomaly-aware text embeddings (5 epochs)
- Stage 2: Learn anomaly-aware visual features (20 epochs)
- Multi-scale feature fusion from ViT layers [6, 12, 18, 24]

**Key Innovation:**
- Lightweight adapters preserve CLIP's zero-shot ability
- Orthogonal text embeddings (normal ⊥ abnormal)
- Multi-scale spatial features for localization

**Main Bottleneck:**
- Stage 2 batch_size=2 (memory limitation)
- Could be improved with gradient accumulation or mixed precision

**Known Issues:**
- Stage 1 multi-scale loss bug (only layer 24 contributes)
- Can be fixed by accumulating loss instead of overwriting

---

**END OF ARCHITECTURE REFERENCE**

Use this document to quickly load context for discussing potential improvements, modifications, or experiments with the AA-CLIP architecture.
