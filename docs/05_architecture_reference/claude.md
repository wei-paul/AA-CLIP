# Complete walkthrough

1. Random 16 image is sampled. Depending on the classes picked, these have their prompts generated (total of 16 (6 normal 10 abnormal)). These are the inputs
2. Now let's talk about what happens to each of the prompts through one forward pass cycle.
3. First token embedding, each of 16 prompts tokenized to 77 tokens -> embedded to 768 dim
4. Positional Embedding # (16, 77, 768)
5. Permute for transformer format x.permute(1,0,2) (77,16,768)
6. Pass through 12 transformer layers WITH adapters for the first 3 layers:
7. Will check current layers, if it's <3 then will create an adapted version where current vector "branch off" and go through the adapter layer. (create adapted version)
8. After the adapter layer will go through normalization
9. Then the vector that passed through adapter layer/normalization will then combine with original self (that didn't pass through adapter layer) (called weighted combination).
10. After the last transformer layer, we permute back to (16, 77, 768)
11. After permuting back then we can do layernorm x = self.clipmodel.ln_final(x) (16, 77, 768). Purpose is to apply normalization to every token embedding before taking EOS representation. Improves stability
12. Extract EOS token (for each of the 16 prompts), extract the last token embedding. (16, 768)
13. This EOS token is then passed through the FINAL text projection adapter where the end result is (16, 768), one embedding per prompt
14. Now that all prompts have their one embedding (after going through adapter), this is then averaged out. Single vector for normal, Single vector for abnormal
15. Stack normal and abnormal together (768, 2)

16. Now the image side, load 16 random images. I.E ```
```
    image[0]: broken_bottle_1.jpg -> class_name = "bottle"
    image[1]: perfect_bottle.jpg -> class_name = "bottle"
	...
	image[15]: grid_damaged.jpg -> class_name = "grid"
```
17. Generate Text Embeddings (The entire forward pass for text, resulting in stacked normal/abnormal (768, 2)
18. **Continuing on after stacking normal/abnormal (768, 2)
19. The (768, 2) is mapped back to the 16 images
```
epoch_text_feature = [16, 768, 2]
epoch_text_feature[0] = "bottle" [768, 2] for image[0]
epoch_text_feature[1] = "cable" [768, 2] for image[1]
```
20. Process 16 images through frozen ViT
```
# Extract patch featuers at multiple scales
_, patch_features = clip_surgery.encode_image(image, [6, 12, 18 ,24])
```
21. Image processing:
```
1. Patch Embedding (ViT conv1)
patches = conv1(image) [16, 1024, 37, 37]
# 518×518 image → 37×37 grid of patches (patch_size=14, so 518/14≈37)
patches = patches.reshape(16, 1024, -1).permute(0, 2, 1)  # [16, 1369, 1024]
# 1369 = 37×37 patches per image

2. Add CLS token + Position embeddings
x = cat([cls_token, patches], dim=1) # [16, 1378, 1024] (1 cls + 1369 patches)
x = x + positional_embedding

3. Pass through 24 ViT layers (frozen!)
   Extract features at specific layers
   levels = [6, 12, 18, 24]
   patch_features = []

Each layer is a transformer block (self-attention + FFN)
Process all 1370 tokens (CLS + patches) together
At layers 6, 12, 18, 24, save the current state of ALL tokens

Layer 6: Early features (edges, textures)
Layer 12: Mid-level features (parts, structures)
Layer 18: High-level features (objects)
Layer 24: Global semantic features

Result = patch_features = [
	[16, 1370, 1024],  # Features from layer 6
    [16, 1370, 1024],  # Features from layer 12
    [16, 1370, 1024],  # Features from layer 18
    [16, 1370, 1024],  # Features from layer 24
]
```
22. Extract CLS token from final layer. One global feature vector per image (16, 768)
23. Process patch features
```
1. Removes CLS token, keeps only patches so each is [16, 1369, 1024]
2. Project form 1024 > 768 to match text diemnsion so result is [16,1369, 768]
3. Normalize
4. Add CLS TOKEN to each patch for every previous layer (RESIDUAL CONNECTION**)
   patch_features = [t + cls_token.unsqueeze(1) for t in patch_features]
Result: each is [16, 1369, 768]
So what ends up happening is
# Before (4 separate tensors, no CLS):
patch_features[0] = layer_6_patches   # [16, 1369, 768]
patch_features[1] = layer_12_patches  # [16, 1369, 768]
patch_features[2] = layer_18_patches  # [16, 1369, 768]
patch_features[3] = layer_24_patches  # [16, 1369, 768]

# After (same 4 tensors, all have CLS_24 added):
patch_features[0] = layer_6_patches + CLS_24   # [16, 1369, 768]
patch_features[1] = layer_12_patches + CLS_24  # [16, 1369, 768]
patch_features[2] = layer_18_patches + CLS_24  # [16, 1369, 768]
patch_features[3] = layer_24_patches + CLS_24  # [16, 1369, 768]

```
```
CLS_24: "This is a broken bottle"                      │
│ (Global semantic understanding from ALL 24 layers)     │
└─────────────────────────────────────────────────────────┘
              │
              │ Inject into all scales
              ↓
    ┌─────────┼─────────┬─────────┬─────────┐
    ↓         ↓         ↓         ↓         ↓
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Layer 6 │ │Layer 12│ │Layer 18│ │Layer 24│
│patches │ │patches │ │patches │ │patches │
└────────┘ └────────┘ └────────┘ └────────┘

```
24. Loop through each of the tensors
```
# patch_features = [
#     [16, 1369, 768],  # Scale 1
#     [16, 1369, 768],  # Scale 2
#     [16, 1369, 768],  # Scale 3
#     [16, 1369, 768],  # Scale 4
# ]

# Line 87: Loop through the LIST (not a single tensor!)
for f in patch_features:
    # f is ONE tensor: [16, 1369, 768]

    # Line 89: Calculate similarity for THIS scale only
    patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
    # f: [16, 1369, 768]
    # epoch_text_feature: [16, 768, 2]
    # Result: [16, 2, 518, 518] after upsampling

    # Line 90: Calculate loss for THIS scale
    loss = calculate_seg_loss(patch_preds, mask)

    # Lines 91-96: Add orthogonal loss
    orthogonal_loss = (
        (epoch_text_feature[:, :, 0] * epoch_text_feature[:, :, 1])
        .sum(1)
        .mean()
    ) ** 2
    loss += orthogonal_loss * text_norm_weight

# Line 98-100: Backprop (only last scale's loss!)
optimizer.zero_grad()
loss.backward()
optimizer.step()

```
```
patch_features = [
    [16, 1369, 768],  # Scale 1
    [16, 1369, 768],  # Scale 2
    [16, 1369, 768],  # Scale 3
    [16, 1369, 768]   # Scale 4
]
         ↓ NO stacking, NO concatenation!
         ↓ Loop through list
         ↓
    ┌────┴────┬────┬────┐
    ↓         ↓    ↓    ↓
  Scale 1   Scale 2  Scale 3  Scale 4
[16,1369,768] [16,1369,768] [16,1369,768] [16,1369,768]
    ↓         ↓    ↓    ↓
  Similarity Similarity Similarity Similarity
  with text  with text  with text  with text
    ↓         ↓    ↓    ↓
  Loss₁     Loss₂  Loss₃  Loss₄
                         ↑
                    Only this one
                    is used! (loop overwrites)
```
⚠️ **IMPORTANT**: In Stage 1 (train.py:87-96), the loss from each scale overwrites the previous one in the loop. Only the **last scale (layer 24)** contributes to gradients. This appears to be intentional, focusing adaptation on the highest-level features.

However, in Stage 2 (train.py:151-154), losses are **accumulated** across all scales via `loss += calculate_seg_loss(...)`, so ALL scales contribute.

---

## Stage 1: Backpropagation and Weight Updates

### What Gets Updated in Stage 1

**Trainable Components** (train.py:263-267):
```python
text_optimizer = torch.optim.Adam(
    model.text_adapter.parameters(),  # ONLY text_adapter parameters
    lr=1e-5,
    betas=(0.5, 0.999),
)
```

**4 Trainable Modules in `text_adapter`** (adapter.py:41-44):
```python
self.text_adapter = nn.ModuleList([
    SimpleAdapter(768, 768),  # Adapter 0 (after Layer 0)
    SimpleAdapter(768, 768),  # Adapter 1 (after Layer 1)
    SimpleAdapter(768, 768),  # Adapter 2 (after Layer 2)
    SimpleProj(768, 768, relu=True)  # Final projection (after EOS extraction)
])
```

### Computation Graph (Forward Pass)

```
Input tokens [B, 77]
    ↓
Token Embedding (frozen) → [B, 77, 768]
    ↓
Positional Embedding (frozen)
    ↓
┌─────────────────────────┐
│ Transformer Layer 0     │ (frozen)
└──────────┬──────────────┘
           ↓
    [text_adapter[0]]  ← TRAINABLE (SimpleAdapter)
           ↓
    Weighted Residual: x = 0.1 × adapted + 0.9 × original
           ↓
┌─────────────────────────┐
│ Transformer Layer 1     │ (frozen)
└──────────┬──────────────┘
           ↓
    [text_adapter[1]]  ← TRAINABLE (SimpleAdapter)
           ↓
    Weighted Residual
           ↓
┌─────────────────────────┐
│ Transformer Layer 2     │ (frozen)
└──────────┬──────────────┘
           ↓
    [text_adapter[2]]  ← TRAINABLE (SimpleAdapter)
           ↓
    Weighted Residual
           ↓
┌─────────────────────────┐
│ Transformer Layers 3-11 │ (frozen, no adapters)
└──────────┬──────────────┘
           ↓
Layer Norm (frozen)
           ↓
Extract EOS token [B, 768]
           ↓
    [text_adapter[3]]  ← TRAINABLE (SimpleProj)
           ↓
Output embedding [B, 768]
```

### Gradient Flow (Backward Pass)

When `loss.backward()` is called (train.py:99), gradients flow **backwards**:

```
Loss (scalar)
    ↓ ∂Loss/∂output
[text_adapter[3]] (Final Projection)  ← Gradient computed FIRST
    ↓ ∂Loss/∂x
EOS extraction
    ↓
Transformer Layers 3-11 (frozen, gradients pass through)
    ↓
[text_adapter[2]] (Adapter 2)  ← Gradient computed SECOND
    ↓
Transformer Layer 2
    ↓
[text_adapter[1]] (Adapter 1)  ← Gradient computed THIRD
    ↓
Transformer Layer 1
    ↓
[text_adapter[0]] (Adapter 0)  ← Gradient computed FOURTH (last)
    ↓
Transformer Layer 0
    ↓
(Gradients stop - earlier layers not in optimizer)
```

**Key Points:**
1. **Gradients computed in REVERSE order**: Final Proj → Adapter 2 → Adapter 1 → Adapter 0
2. **All gradients computed in ONE backward pass** (not iteratively)
3. **Frozen layers don't block gradient flow** - gradients pass through but don't accumulate
4. **Residual connections** (0.9 coefficient) ensure stable gradient flow

### Parameter Update (optimizer.step())

After `optimizer.step()` (train.py:100), **all 4 modules update SIMULTANEOUSLY**:

```python
# Conceptually (Adam optimizer):
for param in [text_adapter[0], text_adapter[1], text_adapter[2], text_adapter[3]]:
    # Update rule: θ_new = θ_old - learning_rate × gradient
    # (Plus Adam momentum and adaptive learning rate adjustments)
    param.data = adam_update(param.data, param.grad, momentum_state, variance_state)
```

**Visual Timeline:**

```
FORWARD PASS (builds computation graph):
─────────────────────────────────────────────────────────────
Input → Layer0 → [Adapt0] → Layer1 → [Adapt1] → Layer2 → [Adapt2]
        → Layer3-11 → EOS → [FinalProj] → Loss

Time: t=0 ──────────────────────────────────────────────────→ t=T


BACKWARD PASS (computes gradients in reverse):
─────────────────────────────────────────────────────────────
Loss → ∂L/∂[FinalProj] → ∂L/∂EOS → ∂L/∂[Adapt2] → ∂L/∂[Adapt1] → ∂L/∂[Adapt0]

Time: t=T ←────────────────────────────────────────────────── t=0


OPTIMIZER.STEP() (updates all parameters simultaneously):
─────────────────────────────────────────────────────────────
θ_Adapt0 ← θ - α·∇L(θ)   ┐
θ_Adapt1 ← θ - α·∇L(θ)   │  All happen
θ_Adapt2 ← θ - α·∇L(θ)   ├─ at the same time
θ_FinalProj ← θ - α·∇L(θ) ┘
```

### Why Gradients Flow Smoothly Through Frozen Layers

Even though adapters are applied sequentially, gradients flow smoothly because:

1. **Residual connections** enable gradient bypass:
   ```python
   x = 0.1 * adapt_out + 0.9 * x
   # Gradient: ∂Loss/∂x_in = 0.1 × ∂Loss/∂adapt_out + 0.9 × ∂Loss/∂x_out
   ```

2. **0.9 coefficient** ensures 90% of gradient flows directly through (prevents vanishing gradients)

3. **Frozen transformer layers** act as differentiable functions - gradients pass through unchanged

---

## Stage 1 Complete: Results After Training

After 5 epochs of Stage 1 training (train.py:38-114):

### What Has Been Learned:
- **4 trainable modules** now contain optimized weights:
  - 3 SimpleAdapter modules (768→768 with LeakyReLU)
  - 1 SimpleProj module (768→768 with LeakyReLU)
- These adapters enable the text encoder to generate **anomaly-aware embeddings**
- Normal and abnormal embeddings are now **orthogonal** (perpendicular in embedding space)

### What Gets Saved:
**Checkpoint saved** (train.py:105-113):
```python
torch.save({
    'epoch': epoch + 1,
    'text_adapter': adapted_model.text_adapter.state_dict(),  # ← 4 modules
    'text_optimizer': optimizer.state_dict(),
}, "text_adapter.pth")
```

### What Gets Frozen:
**Before Stage 2 begins** (train.py:336-344):
```python
# Delete Stage 1 components
del text_dataloader, text_dataset, clip_surgery, text_optimizer
torch.cuda.empty_cache()

# Generate final text embeddings with NO gradient tracking
with torch.no_grad():
    text_embeddings = get_adapted_text_embedding(model, args.dataset, device)
    # Result: Dictionary mapping each class_name → [768, 2] tensor
    # Example: {"bottle": [768, 2], "cable": [768, 2], ...}
```

**These text embeddings are now FROZEN** for Stage 2. The text adapter weights remain in the model but are **not trainable** in Stage 2.

---

## Stage 2: Image Adapter Training

### Transition from Stage 1 → Stage 2

**Key Changes:**

| Aspect | Stage 1 | Stage 2 |
|--------|---------|---------|
| **Trainable** | Text adapters (4 modules) | Image adapters (14 modules) |
| **Frozen** | Text encoder + Image encoder | Text adapters + Text encoder |
| **Text Embeddings** | Generated dynamically each batch | Pre-computed once, reused |
| **Batch Size** | 16 | 2 |
| **Learning Rate** | 1e-5 | 5e-4 (50× higher!) |
| **Dataset** | Same data, text=True (no ColorJitter) | Same data, text=False (with ColorJitter) |
| **Loss** | Lseg + Lorth | Lcls + Lseg |
| **Multi-scale Loss** | Overwrites (last scale only) | Accumulates (all 4 scales) |

### Stage 2: Dataset Clarification

**IMPORTANT:** Stage 1 and Stage 2 use the **SAME underlying data** from the same metadata file, but with **different augmentations**:

```python
# dataset/__init__.py:200-201
text_dataset = BaseDataset(data_path, meta_path, img_size, text=True)   # Stage 1
image_dataset = BaseDataset(data_path, meta_path, img_size, text=False) # Stage 2
```

**Difference in Augmentations:**
- **Stage 1 (text=True)**: No ColorJitter (only rotation, translation, flips)
- **Stage 2 (text=False)**: Includes ColorJitter (brightness, contrast, saturation @ 70% probability)

**So NO, the images are NOT necessarily the same 16 images** - both stages:
- Load from the same training split
- Randomly sample batches (shuffle=True)
- Apply different augmentations
- Stage 2 has smaller batches (2 vs 16) due to higher memory requirements

### What Trainable in Stage 2:

**Image Adapter Components** (adapter.py:27-39):
```python
self.image_adapter = nn.ModuleDict({
    "layer_adapters": nn.ModuleList([
        SimpleAdapter(1024, 1024)  # 6 adapters for ViT layers 0-5
        for _ in range(image_adapt_until)  # default: 6
    ]),
    "seg_proj": nn.ModuleList([
        SimpleProj(1024, 768, relu)  # 4 projections for 4 scales
        for _ in range(len(levels))  # levels = [6, 12, 18, 24]
    ]),
    "det_proj": SimpleProj(1024, 768, relu)  # 1 projection for detection
})
```

**Total: 11 trainable modules**
- 6 SimpleAdapter modules (for ViT layers 0-5)
- 4 SimpleProj modules (seg_proj for 4 scales)
- 1 SimpleProj module (det_proj for classification)

---

## Stage 2: Complete Walkthrough with Code and Dimensions

### STEP 1: Pre-compute Text Embeddings (BEFORE training loop)

**Location:** train.py:338-344

```python
# After Stage 1 completes, delete Stage 1 components
del text_dataloader, text_dataset, clip_surgery, text_optimizer
torch.cuda.empty_cache()

# Pre-compute text embeddings for ALL classes (one time only!)
with torch.no_grad():  # ← NO gradients, frozen!
    text_embeddings = get_adapted_text_embedding(model, args.dataset, device)
```

**What happens in `get_adapted_text_embedding()`?** (forward_utils.py:185-192)

```python
def get_adapted_text_embedding(model, dataset_name, device):
    ret_dict = {}
    # Loop through ALL classes in the dataset
    for class_name in CLASS_NAMES[dataset_name]:
        # For MVTec: ["bottle", "cable", "capsule", ..., "zipper"] (15 classes)
        # For VisA: ["candle", "pcb3", "capsules", ..., "fryum"] (12 classes)

        text_features = get_adapted_single_class_text_embedding(
            model, dataset_name, class_name, device
        )
        # text_features shape: [768, 2]
        #   - text_features[:, 0] = normal embedding [768]
        #   - text_features[:, 1] = abnormal embedding [768]

        ret_dict[class_name] = text_features

    return ret_dict
    # Result for MVTec dataset:
    # {
    #     "bottle": [768, 2],
    #     "cable": [768, 2],
    #     "capsule": [768, 2],
    #     ...
    #     "zipper": [768, 2]
    # }
```

**Example - For MVTec dataset:**
```python
text_embeddings = {
    "bottle": torch.tensor([768, 2]),    # [normal_emb, abnormal_emb]
    "cable": torch.tensor([768, 2]),
    "capsule": torch.tensor([768, 2]),
    "carpet": torch.tensor([768, 2]),
    "grid": torch.tensor([768, 2]),
    "hazelnut": torch.tensor([768, 2]),
    "leather": torch.tensor([768, 2]),
    "metal_nut": torch.tensor([768, 2]),
    "pill": torch.tensor([768, 2]),
    "screw": torch.tensor([768, 2]),
    "tile": torch.tensor([768, 2]),
    "transistor": torch.tensor([768, 2]),
    "toothbrush": torch.tensor([768, 2]),
    "wood": torch.tensor([768, 2]),
    "zipper": torch.tensor([768, 2])
}
```

**Key Point:** These embeddings are computed ONCE and reused for all 20 epochs of Stage 2!

---

### STEP 2: Stage 2 Training Loop Begins

**Location:** train.py:345-357

```python
model = train_image_adapter(
    model=model,
    text_embeddings=text_embeddings,  # ← Pre-computed dictionary passed in
    image_epoch=args.image_epoch,      # default: 20 epochs
    train_loader=image_dataloader,     # batch_size=2
    optimizer=image_optimizer,         # lr=5e-4
    scheduler=image_scheduler,
    device=device,
    start_epoch=image_start_epoch,
    save_path=args.save_path,
    img_size=args.img_size,            # 518
    logger=logger,
)
```

---

### STEP 3: Inside `train_image_adapter()` - Per Batch

**Location:** train.py:117-174

```python
def train_image_adapter(
    model: nn.Module,
    text_embeddings: torch.Tensor,  # ← Pre-computed dictionary
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: str,
    start_epoch: int,
    save_path: str,
    image_epoch: int,
    img_size: int,
    logger: logging.Logger,
):
    for epoch in range(start_epoch, image_epoch):  # 20 epochs
        logger.info(f"training image epoch {epoch}:")
        loss_list = []

        for input_data in tqdm(train_loader):  # batch_size = 2
            # ========================================
            # LOAD BATCH DATA
            # ========================================
            image = input_data["image"].to(device)      # [2, 3, 518, 518]
            mask = input_data["mask"].to(device)        # [2, 1, 518, 518]
            label = input_data["label"].to(device)      # [2] (0=normal, 1=abnormal)
            class_names = input_data["class_name"]      # ["bottle", "cable"]

            B, C, H, W = image.shape  # B=2, C=3, H=518, W=518
```

**Example Batch:**
```python
# Batch of 2 images
image[0]: damaged_bottle.jpg    → class_name="bottle",  label=1 (abnormal)
image[1]: perfect_cable.jpg     → class_name="cable",   label=0 (normal)

# Shapes:
image:  [2, 3, 518, 518]  # 2 RGB images
mask:   [2, 1, 518, 518]  # Binary masks (1=anomaly, 0=normal)
label:  [2]               # [1, 0] (image-level labels)
class_names: ["bottle", "cable"]
```

---

### STEP 4: Lookup Pre-computed Text Embeddings

**Location:** train.py:138-142

```python
# ========================================
# TEXT EMBEDDINGS: LOOKUP (NOT FORWARD PASS!)
# ========================================
class_names = input_data["class_name"]  # ["bottle", "cable"]
epoch_text_feature = torch.stack(
    [text_embeddings[class_name] for class_name in class_names], dim=0
)
# Result: [2, 768, 2]
#   epoch_text_feature[0] = text_embeddings["bottle"]  # [768, 2]
#   epoch_text_feature[1] = text_embeddings["cable"]   # [768, 2]
```

**Dimension breakdown:**
```python
epoch_text_feature shape: [2, 768, 2]
    ↓
    [batch_size, embedding_dim, num_classes]

epoch_text_feature[0]:  # For image[0] (bottle)
    [:, 0] = normal bottle embedding    [768]
    [:, 1] = abnormal bottle embedding  [768]

epoch_text_feature[1]:  # For image[1] (cable)
    [:, 0] = normal cable embedding     [768]
    [:, 1] = abnormal cable embedding   [768]
```

**IMPORTANT:**
- ❌ **NO text encoder forward pass**
- ❌ **NO gradient computation for text**
- ✅ **Just dictionary lookup** (O(1) operation)
- ✅ **Text embeddings are frozen tensors**

---

### STEP 5: Image Forward Pass with Adapters

**Location:** train.py:144-145

```python
# ========================================
# IMAGE FORWARD PASS (WITH ADAPTERS!)
# ========================================
patch_features, det_feature = model(image)  # model.forward() in adapter.py
```

**What happens in `model.forward()`?** (adapter.py:67-112)

Let me break this down step-by-step with dimensions:

#### 5.1: Patch Embedding

```python
# adapter.py:68-70
x = self.image_encoder.conv1(x)  # Conv2d with stride=14
# Input:  [2, 3, 518, 518]
# Output: [2, 1024, 37, 37]  (518÷14 ≈ 37)

x = x.reshape(x.shape[0], x.shape[1], -1)  # [2, 1024, 1369]
x = x.permute(0, 2, 1)                     # [2, 1369, 1024]
# 1369 = 37 × 37 patches
```

#### 5.2: Add CLS Token + Positional Embedding

```python
# adapter.py:72-82
x = torch.cat([
    self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(...),
    x,
], dim=1)
# Output: [2, 1370, 1024]  (1 CLS + 1369 patches)

x = x + self.image_encoder.positional_embedding.to(x.dtype)
# Output: [2, 1370, 1024]

x = self.image_encoder.patch_dropout(x)
x = self.image_encoder.ln_pre(x)
x = x.permute(1, 0, 2)  # [1370, 2, 1024] (transformer format)
```

#### 5.3: Pass Through 24 ViT Layers WITH Adapters

```python
# adapter.py:89-101
tokens = []
for i in range(24):  # 24 ViT transformer layers
    # Frozen transformer block
    x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
    # x shape: [1370, 2, 1024]

    # Apply adapter ONLY for first 6 layers (i < image_adapt_until)
    if i < self.image_adapt_until:  # image_adapt_until = 6
        # Pass through trainable adapter
        adapt_out = self.image_adapter["layer_adapters"][i](x)
        # adapt_out shape: [1370, 2, 1024]

        # Norm-preserving scaling
        adapt_out = (
            adapt_out
            * x.norm(dim=-1, keepdim=True)
            / adapt_out.norm(dim=-1, keepdim=True)
        )

        # Weighted residual connection
        x = self.i_w * adapt_out + (1 - self.i_w) * x
        # x = 0.1 * adapt_out + 0.9 * x
        # x shape: [1370, 2, 1024]

    # Save features at specific layers [6, 12, 18, 24]
    if i + 1 in self.levels:  # levels = [6, 12, 18, 24]
        tokens.append(x[1:, :, :])  # Remove CLS, keep patches
        # tokens[-1] shape: [1369, 2, 1024]

# After loop:
# tokens = [
#     [1369, 2, 1024],  # From layer 6
#     [1369, 2, 1024],  # From layer 12
#     [1369, 2, 1024],  # From layer 18
#     [1369, 2, 1024],  # From layer 24
# ]
```

**Visual representation:**
```
Input [2, 3, 518, 518]
    ↓
Conv1 + Reshape + Permute
    ↓
[1370, 2, 1024]  (CLS + patches, batch, features)
    ↓
┌─────────────────────┐
│ ViT Layer 0 (frozen)│
└──────┬──────────────┘
       ↓
[layer_adapters[0]] ← TRAINABLE
       ↓
Weighted Residual (0.1×adapted + 0.9×original)
       ↓
┌─────────────────────┐
│ ViT Layer 1 (frozen)│
└──────┬──────────────┘
       ↓
[layer_adapters[1]] ← TRAINABLE
       ↓
... (repeat for layers 2-5)
       ↓
┌─────────────────────┐
│ ViT Layer 6 (frozen)│  ← Save patches here
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ ViT Layers 7-11     │  (no adapters)
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ ViT Layer 12        │  ← Save patches here
└──────┬──────────────┘
       ↓
... (continue to layer 24)
```

#### 5.4: Post-process Multi-scale Patch Features

```python
# adapter.py:103-109
x = x.permute(1, 0, 2)  # [2, 1370, 1024]
tokens = [t.permute(1, 0, 2) for t in tokens]  # Each: [2, 1369, 1024]

tokens = [self.image_encoder.ln_post(t) for t in tokens]
# Layer normalization, each: [2, 1369, 1024]

seg_tokens = [
    self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)
]
# Project from 1024 → 768 using TRAINABLE projections
# seg_tokens[0]: [2, 1369, 768]  (from layer 6)
# seg_tokens[1]: [2, 1369, 768]  (from layer 12)
# seg_tokens[2]: [2, 1369, 768]  (from layer 18)
# seg_tokens[3]: [2, 1369, 768]  (from layer 24)

seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens]
# L2 normalization, shapes unchanged
```

#### 5.5: Create Detection Token (for Classification)

```python
# adapter.py:110-111
det_token = self.image_adapter["det_proj"](tokens[-1])
# tokens[-1] is from layer 24: [2, 1369, 1024]
# det_proj: TRAINABLE projection 1024 → 768
# Output: [2, 1369, 768]

det_token = F.normalize(det_token, dim=-1).mean(1)
# Normalize then average across patches
# Output: [2, 768]  (one global feature per image)
```

#### 5.6: Return Results

```python
# adapter.py:112
return seg_tokens, det_token

# seg_tokens (list of 4 tensors):
#   [0]: [2, 1369, 768]  # Scale 1 (layer 6)
#   [1]: [2, 1369, 768]  # Scale 2 (layer 12)
#   [2]: [2, 1369, 768]  # Scale 3 (layer 18)
#   [3]: [2, 1369, 768]  # Scale 4 (layer 24)
#
# det_token: [2, 768]  # Global feature for classification
```

---

### STEP 6: Calculate Losses

**Location:** train.py:146-154

```python
# ========================================
# LOSS CALCULATION
# ========================================
loss = 0.0

# Classification Loss (Lcls)
det_feature = det_feature.unsqueeze(1)  # [2, 768] → [2, 1, 768]
cls_preds = torch.matmul(det_feature, epoch_text_feature)[:, 0]
# Matmul: [2, 1, 768] × [2, 768, 2] → [2, 1, 2]
# [:, 0] extracts: [2, 2]
#   cls_preds[0] = [score_normal_bottle, score_abnormal_bottle]
#   cls_preds[1] = [score_normal_cable, score_abnormal_cable]

loss += F.cross_entropy(cls_preds, label)
# cls_preds: [2, 2]  (logits for 2 classes)
# label: [2]         ([1, 0] = [abnormal, normal])
# Computes classification loss

# Segmentation Loss (Lseg) - Multi-scale
for f in patch_features:  # 4 scales
    # f shape: [2, 1369, 768]

    # Calculate similarity map
    patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
    # Output: [2, 2, 518, 518]

    # Calculate segmentation loss
    loss += calculate_seg_loss(patch_preds, mask)
    # patch_preds: [2, 2, 518, 518]
    # mask: [2, 1, 518, 518]
    # Adds FocalLoss + 2×DiceLoss

# Total loss = Lcls + 4×Lseg (accumulated across all scales!)
```

**Dimension walkthrough for similarity calculation:**

```python
# In calculate_similarity_map() - forward_utils.py:196-216
patch_anomaly_scores = 100.0 * torch.matmul(patch_features, epoch_text_feature)
# [2, 1369, 768] × [2, 768, 2] → [2, 1369, 2]
# Multiply by 100 for scaling

B, L, C = patch_anomaly_scores.shape  # B=2, L=1369, C=2
H = int(np.sqrt(L))  # H = 37

patch_pred = patch_anomaly_scores.permute(0, 2, 1).view(B, C, H, H)
# [2, 1369, 2] → [2, 2, 1369] → [2, 2, 37, 37]

patch_preds = F.interpolate(patch_pred, size=img_size, mode='bilinear', align_corners=True)
# [2, 2, 37, 37] → [2, 2, 518, 518]
# Upsample to original image size

# During training (not test):
patch_preds = torch.softmax(patch_preds, dim=1)
# Apply softmax across normal/abnormal channels
# Output: [2, 2, 518, 518]
#   [:, 0, :, :] = normal probability map
#   [:, 1, :, :] = abnormal probability map
```

---

### STEP 7: Backpropagation and Optimization

**Location:** train.py:155-159

```python
# ========================================
# BACKWARD PASS
# ========================================
optimizer.zero_grad()  # Clear gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update parameters
loss_list.append(loss.item())
scheduler.step()       # Update learning rate
```

**What gets updated:**
- 6 SimpleAdapter modules (ViT layers 0-5)
- 4 SimpleProj modules (seg_proj for 4 scales)
- 1 SimpleProj module (det_proj)
- **Total: 11 modules**

**What stays frozen:**
- All 24 ViT transformer layers
- Text encoder (all 12 layers)
- Text adapters (4 modules from Stage 1)

---

### Summary: Stage 2 Key Differences from Stage 1

| Aspect | Stage 1 | Stage 2 |
|--------|---------|---------|
| **Text Processing** | Dynamic forward pass each batch | Pre-computed lookup (dictionary) |
| **Text Gradients** | Computed and backpropped | None (frozen) |
| **Image Processing** | Frozen ViT | ViT with 6 trainable adapters |
| **Trainable Modules** | 4 (text adapters) | 11 (image adapters) |
| **Batch Size** | 16 | 2 |
| **Loss Components** | Lseg + Lorth | Lcls + 4×Lseg |
| **Multi-scale Loss** | Overwrites (last only) | Accumulates (all 4) |

