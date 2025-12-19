# Cross-Attention Implementation Guide for Stage 1

## Overview

This document provides comprehensive implementation instructions for adding cross-attention to AA-CLIP's Stage 1 text adapter training. The cross-attention mechanism allows text embeddings to attend to image patch features, creating **vision-aware text representations**.

**Goal:** Implement cross-attention where:
- **Q (Query)** = Text features after Transformer Layer 0
- **K (Key)** and **V (Value)** = V1 patch features with Layer 24 CLS token added as residual (original paper's approach)

---

## 1. Verified Dimension Analysis

### 1.1 Model Configuration (ViT-L-14-336)

From code analysis:
- **Image size**: 518 × 518 (default `args.img_size`)
- **Patch size**: 14 × 14
- **Grid size**: 518 ÷ 14 = 37 (rounded)
- **Number of patches**: 37 × 37 = 1369
- **ViT hidden dimension**: 1024
- **Text hidden dimension**: 768
- **Output embedding dimension**: 768
- **Text context length**: 77 tokens
- **Batch size (Stage 1)**: 16

### 1.2 Image Dimension Flow (Verified from transformer.py:490-551)

```python
# Input image
x = [B, 3, 518, 518]  # B=16 for Stage 1

# After conv1 (patch embedding)
x = self.conv1(x)  # [B, 1024, 37, 37]
x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, 1024, 1369]
x = x.permute(0, 2, 1)  # [B, 1369, 1024]

# Add CLS token
x = torch.cat([cls_token, x], dim=1)  # [B, 1370, 1024]
#              ↑ 1 CLS token

# Add positional embedding
x = x + positional_embedding  # [B, 1370, 1024]

# Permute for transformer
x = x.permute(1, 0, 2)  # [1370, B, 1024]  (LND format)

# Pass through 24 ViT layers
# At layer 6: save features [1370, B, 1024]
# At layer 24: final output [1370, B, 1024]

# After all layers, permute back
x = x.permute(1, 0, 2)  # [B, 1370, 1024]  (NLD format)
```

### 1.3 Current Stage 1 Image Processing (Verified from train.py:74-85)

```python
with torch.no_grad():
    # Step 1: Get patch features at layers [6, 12, 18, 24]
    _, patch_features = clip_surgery.encode_image(image, [6, 12, 18, 24])
    # patch_features[0]: [B, 1370, 1024]  (layer 6, includes CLS)
    # patch_features[1]: [B, 1370, 1024]  (layer 12)
    # patch_features[2]: [B, 1370, 1024]  (layer 18)
    # patch_features[3]: [B, 1370, 1024]  (layer 24)

    # Step 2: Get CLS token from layer 24 (via adapted_model)
    cls_token, _ = adapted_model.clipmodel.encode_image(image, [])
    # cls_token: [B, 768]  (already projected!)

    # Step 3: Normalize CLS token
    cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
    # cls_token: [B, 768]

    # Step 4: Remove CLS from patches, keep only spatial patches
    patch_features = [
        clip_surgery.visual.ln_post(t[:, 1:, :]) for t in patch_features
    ]
    # Each: [B, 1369, 1024]  (CLS removed!)

    # Step 5: Project 1024 -> 768
    patch_features = [t @ clip_surgery.visual.proj for t in patch_features]
    # Each: [B, 1369, 768]

    # Step 6: Normalize
    patch_features = [
        t / t.norm(dim=-1, keepdim=True) for t in patch_features
    ]
    # Each: [B, 1369, 768]

    # Step 7: Add layer 24 CLS to each patch (broadcast addition)
    patch_features = [t + cls_token.unsqueeze(1) for t in patch_features]
    # Each: [B, 1369, 768]  (still 1369, CLS info added to each patch)
```

**CRITICAL NOTE:** The original paper's approach adds CLS_24 as a **residual** (element-wise broadcast addition) to each patch, NOT as a separate token. The final shape is `[B, 1369, 768]`, not `[B, 1370, 768]`. **This implementation will follow the original paper's design.**

### 1.4 Text Dimension Flow (Verified from adapter.py:114-145)

```python
def encode_text(self, text, adapt_text=True):
    # text shape: [B, 77]  (tokenized prompts)

    # Token embedding
    x = self.clipmodel.token_embedding(text)  # [B, 77, 768]

    # Positional embedding
    x = x + self.clipmodel.positional_embedding  # [B, 77, 768]

    # Permute for transformer
    x = x.permute(1, 0, 2)  # [77, B, 768]  (LND format)

    # Pass through 12 transformer layers
    for i in range(12):
        x, attn = self.clipmodel.transformer.resblocks[i](x, attn_mask=...)
        # x shape: [77, B, 768]

        # Apply adapter for first 3 layers (i < 3)
        if i < self.text_adapt_until:  # text_adapt_until = 3
            adapt_out = self.text_adapter[i](x)  # [77, B, 768]
            # Norm-preserving scaling
            adapt_out = adapt_out * x.norm(...) / adapt_out.norm(...)
            # Weighted residual
            x = self.t_w * adapt_out + (1 - self.t_w) * x
            # t_w = 0.1, so: x = 0.1 * adapt_out + 0.9 * x

    # Permute back
    x = x.permute(1, 0, 2)  # [B, 77, 768]

    # Final layer norm
    x = self.clipmodel.ln_final(x)  # [B, 77, 768]

    # Extract EOS token (last meaningful token)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]  # [B, 768]

    # Final adapter projection
    x = self.text_adapter[-1](x)  # [B, 768]

    return x  # [B, 768]
```

---

## 2. Cross-Attention Design

### 2.1 Placement

Insert cross-attention **after Transformer Layer 0** and **before text_adapter[0]**:

```
Token Embedding [B, 77, 768]
        ↓
Positional Embedding
        ↓
Permute → [77, B, 768]
        ↓
┌─────────────────────────────────┐
│ Transformer Layer 0 (FROZEN)    │
└─────────────────────────────────┘
        ↓
[77, B, 768]
        ↓
╔═════════════════════════════════╗
║ CROSS-ATTENTION (NEW!)          ║  ← INSERT HERE
║ Q = text [B, 77, 768]           ║
║ K = V = image [B, 1369, 768]    ║
║ (patches with CLS_24 residual)  ║
╚═════════════════════════════════╝
        ↓
[77, B, 768]
        ↓
┌─────────────────────────────────┐
│ text_adapter[0] (TRAINABLE)     │
└─────────────────────────────────┘
        ↓
... continue ...
```

### 2.2 Cross-Attention Dimensions

**Input:**
- **Q (Query)**: Text features after layer 0
  - Shape: `[B, 77, 768]` (after permuting from LND to NLD)
  - B = 16 for text prompts per class

- **K (Key)** and **V (Value)**: Image features
  - Shape: `[B, 1369, 768]`
  - Composed of: 1369 patches (layer 6) with CLS token (layer 24) added as residual

**Cross-Attention Calculation:**

```python
# Standard cross-attention formula:
# Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V

# Step 1: Compute attention scores
# Q: [B, 77, 768]
# K: [B, 1369, 768]
# K^T: [B, 768, 1369]
QK_T = Q @ K.transpose(-2, -1)  # [B, 77, 768] @ [B, 768, 1369] = [B, 77, 1369]

# Step 2: Scale and softmax
attn_weights = softmax(QK_T / sqrt(768))  # [B, 77, 1369]

# Step 3: Apply to values
# V: [B, 1369, 768]
output = attn_weights @ V  # [B, 77, 1369] @ [B, 1369, 768] = [B, 77, 768]
```

**Output:**
- Shape: `[B, 77, 768]` (same as input Q)
- Each text token now contains information from image patches it attended to

### 2.3 Image Context Construction

For cross-attention K and V, we use `[B, 1369, 768]` following the **original paper's approach**:

```python
# Get layer 6 patches (without their CLS)
v1_patches = patch_features_layer6[:, 1:, :]  # [B, 1369, 1024]

# Project to 768
v1_patches = ln_post(v1_patches)
v1_patches = v1_patches @ visual.proj  # [B, 1369, 768]
v1_patches = v1_patches / v1_patches.norm(dim=-1, keepdim=True)

# Get layer 24 CLS token
cls_token_24 = cls_from_layer_24  # [B, 768]
cls_token_24 = cls_token_24.unsqueeze(1)  # [B, 1, 768]

# ADD CLS as residual (broadcast addition - original paper's approach)
image_context = v1_patches + cls_token_24
# Shape: [B, 1369, 768]
#   - Each of 1369 patches now contains global CLS_24 information
#   - This matches the original Stage 1 design (train.py:85)
```

---

## 3. Implementation Instructions

### 3.1 Files to Create/Modify

Create **NEW** files (do not modify originals):

| Original File | New File | Purpose |
|---------------|----------|---------|
| `train.py` | `train_cross_attention.py` | Modified training loop |
| `model/adapter.py` | `model/adapter_cross_attention.py` | AdaptedCLIP with cross-attention |
| `forward_utils.py` | `forward_utils_cross_attention.py` | Modified text embedding generation |

### 3.2 New Module: CrossAttentionModule

Create in `model/adapter_cross_attention.py`:

```python
class CrossAttentionModule(nn.Module):
    """
    Cross-attention module for text-to-image attention.

    Q (Query): Text features [B, L_text, D]
    K (Key): Image features [B, L_img, D]
    V (Value): Image features [B, L_img, D]

    Output: [B, L_text, D]
    """
    def __init__(self, dim=768, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Learnable projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text_features, image_context):
        """
        Args:
            text_features: [B, L_text, D] - Text features (Q)
            image_context: [B, L_img, D] - Image features (K, V)

        Returns:
            output: [B, L_text, D] - Cross-attended text features
        """
        B, L_text, D = text_features.shape
        _, L_img, _ = image_context.shape

        # Project Q, K, V
        Q = self.q_proj(text_features)   # [B, L_text, D]
        K = self.k_proj(image_context)   # [B, L_img, D]
        V = self.v_proj(image_context)   # [B, L_img, D]

        # Reshape for multi-head attention
        # [B, L, D] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        Q = Q.view(B, L_text, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L_img, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L_img, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # [B, heads, L_text, head_dim] @ [B, heads, head_dim, L_img]
        # = [B, heads, L_text, L_img]
        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # [B, heads, L_text, L_img] @ [B, heads, L_img, head_dim]
        # = [B, heads, L_text, head_dim]
        output = attn_weights @ V

        # Reshape back
        # [B, heads, L_text, head_dim] -> [B, L_text, heads, head_dim] -> [B, L_text, D]
        output = output.transpose(1, 2).contiguous().view(B, L_text, D)

        # Final projection
        output = self.out_proj(output)

        return output
```

### 3.3 Modified AdaptedCLIP Class

In `model/adapter_cross_attention.py`:

```python
class AdaptedCLIPWithCrossAttention(nn.Module):
    def __init__(
        self,
        clip_model,
        text_adapt_weight: float = 0.1,
        image_adapt_weight: float = 0.1,
        text_adapt_until: int = 3,
        image_adapt_until: int = 6,
        levels: list = [6, 12, 18, 24],
        relu: bool = True,
        cross_attn_heads: int = 8,
        cross_attn_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        # ... existing initialization ...

        # NEW: Cross-attention module
        self.text_cross_attn = CrossAttentionModule(
            dim=768,
            num_heads=cross_attn_heads,
            dropout=cross_attn_dropout
        )

        # Storage for image context (set during forward pass)
        self.current_image_context = None

    def encode_text(self, text, adapt_text=True):
        """Modified to include cross-attention after layer 0"""
        if not adapt_text:
            return self.clipmodel.encode_text(text)

        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(cast_dtype)
        # x: [B, 77, 768]

        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # [77, B, 768] (LND format)

        for i in range(12):
            # Frozen transformer layer
            x, attn = self.clipmodel.transformer.resblocks[i](
                x, attn_mask=self.clipmodel.attn_mask
            )

            # NEW: Cross-attention after layer 0, before adapter 0
            if i == 0 and self.current_image_context is not None:
                # Permute for attention: [L, B, D] -> [B, L, D]
                x_attn = x.permute(1, 0, 2)  # [B, 77, 768]

                # Apply cross-attention
                cross_out = self.text_cross_attn(
                    x_attn,                       # Q: [B, 77, 768]
                    self.current_image_context    # K,V: [B, 1369, 768]
                )  # Output: [B, 77, 768]

                # Residual connection and permute back
                x = x + cross_out.permute(1, 0, 2)  # [77, B, 768]

            # Existing adapter logic
            if i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x)
                adapt_out = (
                    adapt_out
                    * x.norm(dim=-1, keepdim=True)
                    / adapt_out.norm(dim=-1, keepdim=True)
                )
                x = self.t_w * adapt_out + (1 - self.t_w) * x

        x = x.permute(1, 0, 2)  # [B, 77, 768]
        x = self.clipmodel.ln_final(x)
        x = self.text_adapter[-1](x[torch.arange(x.shape[0]), text.argmax(dim=-1)])

        return x
```

### 3.4 Modified Training Loop

In `train_cross_attention.py`:

```python
def train_text_adapter_with_cross_attention(
    adapted_model: nn.Module,
    clip_surgery: nn.Module,
    text_norm_weight: float,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    start_epoch: int,
    save_path: str,
    text_epoch: int,
    dataset_name: str,
    img_size: int,
    logger: logging.Logger,
):
    for epoch in range(start_epoch, text_epoch):
        logger.info(f"training text epoch {epoch}:")
        loss_list = []

        for input_data in tqdm(train_loader):
            image = input_data["image"].to(device)  # [16, 3, 518, 518]
            mask = input_data["mask"].to(device)    # [16, 1, 518, 518]
            class_names = input_data["class_name"]

            # ========================================
            # STEP 1: PRE-COMPUTE IMAGE CONTEXT (NEW!)
            # ========================================
            with torch.no_grad():
                # Get patch features at layer 6
                _, patch_features = clip_surgery.encode_image(image, [6])
                v1_features = patch_features[0]  # [16, 1370, 1024]

                # Extract layer 6 patches (remove CLS)
                v1_patches = v1_features[:, 1:, :]  # [16, 1369, 1024]

                # Project patches to 768
                v1_patches = clip_surgery.visual.ln_post(v1_patches)
                v1_patches = v1_patches @ clip_surgery.visual.proj  # [16, 1369, 768]
                v1_patches = v1_patches / v1_patches.norm(dim=-1, keepdim=True)

                # Get layer 24 CLS token
                cls_token_24, _ = adapted_model.clipmodel.encode_image(image, [])
                cls_token_24 = cls_token_24 / cls_token_24.norm(dim=-1, keepdim=True)
                cls_token_24 = cls_token_24.unsqueeze(1)  # [16, 1, 768]

                # ADD CLS as residual (original paper's approach)
                image_context = v1_patches + cls_token_24
                # Shape: [16, 1369, 768]
                # Each patch now contains global CLS_24 information

            # Store in model for cross-attention
            adapted_model.current_image_context = image_context

            # ========================================
            # STEP 2: GENERATE TEXT EMBEDDINGS
            # ========================================
            epoch_text_feature_dict = {}
            for class_name in list(set(class_names)):
                text_embedding = get_adapted_single_class_text_embedding(
                    adapted_model, dataset_name, class_name, device
                )
                epoch_text_feature_dict[class_name] = text_embedding

            epoch_text_feature = torch.stack(
                [epoch_text_feature_dict[class_name] for class_name in class_names],
                dim=0,
            )  # [16, 768, 2]

            # ========================================
            # STEP 3: USE V1 PATCHES FOR LOSS (same as original)
            # ========================================
            # Recompute using original behavior (same as train.py:74-85)
            with torch.no_grad():
                _, patch_features = clip_surgery.encode_image(image, [6, 12, 18, 24])
                cls_token, _ = adapted_model.clipmodel.encode_image(image, [])
                cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
                patch_features = [
                    clip_surgery.visual.ln_post(t[:, 1:, :]) for t in patch_features
                ]
                patch_features = [t @ clip_surgery.visual.proj for t in patch_features]
                patch_features = [
                    t / t.norm(dim=-1, keepdim=True) for t in patch_features
                ]
                patch_features = [t + cls_token.unsqueeze(1) for t in patch_features]

            # ========================================
            # STEP 4: CALCULATE LOSS
            # ========================================
            for f in patch_features:
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                loss = calculate_seg_loss(patch_preds, mask)
                orthogonal_loss = (
                    (epoch_text_feature[:, :, 0] * epoch_text_feature[:, :, 1])
                    .sum(1)
                    .mean()
                ) ** 2
                loss += orthogonal_loss * text_norm_weight

            # ========================================
            # STEP 5: BACKPROP
            # ========================================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            # Clear image context
            adapted_model.current_image_context = None

        logger.info(f"loss: {np.mean(loss_list)}")

        # Save checkpoint
        ckp_path = os.path.join(save_path, "text_adapter_cross_attn.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "text_adapter": adapted_model.text_adapter.state_dict(),
                "text_cross_attn": adapted_model.text_cross_attn.state_dict(),
                "text_optimizer": optimizer.state_dict(),
            },
            ckp_path,
        )

    return adapted_model
```

### 3.5 Update Optimizer

```python
# In train_cross_attention.py main():
text_optimizer = torch.optim.Adam(
    list(model.text_adapter.parameters()) +      # Original 4 modules
    list(model.text_cross_attn.parameters()),    # NEW: Cross-attention
    lr=args.text_lr,
    betas=(0.5, 0.999),
)
```

---

## 4. Dimension Verification Summary

### 4.1 Image Context for Cross-Attention

| Component | Shape | Description |
|-----------|-------|-------------|
| Layer 6 features (raw) | `[16, 1370, 1024]` | CLS + patches from layer 6 |
| Layer 6 patches only | `[16, 1369, 1024]` | Remove layer 6 CLS token |
| After projection | `[16, 1369, 768]` | 1024 → 768 via visual.proj |
| Layer 24 CLS token | `[16, 1, 768]` | Global semantic info |
| **Final image_context** | **`[16, 1369, 768]`** | patches_6 + CLS_24 (residual) |

### 4.2 Text Features for Cross-Attention

| Component | Shape | Description |
|-----------|-------|-------------|
| After token embedding | `[16, 77, 768]` | 16 prompts, 77 tokens each |
| After layer 0 (LND) | `[77, 16, 768]` | Transformer format |
| For cross-attn (NLD) | `[16, 77, 768]` | Permute for attention |
| **After cross-attn** | **`[16, 77, 768]`** | Vision-aware text |

### 4.3 Cross-Attention Operation

```
Q: [16, 77, 768]    (text tokens)
K: [16, 1369, 768]  (image patches with CLS_24 residual)
V: [16, 1369, 768]  (image patches with CLS_24 residual)

QK^T: [16, 77, 768] @ [16, 768, 1369] = [16, 77, 1369]
softmax(QK^T / sqrt(768)): [16, 77, 1369]
Output: [16, 77, 1369] @ [16, 1369, 768] = [16, 77, 768]
```

---

## 5. Training Modifications Summary

### 5.1 Trainable Parameters (Stage 1)

**Original:**
- `text_adapter[0]`: SimpleAdapter(768, 768)
- `text_adapter[1]`: SimpleAdapter(768, 768)
- `text_adapter[2]`: SimpleAdapter(768, 768)
- `text_adapter[3]`: SimpleProj(768, 768)
- **Total: 4 modules**

**With Cross-Attention:**
- `text_adapter[0-3]`: Same as above (4 modules)
- `text_cross_attn`: CrossAttentionModule(768, 8 heads)
  - `q_proj`: Linear(768, 768)
  - `k_proj`: Linear(768, 768)
  - `v_proj`: Linear(768, 768)
  - `out_proj`: Linear(768, 768)
- **Total: 5 modules**

### 5.2 Processing Order Change

**Original Stage 1:**
```
1. Load batch (16 images, class_names)
2. Generate text embeddings (text forward pass)
3. Process images (frozen ViT)
4. Calculate loss
5. Backprop (update text adapters)
```

**With Cross-Attention:**
```
1. Load batch (16 images, class_names)
2. PRE-COMPUTE image context (frozen ViT)  ← NEW!
3. Store image_context in model             ← NEW!
4. Generate text embeddings (with cross-attn)
5. Process images for loss (same as before)
6. Calculate loss
7. Backprop (update text adapters + cross-attn)
```

---

## 6. Checkpoint and Saving

### 6.1 New Checkpoint Format

```python
torch.save({
    "epoch": epoch + 1,
    "text_adapter": adapted_model.text_adapter.state_dict(),
    "text_cross_attn": adapted_model.text_cross_attn.state_dict(),  # NEW!
    "text_optimizer": optimizer.state_dict(),
}, "text_adapter_cross_attn.pth")
```

### 6.2 Loading Checkpoint

```python
# In main():
if checkpoint_exists:
    checkpoint = torch.load("text_adapter_cross_attn.pth")
    model.text_adapter.load_state_dict(checkpoint["text_adapter"])
    model.text_cross_attn.load_state_dict(checkpoint["text_cross_attn"])
    optimizer.load_state_dict(checkpoint["text_optimizer"])
```

---

## 7. Important Notes

### 7.1 Why Layer 6 Patches + Layer 24 CLS (Residual)?

- **Layer 6 patches**: Early spatial features (edges, textures, shapes)
  - Good for localization: "where is the damage?"
- **Layer 24 CLS**: Global semantic understanding
  - Good for classification: "is this damaged?"

By **adding CLS_24 as a residual** to each patch:
- Each patch embedding contains both spatial (layer 6) and global (layer 24 CLS) information
- Text tokens attend to 1369 enriched patch representations
- Follows the original paper's design principle

### 7.2 Why Use Original Paper's Residual Addition?

**Original Approach (Residual Addition):**
```python
patches = patches + cls_token  # [B, 1369, 768]
```
- ✅ CLS info is **broadcast** to all patches
- ✅ Each patch contains global context
- ✅ Matches original Stage 1 behavior (train.py:85)
- ✅ No architectural changes needed
- ✅ Simpler implementation

**Alternative (Concatenation - NOT used):**
```python
context = cat([cls_token, patches], dim=1)  # [B, 1370, 768]
```
- Would create separate CLS token
- Requires different attention dimensions
- Deviates from original paper design

### 7.3 Stage 2 Implications

After Stage 1 completes with cross-attention:
1. Text embeddings are pre-computed as usual (via `get_adapted_text_embedding`)
2. **These embeddings no longer depend on specific images** (averaged across all training images)
3. Stage 2 proceeds unchanged (uses frozen text embeddings)

---

## 8. Files to Create

### 8.1 File List

1. **`model/adapter_cross_attention.py`**
   - `CrossAttentionModule` class
   - `AdaptedCLIPWithCrossAttention` class

2. **`train_cross_attention.py`**
   - Modified `train_text_adapter_with_cross_attention` function
   - Modified `main()` with updated optimizer

3. **`forward_utils_cross_attention.py`** (optional, if text embedding generation needs changes)
   - Copy of `forward_utils.py` with any necessary modifications

### 8.2 Do NOT Modify

- Original `train.py`
- Original `model/adapter.py`
- Original `forward_utils.py`

This ensures you can compare results between original and cross-attention versions.

---

## 9. Testing the Implementation

### 9.1 Dimension Verification

Add assertions to verify dimensions:

```python
# In train_cross_attention.py
assert image_context.shape == (16, 1369, 768), f"Expected [16, 1369, 768], got {image_context.shape}"
# Verify it matches original behavior
with torch.no_grad():
    # Recompute using original method
    v1_test = v1_patches + cls_token_24
    assert torch.allclose(image_context, v1_test), "Image context mismatch with original approach"
```

### 9.2 Gradient Flow Check

```python
# After loss.backward()
for name, param in model.text_cross_attn.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.6f}")
    else:
        print(f"{name}: NO GRADIENT!")
```

---

## 10. Expected Behavior

### 10.1 What Should Happen

1. Text tokens (77 per prompt) attend to 1369 image patches (each containing both spatial and global CLS information)
2. Each text token learns which visual features are relevant
3. "damaged" might attend strongly to patches with high CLS_24 influence (global abnormality signal)
4. "bottle" might attend to patches representing bottle-shaped regions
5. Text embeddings become **vision-aware** and **spatially grounded**

### 10.2 Potential Improvements

- Better text-image alignment
- More discriminative normal vs abnormal embeddings
- Improved anomaly localization (text knows spatial structure)

---

## Appendix A: Complete Dimension Reference

### A.1 ViT-L-14-336 Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 518 × 518 |
| Patch size | 14 × 14 |
| Grid size | 37 × 37 |
| Num patches | 1369 |
| ViT width | 1024 |
| ViT layers | 24 |
| Output dim | 768 |

### A.2 Text Transformer Configuration

| Parameter | Value |
|-----------|-------|
| Context length | 77 |
| Text width | 768 |
| Text layers | 12 |
| Output dim | 768 |

### A.3 Prompts per Class

| Type | Count | Templates | Total |
|------|-------|-----------|-------|
| Normal | 3 states | 2 templates | 6 prompts |
| Abnormal | 5 states | 2 templates | 10 prompts |
| **Total** | | | **16 prompts** |

---

## Appendix B: Code Cross-Reference

| Section | File | Lines |
|---------|------|-------|
| Image encoding | `transformer.py` | 490-551 |
| Text encoding (original) | `adapter.py` | 114-145 |
| Stage 1 training | `train.py` | 38-114 |
| Image preprocessing | `train.py` | 74-85 |
| Text embedding generation | `forward_utils.py` | 138-162 |
| SimpleAdapter | `adapter_modules.py` | 6-13 |
| Prompt templates | `constants.py` | 135-148 |
