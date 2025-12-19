"""
Cross-Attention Enhanced Adapter Module for AA-CLIP

This module implements BIDIRECTIONAL cross-attention mechanism:
- Stage 1: Text attends to Image (vision-aware text representations)
- Stage 2: Image attends to Text (text-aware image representations)

Cross-Attention Design:
Stage 1 (Text → Image):
    - Q (Query): Text features after Transformer Layer 0 [B, 77, 768]
    - K/V: Image patches (layer 6) + CLS_24 residual [B, 1369, 768]
    - Placement: After text layer 0, before text_adapter[0]

Stage 2 (Image → Text):
    - Q (Query): Image features after ViT Layer 0 [B, 1370, 1024]
    - K/V: Text features after Stage 1 adapter[0] [B, 1232, 768]
    - Placement: After ViT layer 0, before image_adapter[0]
    - Requires projection: 1024 ↔ 768

Author: Cross-attention implementation for AA-CLIP
"""

import torch
from torch import nn
import torch.nn.functional as F
from .adapter_modules import SimpleAdapter, SimpleProj


class CrossAttentionModule(nn.Module):
    """
    Multi-head Cross-Attention Module for text-to-image attention (Stage 1).

    This module allows text tokens to attend to image patch features,
    enabling the text encoder to generate vision-aware embeddings.

    Architecture:
        Q = W_q(text_features)      # Query from text
        K = W_k(image_context)      # Key from image
        V = W_v(image_context)      # Value from image
        Attention = softmax(QK^T / sqrt(d_k)) @ V
        Output = W_o(Attention)

    Args:
        dim (int): Embedding dimension (default: 768)
        num_heads (int): Number of attention heads (default: 8)
        dropout (float): Dropout probability (default: 0.1)

    Input Shapes:
        text_features: [B, L_text, D] where L_text=77, D=768
        image_context: [B, L_img, D] where L_img=1369, D=768

    Output Shape:
        [B, L_text, D] = [B, 77, 768]
    """

    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        # Learnable projection layers
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for projection layers"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, text_features, image_context, debug=False):
        """
        Forward pass for cross-attention.

        Args:
            text_features: Text features [B_text, L_text, D] - Query source
            image_context: Image features [B_img, L_img, D] - Key/Value source
                          If B_img != B_text, uses mean of image contexts
            debug: If True, print debug information

        Returns:
            output: Cross-attended text features [B_text, L_text, D]
        """
        B_text, L_text, D = text_features.shape
        B_img, L_img, _ = image_context.shape

        # Handle batch size mismatch:
        # During training, we have 16 images but process 6 or 10 prompts at a time
        # Solution: Use mean of image contexts for all prompts (aggregate context)
        if B_text != B_img:
            if debug:
                print(f"\n[BATCH SIZE MISMATCH HANDLING]")
                print(f"  B_text={B_text}, B_img={B_img}")
                print(f"  Aggregating image context: mean across batch dimension")

            image_context = image_context.mean(dim=0, keepdim=True)  # [1, L_img, D]
            image_context = image_context.expand(B_text, -1, -1)  # [B_text, L_img, D]

            if debug:
                print(f"  After aggregation: image_context shape: {image_context.shape}")

        B = B_text

        if debug:
            print("\n" + "="*60)
            print("TEXT CROSS-ATTENTION MODULE DEBUG (Stage 1)")
            print("="*60)
            print(f"[INPUT] text_features shape: {text_features.shape}")
            print(f"[INPUT] image_context shape: {image_context.shape}")
            print(f"[CONFIG] num_heads: {self.num_heads}, head_dim: {self.head_dim}")
            print(f"[CONFIG] scale factor: {self.scale:.6f}")

        # Project Q, K, V
        Q = self.q_proj(text_features)   # [B, L_text, D]
        K = self.k_proj(image_context)   # [B, L_img, D]
        V = self.v_proj(image_context)   # [B, L_img, D]

        if debug:
            print(f"\n[PROJECTION]")
            print(f"  Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")

        # Reshape for multi-head attention
        Q = Q.view(B, L_text, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L_img, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L_img, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        if debug:
            print(f"[ATTENTION] attn_weights shape: {attn_weights.shape}")

        output = attn_weights @ V
        output = output.transpose(1, 2).contiguous().view(B, L_text, D)
        output = self.out_proj(output)

        if debug:
            print(f"[OUTPUT] shape: {output.shape}")
            print("="*60 + "\n")

        return output


class ImageCrossAttentionModule(nn.Module):
    """
    Multi-head Cross-Attention Module for image-to-text attention (Stage 2).

    This module allows image patches to attend to text token features,
    enabling the image encoder to generate text-aware patch embeddings.

    Key Challenge: Dimension mismatch
        - Image features: 1024 dimensions (ViT-L)
        - Text features: 768 dimensions (CLIP text)

    Solution: Projection layers
        - Q projection: 1024 → 768 (project image to text space)
        - K/V projection: 768 → 768 (keep text in same space)
        - Out projection: 768 → 1024 (project back to image space)

    Args:
        image_dim (int): Image embedding dimension (default: 1024)
        text_dim (int): Text embedding dimension (default: 768)
        num_heads (int): Number of attention heads (default: 8)
        dropout (float): Dropout probability (default: 0.1)

    Input Shapes:
        image_features: [B, L_img, 1024] where L_img=1370
        text_context: [B, L_text, 768] where L_text=1232 (16 prompts × 77 tokens)

    Output Shape:
        [B, L_img, 1024] = [B, 1370, 1024]
    """

    def __init__(self, image_dim=1024, text_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = text_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert text_dim % num_heads == 0, f"text_dim {text_dim} must be divisible by num_heads {num_heads}"

        # Projection layers handle dimension mismatch
        self.q_proj = nn.Linear(image_dim, text_dim)   # 1024 → 768
        self.k_proj = nn.Linear(text_dim, text_dim)    # 768 → 768
        self.v_proj = nn.Linear(text_dim, text_dim)    # 768 → 768
        self.out_proj = nn.Linear(text_dim, image_dim) # 768 → 1024

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for projection layers"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, image_features, text_context, debug=False):
        """
        Forward pass for image-to-text cross-attention.

        Args:
            image_features: Image features [B, L_img, 1024] - Query source
            text_context: Text features [B, L_text, 768] - Key/Value source
            debug: If True, print debug information

        Returns:
            output: Cross-attended image features [B, L_img, 1024]
        """
        B_img, L_img, D_img = image_features.shape
        B_text, L_text, D_text = text_context.shape

        if debug:
            print("\n" + "="*60)
            print("IMAGE CROSS-ATTENTION MODULE DEBUG (Stage 2)")
            print("="*60)
            print(f"[INPUT] image_features shape: {image_features.shape}")
            print(f"[INPUT] text_context shape: {text_context.shape}")
            print(f"[CONFIG] num_heads: {self.num_heads}, head_dim: {self.head_dim}")

        # Handle batch size mismatch
        if B_text == 1 and B_img > 1:
            if debug:
                print(f"[BATCH] Expanding text_context from batch=1 to batch={B_img}")
            text_context = text_context.expand(B_img, -1, -1)
            B_text = B_img

        assert B_img == B_text, f"Batch mismatch: image={B_img}, text={B_text}"
        B = B_img

        # Project Q from image space to text space
        Q = self.q_proj(image_features)   # [B, L_img, 1024] → [B, L_img, 768]
        K = self.k_proj(text_context)     # [B, L_text, 768]
        V = self.v_proj(text_context)     # [B, L_text, 768]

        if debug:
            print(f"[PROJECTION] Q: {Q.shape}, K: {K.shape}, V: {V.shape}")

        # Reshape for multi-head attention
        Q = Q.view(B, L_img, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L_text, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L_text, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        if debug:
            print(f"[ATTENTION] attn_weights shape: {attn_weights.shape}")

        output = attn_weights @ V
        output = output.transpose(1, 2).contiguous().view(B, L_img, self.text_dim)
        output = self.out_proj(output)  # Project back to 1024

        if debug:
            print(f"[OUTPUT] shape: {output.shape}")
            print("="*60 + "\n")

        return output


class AdaptedCLIPWithCrossAttention(nn.Module):
    """
    AA-CLIP model with BIDIRECTIONAL Cross-Attention mechanism.

    This class extends the original AdaptedCLIP by adding:
    - Stage 1: Text cross-attention (text attends to image)
    - Stage 2: Image cross-attention (image attends to text)

    Stage 1 Cross-Attention Flow:
        Text tokens [B, 77, 768]
              ↓
        Transformer Layer 0 (frozen)
              ↓
        Cross-Attention (Q=text, KV=image_patches)  ← TEXT CROSS-ATTN
              ↓
        text_adapter[0] (trainable)
              ↓
        ... continue through remaining layers ...

    Stage 2 Cross-Attention Flow:
        Image patches [B, 1370, 1024]
              ↓
        ViT Layer 0 (frozen)
              ↓
        Cross-Attention (Q=image, KV=text_context)  ← IMAGE CROSS-ATTN
              ↓
        image_adapter[0] (trainable)
              ↓
        ... continue through remaining layers ...

    Args:
        clip_model: Pre-trained CLIP model
        text_adapt_weight (float): Weight for text adapter residual (default: 0.1)
        image_adapt_weight (float): Weight for image adapter residual (default: 0.1)
        text_adapt_until (int): Apply text adapters until this layer (default: 3)
        image_adapt_until (int): Apply image adapters until this layer (default: 6)
        levels (list): ViT layers to extract features from (default: [6, 12, 18, 24])
        relu (bool): Use ReLU in projection layers (default: True)
        cross_attn_heads (int): Number of cross-attention heads (default: 8)
        cross_attn_dropout (float): Cross-attention dropout (default: 0.1)
    """

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
        text_cross_attn_weight: float = 0.1,  # NEW: Scale for text cross-attention
        image_cross_attn_weight: float = 0.1,  # NEW: Scale for image cross-attention
        **kwargs,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.text_adapt_until = text_adapt_until
        self.image_adapt_until = image_adapt_until
        self.t_w = text_adapt_weight
        self.i_w = image_adapt_weight
        self.t_ca_w = text_cross_attn_weight  # NEW: Weight for text cross-attention
        self.i_ca_w = image_cross_attn_weight  # NEW: Weight for image cross-attention
        self.levels = levels

        # Image adapters (same as original)
        layer_adapters = nn.ModuleList(
            [SimpleAdapter(1024, 1024) for _ in range(image_adapt_until)]
        )
        seg_proj = nn.ModuleList(
            [SimpleProj(1024, 768, relu) for _ in range(len(levels))]
        )
        det_proj = SimpleProj(1024, 768, relu)
        self.image_adapter = nn.ModuleDict(
            {
                "layer_adapters": layer_adapters,
                "seg_proj": seg_proj,
                "det_proj": det_proj,
            }
        )

        # Text adapters (same as original)
        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(768, 768) for _ in range(text_adapt_until)]
            + [SimpleProj(768, 768, relu=True)]
        )

        # Stage 1: Cross-attention for text-to-image attention
        self.text_cross_attn = CrossAttentionModule(
            dim=768,
            num_heads=cross_attn_heads,
            dropout=cross_attn_dropout
        )

        # Stage 2: Cross-attention for image-to-text attention
        self.image_cross_attn = ImageCrossAttentionModule(
            image_dim=1024,
            text_dim=768,
            num_heads=cross_attn_heads,
            dropout=cross_attn_dropout
        )

        # Storage for cross-attention contexts
        self.current_image_context = None  # For Stage 1: [B, 1369, 768]
        self.current_text_context = None   # For Stage 2: [B, L_text, 768]
        self.text_context_dict = None      # Pre-computed text contexts per class

        # Debug flags
        self.debug_cross_attn = False

        # Initialize weights
        self._init_weights_()

    def _init_weights_(self):
        """Initialize adapter weights with Xavier uniform"""
        for p in self.image_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ========================================
    # Stage 1: Text Cross-Attention Methods
    # ========================================

    def set_image_context(self, image_context, debug=False):
        """
        Set the image context for Stage 1 text cross-attention.

        Args:
            image_context: Image patches with CLS_24 residual [B, 1369, 768]
            debug: If True, print debug info
        """
        self.current_image_context = image_context
        if debug:
            print(f"\n[SET IMAGE CONTEXT for Stage 1]")
            print(f"  shape: {image_context.shape}")

    def clear_image_context(self):
        """Clear the image context after Stage 1 processing"""
        self.current_image_context = None

    # ========================================
    # Stage 2: Image Cross-Attention Methods
    # ========================================

    def load_text_context_dict(self, text_context_dict, debug=False):
        """
        Load pre-computed text context dictionary for Stage 2.

        Args:
            text_context_dict: Dict mapping class_name → [16, 77, 768]
            debug: If True, print debug info
        """
        self.text_context_dict = text_context_dict
        if debug:
            print(f"\n[LOADED TEXT CONTEXT DICT for Stage 2]")
            print(f"  Number of classes: {len(text_context_dict)}")
            for class_name, features in text_context_dict.items():
                print(f"    {class_name}: {features.shape}")

    def prepare_text_context_for_batch(self, class_names, device, debug=False):
        """
        Prepare text context for a batch of images (Stage 2).

        Args:
            class_names: List of class names for each image in batch
            device: torch device
            debug: If True, print debug info

        Returns:
            text_context: [B, L_text, 768] where L_text = 16*77 = 1232
        """
        if self.text_context_dict is None:
            raise ValueError("text_context_dict not loaded! Call load_text_context_dict first.")

        if debug:
            print(f"\n[PREPARE TEXT CONTEXT for Stage 2]")
            print(f"  class_names: {class_names}")

        text_contexts = []
        for class_name in class_names:
            class_text_context = self.text_context_dict[class_name].to(device)
            # Flatten: [16, 77, 768] → [1232, 768]
            # Use reshape() instead of view() to handle non-contiguous tensors
            flattened = class_text_context.reshape(-1, 768)
            text_contexts.append(flattened)

        # Stack: [B, 1232, 768]
        text_context = torch.stack(text_contexts, dim=0)

        # FIX: Normalize text_context to prevent huge cross-attention output
        # This matches the normalization done for image_context in Stage 1
        # Without this, text_context has norm ~17 per token, causing
        # cross-attention output to be 25000x larger than in Stage 1
        text_context = F.normalize(text_context, dim=-1)

        if debug:
            print(f"  text_context shape: {text_context.shape}")
            print(f"  text_context norm (per token): {text_context.norm(dim=-1).mean().item():.4f}")

        self.current_text_context = text_context
        return text_context

    def clear_text_context(self):
        """Clear the text context after Stage 2 processing"""
        self.current_text_context = None

    # ========================================
    # Text Encoding (with Stage 1 cross-attention)
    # ========================================

    def encode_text(self, text, adapt_text=True, debug=False):
        """
        Encode text with cross-attention to image features (Stage 1).

        Cross-attention is applied after layer 0 if current_image_context is set.

        Args:
            text: Tokenized text [B, 77]
            adapt_text: Whether to apply adapters
            debug: Whether to print debug information

        Returns:
            Text embeddings [B, 768]
        """
        if not adapt_text:
            return self.clipmodel.encode_text(text)

        if debug or self.debug_cross_attn:
            print("\n" + "="*70)
            print("ENCODE_TEXT (Stage 1 with Cross-Attention)")
            print("="*70)
            print(f"[INPUT] text shape: {text.shape}")
            print(f"[CONFIG] cross-attention enabled: {self.current_image_context is not None}")

        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(cast_dtype)
        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND: [77, B, 768]

        for i in range(12):
            # Frozen transformer layer
            x, attn = self.clipmodel.transformer.resblocks[i](
                x, attn_mask=self.clipmodel.attn_mask
            )

            # Stage 1 Cross-Attention: After layer 0, before adapter 0
            if i == 0 and self.current_image_context is not None:
                if debug or self.debug_cross_attn:
                    print(f"\n>>> STAGE 1 CROSS-ATTENTION (text→image) <<<")
                    print(f"    Scaling factor: {self.t_ca_w}")

                x_attn = x.permute(1, 0, 2)  # [B, 77, 768]
                cross_out = self.text_cross_attn(
                    x_attn,
                    self.current_image_context,
                    debug=(debug or self.debug_cross_attn)
                )

                # FIX: Norm-match cross-attention output to input (like adapters do)
                # This ensures consistent magnitude across both stages
                cross_out_norm = cross_out.norm(dim=-1, keepdim=True)
                x_attn_norm = x_attn.norm(dim=-1, keepdim=True)
                cross_out_normalized = cross_out * x_attn_norm / (cross_out_norm + 1e-6)

                # Apply residual with scaling weight
                cross_out_scaled = self.t_ca_w * cross_out_normalized
                x = x + cross_out_scaled.permute(1, 0, 2)  # Scaled residual

                if debug or self.debug_cross_attn:
                    print(f"    cross_out norm before: {cross_out.norm(dim=-1).mean().item():.4f}")
                    print(f"    cross_out norm after: {cross_out_normalized.norm(dim=-1).mean().item():.4f}")
                    print(f">>> CROSS-ATTENTION COMPLETE <<<\n")

            # Apply text adapter
            if i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x)
                adapt_out = adapt_out * x.norm(dim=-1, keepdim=True) / adapt_out.norm(dim=-1, keepdim=True)
                x = self.t_w * adapt_out + (1 - self.t_w) * x

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clipmodel.ln_final(x)
        eos_indices = text.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eos_indices]
        x = self.text_adapter[-1](x)

        if debug or self.debug_cross_attn:
            print(f"[OUTPUT] shape: {x.shape}")
            print("="*70 + "\n")

        return x

    def encode_text_intermediate(self, text, debug=False):
        """
        Encode text and return intermediate features after layer 0 + adapter[0].

        This is used to generate text_context_dict for Stage 2.

        Args:
            text: Tokenized text [B, 77]
            debug: If True, print debug info

        Returns:
            intermediate: [B, 77, 768] - features after layer 0 + adapter[0]
        """
        if debug:
            print(f"\n[ENCODE_TEXT_INTERMEDIATE]")
            print(f"  Input: {text.shape}")

        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(cast_dtype)
        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # [77, B, 768]

        # Only process layer 0 + adapter[0]
        x, attn = self.clipmodel.transformer.resblocks[0](
            x, attn_mask=self.clipmodel.attn_mask
        )

        # Apply adapter 0
        adapt_out = self.text_adapter[0](x)
        adapt_out = adapt_out * x.norm(dim=-1, keepdim=True) / adapt_out.norm(dim=-1, keepdim=True)
        x = self.t_w * adapt_out + (1 - self.t_w) * x

        x = x.permute(1, 0, 2)  # [B, 77, 768]

        if debug:
            print(f"  Output: {x.shape}")

        return x

    # ========================================
    # Image Encoding (with Stage 2 cross-attention)
    # ========================================

    def forward(self, x, use_cross_attention=False, debug=False):
        """
        Forward pass for image encoding with adapters.

        If use_cross_attention=True and current_text_context is set,
        applies cross-attention after ViT layer 0.

        Args:
            x: Input images [B, 3, H, W]
            use_cross_attention: Whether to apply Stage 2 cross-attention
            debug: Whether to print debug information

        Returns:
            seg_tokens: List of 4 tensors, each [B, 1369, 768]
            det_token: [B, 768]
        """
        if debug or self.debug_cross_attn:
            print("\n" + "="*70)
            print("IMAGE FORWARD (Stage 2 with Cross-Attention)")
            print("="*70)
            print(f"[INPUT] x shape: {x.shape}")
            print(f"[CONFIG] use_cross_attention: {use_cross_attention}")
            print(f"[CONFIG] text_context available: {self.current_text_context is not None}")

        # Patch embedding
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # Add CLS token + positional embedding
        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # [1370, B, 1024]

        tokens = []
        for i in range(24):
            # Frozen ViT layer
            x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)

            # Stage 2 Cross-Attention: After layer 0, before adapter 0
            if i == 0 and use_cross_attention and self.current_text_context is not None:
                if debug or self.debug_cross_attn:
                    print(f"\n>>> STAGE 2 CROSS-ATTENTION (image→text) <<<")
                    print(f"    Scaling factor: {self.i_ca_w}")

                x_attn = x.permute(1, 0, 2)  # [B, 1370, 1024]
                cross_out = self.image_cross_attn(
                    x_attn,
                    self.current_text_context,
                    debug=(debug or self.debug_cross_attn)
                )

                # FIX: Norm-match cross-attention output to input (like adapters do)
                # This ensures the cross-attention contribution has similar magnitude
                # to the input features, preventing training instability
                cross_out_norm = cross_out.norm(dim=-1, keepdim=True)
                x_attn_norm = x_attn.norm(dim=-1, keepdim=True)
                cross_out_normalized = cross_out * x_attn_norm / (cross_out_norm + 1e-6)

                # Apply residual with scaling weight
                cross_out_scaled = self.i_ca_w * cross_out_normalized
                x = x + cross_out_scaled.permute(1, 0, 2)  # Scaled residual

                if debug or self.debug_cross_attn:
                    print(f"    cross_out norm before: {cross_out.norm(dim=-1).mean().item():.4f}")
                    print(f"    cross_out norm after: {cross_out_normalized.norm(dim=-1).mean().item():.4f}")
                    print(f">>> CROSS-ATTENTION COMPLETE <<<\n")

            # Apply image adapter
            if i < self.image_adapt_until:
                adapt_out = self.image_adapter["layer_adapters"][i](x)
                adapt_out = adapt_out * x.norm(dim=-1, keepdim=True) / adapt_out.norm(dim=-1, keepdim=True)
                x = self.i_w * adapt_out + (1 - self.i_w) * x

            # Save features at specified layers
            if i + 1 in self.levels:
                tokens.append(x[1:, :, :])  # Remove CLS

        # Post-processing
        x = x.permute(1, 0, 2)
        tokens = [t.permute(1, 0, 2) for t in tokens]
        tokens = [self.image_encoder.ln_post(t) for t in tokens]

        seg_tokens = [self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)]
        seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens]

        det_token = self.image_adapter["det_proj"](tokens[-1])
        det_token = F.normalize(det_token, dim=-1).mean(1)

        if debug or self.debug_cross_attn:
            print(f"[OUTPUT] seg_tokens: {[t.shape for t in seg_tokens]}")
            print(f"[OUTPUT] det_token: {det_token.shape}")
            print("="*70 + "\n")

        return seg_tokens, det_token

    def forward_original(self, x, modality="visual"):
        """Original forward pass (unchanged from AdaptedCLIP)"""
        if modality == "visual":
            cls_features, patch_features = self.clipmodel.encode_image(x, [24])
            patch_features = [self.clipmodel.visual._global_pool(t)[1] for t in patch_features]
            patch_features = [self.clipmodel.visual.ln_post(t) for t in patch_features]
            patch_features = [t @ self.clipmodel.visual.proj for t in patch_features]
            return patch_features, cls_features
        else:
            raise ValueError("modality must be visual")
