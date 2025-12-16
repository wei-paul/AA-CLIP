"""
Forward Utilities for Cross-Attention Enhanced AA-CLIP

This module provides utility functions for generating text embeddings
when using the cross-attention mechanism. The main difference from
the original forward_utils.py is that text embeddings are generated
with awareness of the image context through cross-attention.

Key Functions:
- get_adapted_single_class_text_embedding_with_cross_attention:
  Generates text embeddings for a single class with cross-attention
- get_adapted_text_embedding_with_cross_attention:
  Generates text embeddings for all classes with cross-attention

Note: During inference (Stage 2), the original get_adapted_text_embedding
can still be used since cross-attention is not applied when
current_image_context is None.

Author: Cross-attention implementation for AA-CLIP
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataset.constants import CLASS_NAMES, REAL_NAMES, PROMPTS
from model.tokenizer import tokenize


# ================================================================================================
# Prompt configuration (same as original)
prompt = PROMPTS
prompt_normal = prompt["prompt_normal"]
prompt_abnormal = prompt["prompt_abnormal"]
prompt_state = [prompt_normal, prompt_abnormal]
prompt_templates = prompt["prompt_templates"]


def get_adapted_single_class_text_embedding_with_cross_attention(
    model, dataset_name, class_name, device, debug=False
):
    """
    Generate text embeddings for a single class with cross-attention.

    This function generates the normal and abnormal text embeddings for
    a given class. When the model has current_image_context set, the
    text encoding will use cross-attention to create vision-aware embeddings.

    Process:
    1. Generate all prompts for the class (6 normal + 10 abnormal = 16 total)
    2. Tokenize prompts → [16, 77]
    3. Encode through model.encode_text() (with cross-attention if image_context set)
    4. Normalize embeddings
    5. Average per category (normal, abnormal)
    6. Stack → [768, 2]

    Args:
        model: AdaptedCLIPWithCrossAttention model
        dataset_name: Name of dataset (MVTec, VisA, etc.)
        class_name: Name of class (bottle, cable, etc.)
        device: torch device
        debug: If True, print debug information

    Returns:
        text_features: [768, 2] - stacked normal and abnormal embeddings
            text_features[:, 0] = normal embedding [768]
            text_features[:, 1] = abnormal embedding [768]
    """
    if debug:
        print(f"\n{'='*60}")
        print(f"GENERATING TEXT EMBEDDING FOR CLASS: {class_name}")
        print(f"{'='*60}")

    # Get the real name for the class
    if class_name == "object":
        real_name = class_name
    else:
        assert class_name in CLASS_NAMES[dataset_name], (
            f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
        )
        real_name = REAL_NAMES[dataset_name][class_name]

    if debug:
        print(f"  Class name: {class_name}")
        print(f"  Real name: {real_name}")
        print(f"  Cross-attention enabled: {model.current_image_context is not None}")

    text_features = []

    # Process normal (i=0) and abnormal (i=1) prompts
    for i in range(len(prompt_state)):
        # Generate prompts for this state
        prompted_state = [state.format(real_name) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))

        if debug:
            state_name = "Normal" if i == 0 else "Abnormal"
            print(f"\n  [{state_name} Prompts]")
            print(f"    Number of prompts: {len(prompted_sentence)}")
            print(f"    Example prompts:")
            for j, p in enumerate(prompted_sentence[:3]):
                print(f"      {j}: {p}")
            if len(prompted_sentence) > 3:
                print(f"      ... and {len(prompted_sentence) - 3} more")

        # Tokenize prompts
        prompted_sentence = tokenize(prompted_sentence).to(device)
        # prompted_sentence: [num_prompts, 77]

        if debug:
            print(f"    Tokenized shape: {prompted_sentence.shape}")

        # Encode text (WITH cross-attention if image_context is set)
        class_embeddings = model.encode_text(prompted_sentence)
        # class_embeddings: [num_prompts, 768]

        if debug:
            print(f"    Embeddings shape: {class_embeddings.shape}")
            print(f"    Embeddings norm (before normalize): {class_embeddings.norm(dim=-1).mean().item():.4f}")

        # Normalize each embedding
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

        if debug:
            print(f"    Embeddings norm (after normalize): {class_embeddings.norm(dim=-1).mean().item():.4f}")

        # Average across all prompts for this category
        class_embedding = class_embeddings.mean(dim=0)
        # class_embedding: [768]

        if debug:
            print(f"    Mean embedding shape: {class_embedding.shape}")
            print(f"    Mean embedding norm (before final normalize): {class_embedding.norm().item():.4f}")

        # Final normalization
        class_embedding = class_embedding / class_embedding.norm()

        if debug:
            print(f"    Mean embedding norm (after final normalize): {class_embedding.norm().item():.4f}")

        text_features.append(class_embedding)

    # Stack normal and abnormal embeddings
    text_features = torch.stack(text_features, dim=1).to(device)
    # text_features: [768, 2]

    if debug:
        print(f"\n  [Final Output]")
        print(f"    text_features shape: {text_features.shape}")
        print(f"    Normal embedding norm: {text_features[:, 0].norm().item():.4f}")
        print(f"    Abnormal embedding norm: {text_features[:, 1].norm().item():.4f}")
        # Check orthogonality
        dot_product = (text_features[:, 0] * text_features[:, 1]).sum()
        print(f"    Normal-Abnormal dot product: {dot_product.item():.6f}")
        print(f"{'='*60}\n")

    return text_features


def get_adapted_text_embedding_with_cross_attention(model, dataset_name, device, debug=False):
    """
    Generate text embeddings for all classes with cross-attention.

    This function generates text embeddings for all classes in the dataset.
    When the model has current_image_context set, cross-attention will be
    applied during encoding.

    Args:
        model: AdaptedCLIPWithCrossAttention model
        dataset_name: Name of dataset (MVTec, VisA, etc.)
        device: torch device
        debug: If True, print debug information

    Returns:
        ret_dict: Dictionary mapping class_name → [768, 2] tensor
            Example: {"bottle": [768, 2], "cable": [768, 2], ...}
    """
    if debug:
        print(f"\n{'='*70}")
        print(f"GENERATING TEXT EMBEDDINGS FOR ALL CLASSES")
        print(f"Dataset: {dataset_name}")
        print(f"Classes: {CLASS_NAMES[dataset_name]}")
        print(f"{'='*70}")

    ret_dict = {}
    for class_name in CLASS_NAMES[dataset_name]:
        text_features = get_adapted_single_class_text_embedding_with_cross_attention(
            model, dataset_name, class_name, device, debug=debug
        )
        ret_dict[class_name] = text_features

    if debug:
        print(f"\n[SUMMARY]")
        print(f"  Generated embeddings for {len(ret_dict)} classes")
        for class_name, features in ret_dict.items():
            print(f"    {class_name}: {features.shape}")

    return ret_dict


def get_adapted_single_sentence_text_embedding_with_cross_attention(
    model, dataset_name, class_name, device, debug=False
):
    """
    Generate individual sentence embeddings (not averaged) with cross-attention.

    This is useful for analysis/visualization to see how different prompts
    produce different embeddings.

    Args:
        model: AdaptedCLIPWithCrossAttention model
        dataset_name: Name of dataset
        class_name: Name of class
        device: torch device
        debug: If True, print debug information

    Returns:
        text_features: [num_prompts_normal + num_prompts_abnormal, 768]
            All individual prompt embeddings concatenated
    """
    assert class_name in CLASS_NAMES[dataset_name], (
        f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
    )
    real_name = REAL_NAMES[dataset_name][class_name]

    if debug:
        print(f"\n[Generating sentence embeddings for {class_name}]")
        print(f"  Real name: {real_name}")

    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(real_name) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence).to(device)

        # Encode with cross-attention
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        text_features.append(class_embeddings)

    text_features = torch.cat(text_features, dim=0).to(device)

    if debug:
        print(f"  Output shape: {text_features.shape}")
        print(f"  (First {len(prompt_state[0]) * len(prompt_templates)} are normal prompts)")

    return text_features


# ================================================================================================
# Debug utility functions

def analyze_cross_attention_effect(
    model, clip_surgery, image, dataset_name, class_name, device
):
    """
    Analyze the effect of cross-attention on text embeddings.

    Compares text embeddings generated:
    1. WITHOUT cross-attention (no image context)
    2. WITH cross-attention (image context from the given image)

    This helps understand how much the text embeddings change when
    attending to different images.

    Args:
        model: AdaptedCLIPWithCrossAttention model
        clip_surgery: CLIP model for feature extraction
        image: Input image [1, 3, H, W]
        dataset_name: Name of dataset
        class_name: Name of class
        device: torch device

    Returns:
        dict with analysis results
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING CROSS-ATTENTION EFFECT")
    print(f"Class: {class_name}")
    print(f"{'='*70}")

    # Generate embeddings WITHOUT cross-attention
    model.clear_image_context()
    with torch.no_grad():
        emb_without = get_adapted_single_class_text_embedding_with_cross_attention(
            model, dataset_name, class_name, device
        )

    # Compute image context
    from train_cross_attention import compute_image_context
    with torch.no_grad():
        image_context = compute_image_context(image, clip_surgery, model, device)

    # Generate embeddings WITH cross-attention
    model.set_image_context(image_context)
    with torch.no_grad():
        emb_with = get_adapted_single_class_text_embedding_with_cross_attention(
            model, dataset_name, class_name, device
        )
    model.clear_image_context()

    # Analyze differences
    normal_diff = (emb_with[:, 0] - emb_without[:, 0]).norm().item()
    abnormal_diff = (emb_with[:, 1] - emb_without[:, 1]).norm().item()

    normal_cos_sim = F.cosine_similarity(
        emb_with[:, 0].unsqueeze(0), emb_without[:, 0].unsqueeze(0)
    ).item()
    abnormal_cos_sim = F.cosine_similarity(
        emb_with[:, 1].unsqueeze(0), emb_without[:, 1].unsqueeze(0)
    ).item()

    print(f"\n[Normal Embedding]")
    print(f"  L2 difference: {normal_diff:.6f}")
    print(f"  Cosine similarity: {normal_cos_sim:.6f}")

    print(f"\n[Abnormal Embedding]")
    print(f"  L2 difference: {abnormal_diff:.6f}")
    print(f"  Cosine similarity: {abnormal_cos_sim:.6f}")

    # Orthogonality analysis
    ortho_without = (emb_without[:, 0] * emb_without[:, 1]).sum().item()
    ortho_with = (emb_with[:, 0] * emb_with[:, 1]).sum().item()

    print(f"\n[Orthogonality (Normal·Abnormal)]")
    print(f"  Without cross-attention: {ortho_without:.6f}")
    print(f"  With cross-attention: {ortho_with:.6f}")

    return {
        "emb_without": emb_without,
        "emb_with": emb_with,
        "normal_diff": normal_diff,
        "abnormal_diff": abnormal_diff,
        "normal_cos_sim": normal_cos_sim,
        "abnormal_cos_sim": abnormal_cos_sim,
        "ortho_without": ortho_without,
        "ortho_with": ortho_with,
    }


def visualize_attention_weights(model, text_tokens, image_context, device):
    """
    Extract and visualize attention weights from cross-attention.

    This function extracts the attention weights to see which image patches
    each text token attends to most strongly.

    Args:
        model: AdaptedCLIPWithCrossAttention model
        text_tokens: Tokenized text [B, 77]
        image_context: Image patches [B, 1369, 768]
        device: torch device

    Returns:
        attention_weights: [B, num_heads, 77, 1369]
    """
    print("\n[Extracting attention weights for visualization]")

    # This would require modifying CrossAttentionModule to return attention weights
    # For now, this is a placeholder that shows the concept

    model.set_image_context(image_context)

    # We'd need to modify the forward pass to return attention weights
    # This is left as a future enhancement

    print("  Note: Full attention visualization requires modifying CrossAttentionModule")
    print("  to return attention weights. See adapter_cross_attention.py for details.")

    model.clear_image_context()

    return None
