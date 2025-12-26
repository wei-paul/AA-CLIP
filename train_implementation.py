"""
Unified Training Script for AA-CLIP

This script consolidates ALL training modes into a single entry point:
  - Base model training (original AA-CLIP without cross-attention)
  - Cross-attention training (single run, no feedback loops)
  - Cross-attention with feedback loops (with automatic plateau detection)
  - Freeze adapters + train cross-attention only (Variant A)
  - Freeze adapters[1:] + train adapter[0] + cross-attention (Variant B)

Usage Examples:
    # Base model (no cross-attention)
    python train_implementation.py --mode base --dataset VisA --save_path ckpt/base

    # Cross-attention (single run)
    python train_implementation.py --mode cross_attention --dataset VisA --save_path ckpt/cross_attn

    # Cross-attention with feedback loops (automatic plateau detection)
    python train_implementation.py --mode feedback --dataset VisA --save_path ckpt/feedback \
        --num_feedback_loops 10 \
        --plateau_threshold 0.01 \
        --plateau_patience 2

    # Variant A: Freeze adapters, train only cross-attention
    python train_implementation.py --mode freeze_adapters --dataset VisA --save_path ckpt/freeze_adapt \
        --load_base_text_adapter ckpt/base/text_adapter.pth \
        --load_base_image_adapter ckpt/base/image_adapter.pth

    # Variant B: Freeze adapters[1:], train adapter[0] + cross-attention
    python train_implementation.py --mode variant_b --dataset VisA --save_path ckpt/variant_b \
        --load_base_text_adapter ckpt/base/text_adapter.pth \
        --load_base_image_adapter ckpt/base/image_adapter.pth

Feedback Loop Parameters:
    --num_feedback_loops: Maximum number of loops (default: 10)
    --plateau_threshold: Stop if relative loss improvement < threshold (default: 0.01 = 1%)
    --plateau_patience: Stop after N consecutive loops below threshold (default: 2)

    The feedback loop will automatically stop early when the combined loss
    (text + image) improvement falls below the threshold for consecutive loops.

Author: AA-CLIP unified training implementation
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.adapter_cross_attention import AdaptedCLIPWithCrossAttention
from model.clip import create_model
from dataset import get_dataset
from dataset.constants import CLASS_NAMES, REAL_NAMES, PROMPTS
from model.tokenizer import tokenize
from forward_utils import (
    get_adapted_text_embedding,
    get_adapted_single_class_text_embedding,
    calculate_similarity_map,
    calculate_seg_loss,
)
from forward_utils_cross_attention import (
    get_adapted_single_class_text_embedding_with_cross_attention,
)
import warnings

warnings.filterwarnings("ignore")

cpu_num = 4
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Prompt configuration
prompt = PROMPTS
prompt_normal = prompt["prompt_normal"]
prompt_abnormal = prompt["prompt_abnormal"]
prompt_state = [prompt_normal, prompt_abnormal]
prompt_templates = prompt["prompt_templates"]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """Creates a scheduler with linear warmup followed by cosine decay."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def compute_gradient_norm(model_parameters):
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model_parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


# ============================================================================
# BASE MODEL TRAINING (No Cross-Attention)
# ============================================================================

def train_text_adapter_base(
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
    """Train text adapter (base model, no cross-attention)."""
    for epoch in range(start_epoch, text_epoch):
        logger.info(f"training text epoch {epoch}:")
        print(f"\n[BASE] Epoch {epoch}/{text_epoch-1} - Stage 1: Text Adapter")

        loss_list = []
        grad_norm_list = []
        for input_data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            class_names = input_data["class_name"]

            # forward text
            epoch_text_feature_dict = {}
            for class_name in list(set(class_names)):
                text_embedding = get_adapted_single_class_text_embedding(
                    adapted_model, dataset_name, class_name, device
                )
                epoch_text_feature_dict[class_name] = text_embedding
            epoch_text_feature = torch.stack(
                [epoch_text_feature_dict[class_name] for class_name in class_names],
                dim=0,
            )  # bs,768,2

            # forward image (frozen)
            with torch.no_grad():
                _, patch_features = clip_surgery.encode_image(image, [6, 12, 18, 24])
                cls_token, _ = adapted_model.clipmodel.encode_image(image, [])
                cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
                patch_features = [clip_surgery.visual.ln_post(t[:, 1:, :]) for t in patch_features]
                patch_features = [t @ clip_surgery.visual.proj for t in patch_features]
                patch_features = [t / t.norm(dim=-1, keepdim=True) for t in patch_features]
                patch_features = [t + cls_token.unsqueeze(1) for t in patch_features]

            # calculate similarity and get prediction
            # FIX: Accumulate loss across all 4 feature scales (was overwriting instead of accumulating)
            # This ensures text embeddings are optimized for all scales [6, 12, 18, 24], not just layer 24
            loss = 0.0
            for f in patch_features:
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                loss += calculate_seg_loss(patch_preds, mask)

            # Orthogonal loss: computed once outside loop (doesn't depend on patch features)
            orthogonal_loss = (
                (epoch_text_feature[:, :, 0] * epoch_text_feature[:, :, 1]).sum(1).mean()
            ) ** 2
            loss += orthogonal_loss * text_norm_weight

            # backward
            optimizer.zero_grad()
            loss.backward()

            # Compute gradient magnitude
            grad_norm = compute_gradient_norm(model.text_adapter.parameters())

            optimizer.step()
            loss_list.append(loss.item())
            grad_norm_list.append(grad_norm)

            # Log gradient magnitude every 50 batches
            batch_idx = len(loss_list)
            if batch_idx % 50 == 0:
                print(f"  [Batch {batch_idx}] Loss: {loss.item():.6f}, GRADIENT MAGNITUDE: {grad_norm:.6f}")

        logger.info(f"loss: {np.mean(loss_list)}")
        print(f"  Average Loss: {np.mean(loss_list):.6f}")
        print(f"  Gradient Stats: mean={np.mean(grad_norm_list):.6f}, max={np.max(grad_norm_list):.6f}, min={np.min(grad_norm_list):.6f}")

        # save checkpoint
        ckp_path = os.path.join(save_path, "text_adapter.pth")
        torch.save({
            "epoch": epoch + 1,
            "text_adapter": adapted_model.text_adapter.state_dict(),
            "text_optimizer": optimizer.state_dict(),
        }, ckp_path)

    return adapted_model


def train_image_adapter_base(
    model: nn.Module,
    text_embeddings: torch.Tensor,
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
    """Train image adapter (base model, no cross-attention)."""
    for epoch in range(start_epoch, image_epoch):
        logger.info(f"training image epoch {epoch}:")
        print(f"\n[BASE] Epoch {epoch}/{image_epoch-1} - Stage 2: Image Adapter")

        loss_list = []
        grad_norm_list = []
        for input_data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            label = input_data["label"].to(device)
            B, C, H, W = image.shape

            # forward text
            class_names = input_data["class_name"]
            epoch_text_feature = torch.stack(
                [text_embeddings[class_name] for class_name in class_names], dim=0
            )

            # forward image
            patch_features, det_feature = model(image)

            # calculate similarity and get prediction
            loss = 0.0
            det_feature = det_feature.unsqueeze(1)
            cls_preds = torch.matmul(det_feature, epoch_text_feature)[:, 0]
            loss += F.cross_entropy(cls_preds, label)

            for f in patch_features:
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                loss += calculate_seg_loss(patch_preds, mask)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # Compute gradient magnitude
            grad_norm = compute_gradient_norm(model.image_adapter.parameters())

            optimizer.step()
            loss_list.append(loss.item())
            grad_norm_list.append(grad_norm)

            # Log gradient magnitude every 50 batches
            batch_idx = len(loss_list)
            if batch_idx % 50 == 0:
                print(f"  [Batch {batch_idx}] Loss: {loss.item():.6f}, GRADIENT MAGNITUDE: {grad_norm:.6f}")

            scheduler.step()

        logger.info(f"loss: {np.mean(loss_list)}")
        print(f"  Average Loss: {np.mean(loss_list):.6f}")
        print(f"  Gradient Stats: mean={np.mean(grad_norm_list):.6f}, max={np.max(grad_norm_list):.6f}, min={np.min(grad_norm_list):.6f}")

        # save checkpoint
        model_dict = {
            "epoch": epoch + 1,
            "image_adapter": model.image_adapter.state_dict(),
            "image_optimizer": optimizer.state_dict(),
        }
        torch.save(model_dict, os.path.join(save_path, "image_adapter.pth"))
        if (epoch + 1) % 1 == 0:
            ckp_path = os.path.join(save_path, f"image_adapter_{epoch + 1}.pth")
            torch.save(model_dict, ckp_path)

    return model


# ============================================================================
# CROSS-ATTENTION IMAGE CONTEXT COMPUTATION
# ============================================================================

def compute_image_context_frozen(image, clip_surgery, adapted_model, device, debug=False):
    """
    Compute image context for Stage 1 cross-attention using FROZEN clip_surgery.
    Returns V1 patches (layer 6) + CLS_24 residual = [B, 1369, 768]
    """
    with torch.no_grad():
        if debug:
            print(f"\n[IMAGE CONTEXT] Computing from FROZEN clip_surgery")

        # Get V1 (layer 6) features
        _, patch_features = clip_surgery.encode_image(image, [6])
        v1_features = patch_features[0]  # [B, 1370, 1024]

        # Remove CLS token
        v1_patches = v1_features[:, 1:, :]  # [B, 1369, 1024]

        # Layer norm + projection
        v1_patches = clip_surgery.visual.ln_post(v1_patches)
        v1_patches = v1_patches @ clip_surgery.visual.proj  # [B, 1369, 768]

        # Normalize
        v1_patches = v1_patches / v1_patches.norm(dim=-1, keepdim=True)

        # Get CLS_24
        cls_token_24, _ = adapted_model.clipmodel.encode_image(image, [])
        cls_token_24 = cls_token_24 / cls_token_24.norm(dim=-1, keepdim=True)

        # Add residual
        image_context = v1_patches + cls_token_24.unsqueeze(1)  # [B, 1369, 768]

        if debug:
            print(f"  Output shape: {image_context.shape}")

    return image_context


def compute_image_context_trained(image, adapted_model, device, debug=False):
    """
    Compute image context using TRAINED image_adapter (for feedback loops 2+).
    Returns V1 patches (layer 6 through trained adapters) + CLS_24 residual = [B, 1369, 768]
    """
    with torch.no_grad():
        if debug:
            print(f"\n[IMAGE CONTEXT] Computing from TRAINED image_adapter")

        # Forward through ViT with trained adapters up to layer 6
        x = adapted_model.image_encoder.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # [B, 1369, 1024]

        # Add CLS token + positional embedding
        x = torch.cat([
            adapted_model.image_encoder.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)  # [B, 1370, 1024]
        x = x + adapted_model.image_encoder.positional_embedding.to(x.dtype)
        x = adapted_model.image_encoder.patch_dropout(x)
        x = adapted_model.image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # [1370, B, 1024]

        # Forward through layers 0-5 WITH trained adapters
        for i in range(6):
            x, _ = adapted_model.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            if i < adapted_model.image_adapt_until:
                adapt_out = adapted_model.image_adapter["layer_adapters"][i](x)
                adapt_out = (
                    adapt_out * x.norm(dim=-1, keepdim=True) /
                    adapt_out.norm(dim=-1, keepdim=True)
                )
                x = adapted_model.i_w * adapt_out + (1 - adapted_model.i_w) * x

        # Forward through layer 6 (no adapter)
        x, _ = adapted_model.image_encoder.transformer.resblocks[6](x, attn_mask=None)

        # Extract patches
        x = x.permute(1, 0, 2)  # [B, 1370, 1024]
        v1_patches = x[:, 1:, :]  # [B, 1369, 1024]

        # Apply layer norm and trained seg_proj[0]
        v1_patches = adapted_model.image_encoder.ln_post(v1_patches)
        v1_patches = adapted_model.image_adapter["seg_proj"][0](v1_patches)  # [B, 1369, 768]
        v1_patches = F.normalize(v1_patches, dim=-1)

        # Get CLS_24
        cls_token_24, _ = adapted_model.clipmodel.encode_image(image, [])
        cls_token_24 = cls_token_24 / cls_token_24.norm(dim=-1, keepdim=True)

        # Add residual
        image_context = v1_patches + cls_token_24.unsqueeze(1)  # [B, 1369, 768]

        if debug:
            print(f"  Output shape: {image_context.shape}")

    return image_context


def compute_patch_features_frozen(image, clip_surgery, adapted_model, device):
    """Compute multi-scale patch features using FROZEN clip_surgery."""
    with torch.no_grad():
        _, patch_features = clip_surgery.encode_image(image, [6, 12, 18, 24])
        cls_token, _ = adapted_model.clipmodel.encode_image(image, [])
        cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)

        patch_features = [clip_surgery.visual.ln_post(t[:, 1:, :]) for t in patch_features]
        patch_features = [t @ clip_surgery.visual.proj for t in patch_features]
        patch_features = [t / t.norm(dim=-1, keepdim=True) for t in patch_features]
        patch_features = [t + cls_token.unsqueeze(1) for t in patch_features]

    return patch_features


def compute_patch_features_trained(image, adapted_model, device):
    """Compute multi-scale patch features using TRAINED image_adapter."""
    with torch.no_grad():
        x = adapted_model.image_encoder.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat([
            adapted_model.image_encoder.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + adapted_model.image_encoder.positional_embedding.to(x.dtype)
        x = adapted_model.image_encoder.patch_dropout(x)
        x = adapted_model.image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)

        tokens = []
        levels = [6, 12, 18, 24]

        for i in range(24):
            x, _ = adapted_model.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            if i < adapted_model.image_adapt_until:
                adapt_out = adapted_model.image_adapter["layer_adapters"][i](x)
                adapt_out = (
                    adapt_out * x.norm(dim=-1, keepdim=True) /
                    adapt_out.norm(dim=-1, keepdim=True)
                )
                x = adapted_model.i_w * adapt_out + (1 - adapted_model.i_w) * x
            if (i + 1) in levels:
                tokens.append(x[1:, :, :])

        tokens = [t.permute(1, 0, 2) for t in tokens]
        tokens = [adapted_model.image_encoder.ln_post(t) for t in tokens]

        patch_features = []
        for i, t in enumerate(tokens):
            projected = adapted_model.image_adapter["seg_proj"][i](t)
            projected = F.normalize(projected, dim=-1)
            patch_features.append(projected)

        cls_token_24, _ = adapted_model.clipmodel.encode_image(image, [])
        cls_token_24 = cls_token_24 / cls_token_24.norm(dim=-1, keepdim=True)
        patch_features = [t + cls_token_24.unsqueeze(1) for t in patch_features]

    return patch_features


# ============================================================================
# CROSS-ATTENTION TRAINING (Single Run)
# ============================================================================

def train_text_adapter_cross_attention(
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
    freeze_adapter: bool = False,
    use_trained_features: bool = False,
    loop_iteration: int = 1,
    grad_clip: float = 0.0,
    debug_frequency: int = 100,
):
    """
    Train text adapter with cross-attention to image features (Stage 1).

    Args:
        freeze_adapter: If True, freeze text_adapter and only train text_cross_attn
        use_trained_features: If True, use trained image_adapter for image context
        loop_iteration: Feedback loop number (for logging)
        grad_clip: Gradient clipping threshold (0 = disabled)
    """
    batch_count = 0

    for epoch in range(start_epoch, text_epoch):
        logger.info(f"Loop {loop_iteration} - training text epoch {epoch} (freeze_adapter={freeze_adapter}):")
        print(f"\n[CROSS-ATTN] Loop {loop_iteration} Epoch {epoch}/{text_epoch-1} - Stage 1: Text")
        if freeze_adapter:
            print("  Mode: FREEZE adapter, train cross-attention ONLY")

        loss_list = []
        grad_norm_list = []

        for input_data in tqdm(train_loader, desc=f"Loop {loop_iteration} Epoch {epoch}"):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            class_names = input_data["class_name"]

            debug_this_batch = (batch_count % debug_frequency == 0)

            # Compute image context for cross-attention
            if use_trained_features:
                image_context = compute_image_context_trained(
                    image, adapted_model, device, debug=debug_this_batch
                )
            else:
                image_context = compute_image_context_frozen(
                    image, clip_surgery, adapted_model, device, debug=debug_this_batch
                )
            adapted_model.set_image_context(image_context, debug=debug_this_batch)

            # Generate text embeddings WITH cross-attention
            epoch_text_feature_dict = {}
            for class_name in list(set(class_names)):
                text_embedding = get_adapted_single_class_text_embedding_with_cross_attention(
                    adapted_model, dataset_name, class_name, device
                )
                epoch_text_feature_dict[class_name] = text_embedding

            epoch_text_feature = torch.stack(
                [epoch_text_feature_dict[class_name] for class_name in class_names],
                dim=0,
            )  # [B, 768, 2]

            # Compute patch features for loss
            if use_trained_features:
                patch_features = compute_patch_features_trained(image, adapted_model, device)
            else:
                patch_features = compute_patch_features_frozen(
                    image, clip_surgery, adapted_model, device
                )

            # Calculate loss
            # FIX: Accumulate loss across all 4 feature scales (was overwriting instead of accumulating)
            # This ensures text embeddings are optimized for all scales [6, 12, 18, 24], not just layer 24
            loss = 0.0
            for f in patch_features:
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                loss += calculate_seg_loss(patch_preds, mask)

            # Orthogonal loss: computed once outside loop (doesn't depend on patch features)
            orthogonal_loss = ((epoch_text_feature[:, :, 0] * epoch_text_feature[:, :, 1]).sum(1).mean()) ** 2
            loss += orthogonal_loss * text_norm_weight

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            # Compute gradient magnitude before clipping
            params_to_check = list(adapted_model.text_cross_attn.parameters())
            if not freeze_adapter:
                params_to_check += list(adapted_model.text_adapter.parameters())
            grad_norm = compute_gradient_norm(params_to_check)

            # Gradient clipping
            if grad_clip > 0:
                params_to_clip = list(adapted_model.text_cross_attn.parameters())
                if not freeze_adapter:
                    params_to_clip += list(adapted_model.text_adapter.parameters())
                torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip)

            optimizer.step()
            loss_list.append(loss.item())
            grad_norm_list.append(grad_norm)

            # Log gradient magnitude every 50 batches
            if batch_count % 50 == 0:
                print(f"  [Batch {batch_count}] Loss: {loss.item():.6f}, GRADIENT MAGNITUDE: {grad_norm:.6f}")

            adapted_model.clear_image_context()
            batch_count += 1

        # End of epoch
        avg_loss = np.mean(loss_list)
        logger.info(f"Loop {loop_iteration} - text loss: {avg_loss}")
        print(f"  Average Loss: {avg_loss:.6f}")
        print(f"  Gradient Stats: mean={np.mean(grad_norm_list):.6f}, max={np.max(grad_norm_list):.6f}, min={np.min(grad_norm_list):.6f}")

        # Save checkpoint
        ckp_path = os.path.join(save_path, "text_adapter_cross_attn.pth")
        torch.save({
            "epoch": epoch + 1,
            "loop": loop_iteration,
            "text_adapter": adapted_model.text_adapter.state_dict(),
            "text_cross_attn": adapted_model.text_cross_attn.state_dict(),
            "text_optimizer": optimizer.state_dict(),
        }, ckp_path)

    # Return model and final epoch's average loss
    return adapted_model, avg_loss


def train_image_adapter_cross_attention(
    model: nn.Module,
    text_embeddings: dict,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    start_epoch: int,
    save_path: str,
    image_epoch: int,
    img_size: int,
    logger: logging.Logger,
    freeze_adapter: bool = False,
    loop_iteration: int = 1,
    grad_clip: float = 0.0,
    debug_frequency: int = 100,
):
    """
    Train image adapter with cross-attention to text features (Stage 2).

    Args:
        freeze_adapter: If True, freeze image_adapter and only train image_cross_attn
        loop_iteration: Feedback loop number (for logging)
        grad_clip: Gradient clipping threshold (0 = disabled)
    """
    batch_count = 0

    for epoch in range(start_epoch, image_epoch):
        logger.info(f"Loop {loop_iteration} - training image epoch {epoch} (freeze_adapter={freeze_adapter}):")
        print(f"\n[CROSS-ATTN] Loop {loop_iteration} Epoch {epoch}/{image_epoch-1} - Stage 2: Image")
        if freeze_adapter:
            print("  Mode: FREEZE adapter, train cross-attention ONLY")

        loss_list = []
        grad_norm_list = []

        for input_data in tqdm(train_loader, desc=f"Loop {loop_iteration} Epoch {epoch}"):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            label = input_data["label"].to(device)
            class_names = input_data["class_name"]

            debug_this_batch = (batch_count % debug_frequency == 0)

            # Prepare text context for cross-attention (per-class)
            text_context = model.prepare_text_context_for_batch(
                class_names, device, debug=debug_this_batch
            )

            # Look up final text embeddings for loss
            epoch_text_feature = torch.stack(
                [text_embeddings[cn] for cn in class_names], dim=0
            ).to(device)  # [B, 768, 2]

            # Forward image WITH cross-attention
            patch_features, det_feature = model(
                image,
                use_cross_attention=True,
                debug=debug_this_batch
            )

            # Calculate loss
            loss = 0.0

            # Classification loss
            det_feature_exp = det_feature.unsqueeze(1)
            cls_preds = torch.matmul(det_feature_exp, epoch_text_feature)[:, 0]
            cls_loss = F.cross_entropy(cls_preds, label)
            loss += cls_loss

            # Segmentation loss
            for f in patch_features:
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                seg_loss = calculate_seg_loss(patch_preds, mask)
                loss += seg_loss

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            # Compute gradient magnitude before clipping
            params_to_check = list(model.image_cross_attn.parameters())
            if not freeze_adapter:
                params_to_check += list(model.image_adapter.parameters())
            grad_norm = compute_gradient_norm(params_to_check)

            # Gradient clipping
            if grad_clip > 0:
                params_to_clip = list(model.image_cross_attn.parameters())
                if not freeze_adapter:
                    params_to_clip += list(model.image_adapter.parameters())
                torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip)

            optimizer.step()
            loss_list.append(loss.item())
            grad_norm_list.append(grad_norm)

            # Log gradient magnitude every 50 batches
            if batch_count % 50 == 0:
                print(f"  [Batch {batch_count}] Loss: {loss.item():.6f}, GRADIENT MAGNITUDE: {grad_norm:.6f}")

            if scheduler is not None:
                scheduler.step()
            model.clear_text_context()
            batch_count += 1

        # End of epoch
        avg_loss = np.mean(loss_list)
        logger.info(f"Loop {loop_iteration} - image loss: {avg_loss}")
        print(f"  Average Loss: {avg_loss:.6f}")
        print(f"  Gradient Stats: mean={np.mean(grad_norm_list):.6f}, max={np.max(grad_norm_list):.6f}, min={np.min(grad_norm_list):.6f}")

        # Save checkpoint
        ckp_path = os.path.join(save_path, "image_adapter_cross_attn.pth")
        torch.save({
            "epoch": epoch + 1,
            "loop": loop_iteration,
            "image_adapter": model.image_adapter.state_dict(),
            "image_cross_attn": model.image_cross_attn.state_dict(),
            "image_optimizer": optimizer.state_dict(),
        }, ckp_path)

        # Epoch-specific checkpoint
        epoch_ckp = os.path.join(save_path, f"image_adapter_cross_attn_{epoch + 1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "loop": loop_iteration,
            "image_adapter": model.image_adapter.state_dict(),
            "image_cross_attn": model.image_cross_attn.state_dict(),
            "image_optimizer": optimizer.state_dict(),
        }, epoch_ckp)

    # Return model and final epoch's average loss
    return model, avg_loss


# ============================================================================
# TEXT CONTEXT GENERATION FOR STAGE 2
# ============================================================================

def generate_text_context_dict(model, dataset_name, device, debug=False):
    """Generate text context dictionary for Stage 2 cross-attention."""
    print("\n" + "="*70)
    print("GENERATING TEXT CONTEXT DICT FOR STAGE 2")
    print("="*70)

    text_context_dict = {}

    for class_name in CLASS_NAMES[dataset_name]:
        real_name = REAL_NAMES[dataset_name][class_name]

        # Generate all prompts
        all_prompts = []
        for i in range(len(prompt_state)):
            prompted_state = [state.format(real_name) for state in prompt_state[i]]
            for s in prompted_state:
                for template in prompt_templates:
                    all_prompts.append(template.format(s))

        # Tokenize
        tokens = tokenize(all_prompts).to(device)  # [16, 77]

        # Get intermediate features
        with torch.no_grad():
            intermediate = model.encode_text_intermediate(tokens, debug=debug)
            # [16, 77, 768]

        text_context_dict[class_name] = intermediate.cpu()
        print(f"  {class_name}: {intermediate.shape}")

    print(f"\nGenerated for {len(text_context_dict)} classes")
    return text_context_dict


# ============================================================================
# MAIN TRAINING MODES
# ============================================================================

def train_base_mode(args, logger, device, use_cuda):
    """Base model training (no cross-attention)."""
    print("\n" + "="*70)
    print("MODE: BASE (No Cross-Attention)")
    print("="*70)

    # Load models
    clip_surgery = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_surgery.eval()
    clip_surgery.visual.DAPM_replace(DPAM_layer=args.surgery_until_layer)

    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()

    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
    ).to(device)
    model.eval()

    # Optimizers
    text_optimizer = torch.optim.Adam(
        model.text_adapter.parameters(),
        lr=args.text_lr,
        betas=(0.5, 0.999),
    )
    image_optimizer = torch.optim.Adam(
        model.image_adapter.parameters(),
        lr=args.image_lr,
        betas=(0.5, 0.999),
    )
    image_scheduler = MultiStepLR(image_optimizer, milestones=[16000, 32000], gamma=0.5)

    # Load checkpoints if exist
    text_file = glob(args.save_path + "/text_adapter.pth")
    if len(text_file) > 0:
        checkpoint = torch.load(text_file[0])
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        text_optimizer.load_state_dict(checkpoint["text_optimizer"])
        text_start_epoch = checkpoint["epoch"]
        adapt_text = not (text_start_epoch == (args.text_epoch - 1))
    elif args.text_epoch == 0:
        adapt_text = False
        text_start_epoch = 0
    else:
        text_start_epoch = 0
        adapt_text = True

    file = glob(args.save_path + "/image_adapter.pth")
    if len(file) > 0:
        checkpoint = torch.load(file[0])
        image_start_epoch = checkpoint["epoch"]
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
        image_optimizer.load_state_dict(checkpoint["image_optimizer"])
    else:
        image_start_epoch = 0

    # Load dataset
    if args.training_mode == "full_shot":
        args.shot = -1
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    text_dataset, image_dataset = get_dataset(
        args.dataset, args.img_size, args.training_mode, args.shot, "train", logger
    )
    text_dataloader = torch.utils.data.DataLoader(
        text_dataset, batch_size=args.text_batch_size, shuffle=True, **kwargs
    )
    image_dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=args.image_batch_size, shuffle=True, **kwargs
    )

    # Stage 1: Text Adapter
    if adapt_text:
        model = train_text_adapter_base(
            adapted_model=model,
            clip_surgery=clip_surgery,
            text_norm_weight=args.text_norm_weight,
            train_loader=text_dataloader,
            optimizer=text_optimizer,
            device=device,
            start_epoch=text_start_epoch,
            dataset_name=args.dataset,
            save_path=args.save_path,
            text_epoch=args.text_epoch,
            img_size=args.img_size,
            logger=logger,
        )

    del text_dataloader, text_dataset, clip_surgery, text_optimizer
    torch.cuda.empty_cache()

    # Generate text embeddings
    with torch.no_grad():
        if args.text_epoch == 0:
            text_embeddings = get_adapted_text_embedding(clip_model, args.dataset, device)
        else:
            text_embeddings = get_adapted_text_embedding(model, args.dataset, device)

    # Stage 2: Image Adapter
    model = train_image_adapter_base(
        model=model,
        text_embeddings=text_embeddings,
        image_epoch=args.image_epoch,
        train_loader=image_dataloader,
        optimizer=image_optimizer,
        scheduler=image_scheduler,
        device=device,
        start_epoch=image_start_epoch,
        save_path=args.save_path,
        img_size=args.img_size,
        logger=logger,
    )

    print("\n[BASE] Training Complete!")


def train_cross_attention_mode(args, logger, device, use_cuda, freeze_adapters=False):
    """
    Cross-attention training (single run).

    Args:
        freeze_adapters: If True, freeze adapters and train only cross-attention modules
    """
    mode_name = "FREEZE_ADAPTERS" if freeze_adapters else "CROSS_ATTENTION"
    print("\n" + "="*70)
    print(f"MODE: {mode_name}")
    print("="*70)

    # Load models
    clip_surgery = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_surgery.eval()
    clip_surgery.visual.DAPM_replace(DPAM_layer=args.surgery_until_layer)

    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()

    model = AdaptedCLIPWithCrossAttention(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
        cross_attn_heads=args.cross_attn_heads,
        cross_attn_dropout=args.cross_attn_dropout,
    ).to(device)
    model.eval()

    # Load pre-trained adapters if specified
    if args.load_base_text_adapter is not None:
        print(f"\nLoading pre-trained text adapter: {args.load_base_text_adapter}")
        base_ckpt = torch.load(args.load_base_text_adapter, map_location=device)
        model.text_adapter.load_state_dict(base_ckpt["text_adapter"])
        logger.info(f"Loaded base text_adapter from: {args.load_base_text_adapter}")

    if args.load_base_image_adapter is not None:
        print(f"Loading pre-trained image adapter: {args.load_base_image_adapter}")
        base_ckpt = torch.load(args.load_base_image_adapter, map_location=device)
        model.image_adapter.load_state_dict(base_ckpt["image_adapter"])
        logger.info(f"Loaded base image_adapter from: {args.load_base_image_adapter}")

    # Freeze adapters if requested
    if freeze_adapters:
        print("\n[FREEZE] Freezing text_adapter and image_adapter")
        for param in model.text_adapter.parameters():
            param.requires_grad = False
        for param in model.image_adapter.parameters():
            param.requires_grad = False

        # Only train cross-attention modules
        text_params = list(model.text_cross_attn.parameters())
        image_params = list(model.image_cross_attn.parameters())

        print(f"  Trainable text params: {sum(p.numel() for p in text_params):,}")
        print(f"  Trainable image params: {sum(p.numel() for p in image_params):,}")
    else:
        text_params = list(model.text_adapter.parameters()) + list(model.text_cross_attn.parameters())
        image_params = list(model.image_adapter.parameters()) + list(model.image_cross_attn.parameters())

    # Optimizers
    text_optimizer = torch.optim.Adam(text_params, lr=args.text_lr, betas=(0.5, 0.999))
    image_optimizer = torch.optim.Adam(image_params, lr=args.image_lr, betas=(0.5, 0.999))
    image_scheduler = MultiStepLR(image_optimizer, milestones=[16000, 32000], gamma=0.5)

    # Load checkpoints if exist (resume training)
    text_file = glob(args.save_path + "/text_adapter_cross_attn.pth")
    if len(text_file) > 0:
        ckpt = torch.load(text_file[0], map_location=device)
        model.text_adapter.load_state_dict(ckpt["text_adapter"])
        model.text_cross_attn.load_state_dict(ckpt["text_cross_attn"])
        text_optimizer.load_state_dict(ckpt["text_optimizer"])
        text_start_epoch = ckpt["epoch"]
        adapt_text = not (text_start_epoch >= args.text_epoch)
    elif args.text_epoch == 0:
        adapt_text = False
        text_start_epoch = 0
    else:
        text_start_epoch = 0
        adapt_text = True

    image_file = glob(args.save_path + "/image_adapter_cross_attn.pth")
    if len(image_file) > 0:
        ckpt = torch.load(image_file[0], map_location=device)
        model.image_adapter.load_state_dict(ckpt["image_adapter"])
        model.image_cross_attn.load_state_dict(ckpt["image_cross_attn"])
        image_optimizer.load_state_dict(ckpt["image_optimizer"])
        image_start_epoch = ckpt["epoch"]
    else:
        image_start_epoch = 0

    # Load dataset
    if args.training_mode == "full_shot":
        args.shot = -1
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    text_dataset, image_dataset = get_dataset(
        args.dataset, args.img_size, args.training_mode, args.shot, "train", logger
    )
    text_dataloader = torch.utils.data.DataLoader(
        text_dataset, batch_size=args.text_batch_size, shuffle=True, **kwargs
    )
    image_dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=args.image_batch_size, shuffle=True, **kwargs
    )

    # Stage 1: Text Adapter with Cross-Attention
    if adapt_text:
        model, _ = train_text_adapter_cross_attention(
            adapted_model=model,
            clip_surgery=clip_surgery,
            text_norm_weight=args.text_norm_weight,
            train_loader=text_dataloader,
            optimizer=text_optimizer,
            device=device,
            start_epoch=text_start_epoch,
            dataset_name=args.dataset,
            save_path=args.save_path,
            text_epoch=args.text_epoch,
            img_size=args.img_size,
            logger=logger,
            freeze_adapter=freeze_adapters,
            grad_clip=args.grad_clip,
            debug_frequency=args.debug_frequency,
        )

    del text_dataloader, text_dataset, clip_surgery, text_optimizer
    torch.cuda.empty_cache()

    # Generate text context dict for Stage 2
    text_context_path = os.path.join(args.save_path, "text_context_dict.pth")
    if os.path.exists(text_context_path):
        print(f"\nLoading text_context_dict from: {text_context_path}")
        text_context_dict = torch.load(text_context_path, map_location=device)
    else:
        text_context_dict = generate_text_context_dict(model, args.dataset, device)
        torch.save(text_context_dict, text_context_path)

    model.load_text_context_dict(text_context_dict, debug=True)

    # Generate final text embeddings for loss
    with torch.no_grad():
        model.current_image_context = None
        text_embeddings = get_adapted_text_embedding(model, args.dataset, device)

    # Stage 2: Image Adapter with Cross-Attention
    model, _ = train_image_adapter_cross_attention(
        model=model,
        text_embeddings=text_embeddings,
        train_loader=image_dataloader,
        optimizer=image_optimizer,
        scheduler=image_scheduler,
        device=device,
        start_epoch=image_start_epoch,
        save_path=args.save_path,
        image_epoch=args.image_epoch,
        img_size=args.img_size,
        logger=logger,
        freeze_adapter=freeze_adapters,
        grad_clip=args.grad_clip,
        debug_frequency=args.debug_frequency,
    )

    print(f"\n[{mode_name}] Training Complete!")


def train_feedback_mode(args, logger, device, use_cuda):
    """
    Cross-attention training with feedback loops.

    Implements automatic early stopping based on loss plateau detection.
    Training stops when the combined loss improvement falls below a threshold
    for a specified number of consecutive loops (patience).
    """
    print("\n" + "="*70)
    print("MODE: FEEDBACK (Cross-Attention with Feedback Loops)")
    print("="*70)
    print(f"Maximum feedback loops: {args.num_feedback_loops}")
    print(f"Plateau threshold: {args.plateau_threshold:.2%} improvement")
    print(f"Plateau patience: {args.plateau_patience} consecutive loops")

    # Load models
    clip_surgery = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_surgery.eval()
    clip_surgery.visual.DAPM_replace(DPAM_layer=args.surgery_until_layer)

    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()

    model = AdaptedCLIPWithCrossAttention(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
        cross_attn_heads=args.cross_attn_heads,
        cross_attn_dropout=args.cross_attn_dropout,
    ).to(device)
    model.eval()

    # Load pre-trained adapters if specified
    if args.load_base_text_adapter is not None:
        print(f"\nLoading pre-trained text adapter: {args.load_base_text_adapter}")
        base_ckpt = torch.load(args.load_base_text_adapter, map_location=device)
        model.text_adapter.load_state_dict(base_ckpt["text_adapter"])

    if args.load_base_image_adapter is not None:
        print(f"Loading pre-trained image adapter: {args.load_base_image_adapter}")
        base_ckpt = torch.load(args.load_base_image_adapter, map_location=device)
        model.image_adapter.load_state_dict(base_ckpt["image_adapter"])

    # Load dataset
    if args.training_mode == "full_shot":
        args.shot = -1
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    text_dataset, image_dataset = get_dataset(
        args.dataset, args.img_size, args.training_mode, args.shot, "train", logger
    )

    # Plateau detection tracking
    loop_losses = []  # Track combined (text + image) loss per loop
    plateau_counter = 0  # Count consecutive loops below threshold
    stopped_early = False

    # Feedback loop
    for loop_iteration in range(1, args.num_feedback_loops + 1):
        print("\n" + "#"*70)
        print(f"#  FEEDBACK LOOP {loop_iteration} / {args.num_feedback_loops}")
        print("#"*70)

        # Calculate epochs and LR for this loop (decay)
        lr_factor = args.lr_decay_per_loop ** (loop_iteration - 1)
        epoch_factor = args.epochs_decay_rate ** (loop_iteration - 1)
        text_epoch = max(1, int(args.text_epoch * epoch_factor))
        image_epoch = max(2, int(args.image_epoch * epoch_factor))
        current_text_lr = args.text_lr * lr_factor
        current_image_lr = args.image_lr * lr_factor

        print(f"  Text epochs: {text_epoch}, Image epochs: {image_epoch}")
        print(f"  Text LR: {current_text_lr:.8f}, Image LR: {current_image_lr:.8f}")

        # Optimizers (fresh each loop)
        text_optimizer = torch.optim.Adam(
            list(model.text_adapter.parameters()) + list(model.text_cross_attn.parameters()),
            lr=current_text_lr,
            betas=(0.5, 0.999),
        )
        image_optimizer = torch.optim.Adam(
            list(model.image_adapter.parameters()) + list(model.image_cross_attn.parameters()),
            lr=current_image_lr,
            betas=(0.5, 0.999),
        )
        image_scheduler = MultiStepLR(image_optimizer, milestones=[8000, 16000], gamma=0.5)

        # DataLoaders
        text_dataloader = torch.utils.data.DataLoader(
            text_dataset, batch_size=args.text_batch_size, shuffle=True, **kwargs
        )
        image_dataloader = torch.utils.data.DataLoader(
            image_dataset, batch_size=args.image_batch_size, shuffle=True, **kwargs
        )

        # Stage 1: Text Adapter
        use_trained_features = (loop_iteration > 1)
        model, text_loss = train_text_adapter_cross_attention(
            adapted_model=model,
            clip_surgery=clip_surgery,
            text_norm_weight=args.text_norm_weight,
            train_loader=text_dataloader,
            optimizer=text_optimizer,
            device=device,
            start_epoch=0,
            dataset_name=args.dataset,
            save_path=args.save_path,
            text_epoch=text_epoch,
            img_size=args.img_size,
            logger=logger,
            use_trained_features=use_trained_features,
            loop_iteration=loop_iteration,
            grad_clip=args.grad_clip,
            debug_frequency=args.debug_frequency,
        )

        # Generate text context dict
        text_context_dict = generate_text_context_dict(model, args.dataset, device)
        torch.save(text_context_dict, os.path.join(args.save_path, f"loop_{loop_iteration}_text_context_dict.pth"))
        model.load_text_context_dict(text_context_dict, debug=True)

        # Generate final text embeddings
        with torch.no_grad():
            model.current_image_context = None
            text_embeddings = get_adapted_text_embedding(model, args.dataset, device)

        # Stage 2: Image Adapter
        model, image_loss = train_image_adapter_cross_attention(
            model=model,
            text_embeddings=text_embeddings,
            train_loader=image_dataloader,
            optimizer=image_optimizer,
            scheduler=image_scheduler,
            device=device,
            start_epoch=0,
            save_path=args.save_path,
            image_epoch=image_epoch,
            img_size=args.img_size,
            logger=logger,
            loop_iteration=loop_iteration,
            grad_clip=args.grad_clip,
            debug_frequency=args.debug_frequency,
        )

        # Calculate combined loss for this loop
        combined_loss = text_loss + image_loss
        loop_losses.append({
            "loop": loop_iteration,
            "text_loss": text_loss,
            "image_loss": image_loss,
            "combined_loss": combined_loss,
        })

        # Save loop checkpoint
        loop_ckp = os.path.join(args.save_path, f"loop_{loop_iteration}_final.pth")
        torch.save({
            "loop": loop_iteration,
            "text_adapter": model.text_adapter.state_dict(),
            "text_cross_attn": model.text_cross_attn.state_dict(),
            "image_adapter": model.image_adapter.state_dict(),
            "image_cross_attn": model.image_cross_attn.state_dict(),
            "text_loss": text_loss,
            "image_loss": image_loss,
            "combined_loss": combined_loss,
        }, loop_ckp)
        print(f"  Saved: {loop_ckp}")

        # Plateau detection (need at least 2 loops to compare)
        if loop_iteration >= 2:
            prev_loss = loop_losses[-2]["combined_loss"]
            curr_loss = loop_losses[-1]["combined_loss"]

            # Calculate relative improvement
            if prev_loss > 0:
                relative_improvement = (prev_loss - curr_loss) / prev_loss
            else:
                relative_improvement = 0.0

            print(f"\n  [PLATEAU CHECK]")
            print(f"    Previous combined loss: {prev_loss:.6f}")
            print(f"    Current combined loss:  {curr_loss:.6f}")
            print(f"    Relative improvement:   {relative_improvement:.4%}")
            print(f"    Threshold:              {args.plateau_threshold:.4%}")

            logger.info(f"Loop {loop_iteration} - plateau check: prev={prev_loss:.6f}, curr={curr_loss:.6f}, improvement={relative_improvement:.4%}")

            if relative_improvement < args.plateau_threshold:
                plateau_counter += 1
                print(f"    Status: BELOW threshold ({plateau_counter}/{args.plateau_patience} patience)")
                logger.info(f"Loop {loop_iteration} - below threshold ({plateau_counter}/{args.plateau_patience})")

                if plateau_counter >= args.plateau_patience:
                    print(f"\n  [EARLY STOPPING] Plateau detected!")
                    print(f"    Loss improvement below {args.plateau_threshold:.2%} for {args.plateau_patience} consecutive loops")
                    logger.info(f"Early stopping at loop {loop_iteration} - plateau detected")
                    stopped_early = True
                    break
            else:
                plateau_counter = 0  # Reset counter if improvement is above threshold
                print(f"    Status: ABOVE threshold (reset patience counter)")

    # Summary
    print("\n" + "="*70)
    print("FEEDBACK TRAINING SUMMARY")
    print("="*70)
    print(f"{'Loop':<6} {'Text Loss':<12} {'Image Loss':<12} {'Combined':<12} {'Improvement':<12}")
    print("-"*54)
    for i, loss_info in enumerate(loop_losses):
        if i == 0:
            improvement_str = "N/A"
        else:
            prev = loop_losses[i-1]["combined_loss"]
            curr = loss_info["combined_loss"]
            improvement = (prev - curr) / prev if prev > 0 else 0
            improvement_str = f"{improvement:.4%}"
        print(f"{loss_info['loop']:<6} {loss_info['text_loss']:<12.6f} {loss_info['image_loss']:<12.6f} {loss_info['combined_loss']:<12.6f} {improvement_str:<12}")

    if stopped_early:
        print(f"\nStopped early at loop {len(loop_losses)} (plateau detected)")
    else:
        print(f"\nCompleted all {args.num_feedback_loops} loops")

    # Save final summary
    summary_path = os.path.join(args.save_path, "feedback_summary.pth")
    torch.save({
        "loop_losses": loop_losses,
        "stopped_early": stopped_early,
        "final_loop": len(loop_losses),
        "plateau_threshold": args.plateau_threshold,
        "plateau_patience": args.plateau_patience,
    }, summary_path)
    print(f"Summary saved to: {summary_path}")

    print("\n[FEEDBACK] Training Complete!")


def train_variant_b_mode(args, logger, device, use_cuda):
    """
    Variant B: Freeze adapters[1:], train adapter[0] + cross-attention.

    This mode allows the first adapter layer to co-train with cross-attention,
    giving it flexibility to integrate cross-attention features while keeping
    the rest of the proven adapter weights frozen.
    """
    print("\n" + "="*70)
    print("MODE: VARIANT B (Freeze adapters[1:], train adapter[0] + cross-attn)")
    print("="*70)

    # Load models
    clip_surgery = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_surgery.eval()
    clip_surgery.visual.DAPM_replace(DPAM_layer=args.surgery_until_layer)

    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()

    model = AdaptedCLIPWithCrossAttention(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
        cross_attn_heads=args.cross_attn_heads,
        cross_attn_dropout=args.cross_attn_dropout,
    ).to(device)
    model.eval()

    # Load pre-trained adapters (REQUIRED for Variant B)
    if args.load_base_text_adapter is None or args.load_base_image_adapter is None:
        raise ValueError("Variant B requires --load_base_text_adapter and --load_base_image_adapter")

    print(f"\nLoading pre-trained text adapter: {args.load_base_text_adapter}")
    base_ckpt = torch.load(args.load_base_text_adapter, map_location=device)
    model.text_adapter.load_state_dict(base_ckpt["text_adapter"])
    logger.info(f"Loaded base text_adapter from: {args.load_base_text_adapter}")

    print(f"Loading pre-trained image adapter: {args.load_base_image_adapter}")
    base_ckpt = torch.load(args.load_base_image_adapter, map_location=device)
    model.image_adapter.load_state_dict(base_ckpt["image_adapter"])
    logger.info(f"Loaded base image_adapter from: {args.load_base_image_adapter}")

    # Freeze adapters[1:] for text
    print("\n[VARIANT B] Freezing text_adapter[1:], training text_adapter[0] + text_cross_attn")
    for i, adapter in enumerate(model.text_adapter):
        if i >= 1:  # Freeze adapters 1, 2, ... (keep 0 trainable)
            for param in adapter.parameters():
                param.requires_grad = False
            print(f"  Froze text_adapter[{i}]")

    # Freeze adapters[1:] for image
    print("\n[VARIANT B] Freezing image_adapter[1:], training image_adapter[0] + image_cross_attn")
    # Note: image_adapter is a dict with 'layer_adapters' and 'seg_proj'
    for i in range(1, len(model.image_adapter["layer_adapters"])):
        for param in model.image_adapter["layer_adapters"][i].parameters():
            param.requires_grad = False
        print(f"  Froze image_adapter['layer_adapters'][{i}]")

    # Also freeze seg_proj[1:] (keep seg_proj[0] trainable)
    for i in range(1, len(model.image_adapter["seg_proj"])):
        for param in model.image_adapter["seg_proj"][i].parameters():
            param.requires_grad = False
        print(f"  Froze image_adapter['seg_proj'][{i}]")

    # Collect trainable parameters
    text_params = (
        list(model.text_adapter[0].parameters()) +  # adapter[0]
        list(model.text_cross_attn.parameters())    # cross_attn
    )
    image_params = (
        list(model.image_adapter["layer_adapters"][0].parameters()) +  # adapter[0]
        list(model.image_adapter["seg_proj"][0].parameters()) +        # seg_proj[0]
        list(model.image_cross_attn.parameters())                       # cross_attn
    )

    print(f"\n[TRAINABLE PARAMS]")
    print(f"  Text: adapter[0] + cross_attn = {sum(p.numel() for p in text_params):,}")
    print(f"  Image: adapter[0] + seg_proj[0] + cross_attn = {sum(p.numel() for p in image_params):,}")

    # Optimizers
    text_optimizer = torch.optim.Adam(text_params, lr=args.text_lr, betas=(0.5, 0.999))
    image_optimizer = torch.optim.Adam(image_params, lr=args.image_lr, betas=(0.5, 0.999))
    image_scheduler = MultiStepLR(image_optimizer, milestones=[16000, 32000], gamma=0.5)

    # Load checkpoints if exist (resume training)
    text_file = glob(args.save_path + "/text_adapter_cross_attn.pth")
    if len(text_file) > 0:
        ckpt = torch.load(text_file[0], map_location=device)
        model.text_adapter.load_state_dict(ckpt["text_adapter"])
        model.text_cross_attn.load_state_dict(ckpt["text_cross_attn"])
        text_optimizer.load_state_dict(ckpt["text_optimizer"])
        text_start_epoch = ckpt["epoch"]
        adapt_text = not (text_start_epoch >= args.text_epoch)
        print(f"\nResuming from epoch {text_start_epoch}")
    elif args.text_epoch == 0:
        adapt_text = False
        text_start_epoch = 0
    else:
        text_start_epoch = 0
        adapt_text = True

    image_file = glob(args.save_path + "/image_adapter_cross_attn.pth")
    if len(image_file) > 0:
        ckpt = torch.load(image_file[0], map_location=device)
        model.image_adapter.load_state_dict(ckpt["image_adapter"])
        model.image_cross_attn.load_state_dict(ckpt["image_cross_attn"])
        image_optimizer.load_state_dict(ckpt["image_optimizer"])
        image_start_epoch = ckpt["epoch"]
    else:
        image_start_epoch = 0

    # Load dataset
    if args.training_mode == "full_shot":
        args.shot = -1
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    text_dataset, image_dataset = get_dataset(
        args.dataset, args.img_size, args.training_mode, args.shot, "train", logger
    )
    text_dataloader = torch.utils.data.DataLoader(
        text_dataset, batch_size=args.text_batch_size, shuffle=True, **kwargs
    )
    image_dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=args.image_batch_size, shuffle=True, **kwargs
    )

    # Stage 1: Text Adapter with Cross-Attention
    if adapt_text:
        model, _ = train_text_adapter_cross_attention(
            adapted_model=model,
            clip_surgery=clip_surgery,
            text_norm_weight=args.text_norm_weight,
            train_loader=text_dataloader,
            optimizer=text_optimizer,
            device=device,
            start_epoch=text_start_epoch,
            dataset_name=args.dataset,
            save_path=args.save_path,
            text_epoch=args.text_epoch,
            img_size=args.img_size,
            logger=logger,
            freeze_adapter=False,  # adapter[0] is trainable
            grad_clip=args.grad_clip,
            debug_frequency=args.debug_frequency,
        )

    del text_dataloader, text_dataset, clip_surgery, text_optimizer
    torch.cuda.empty_cache()

    # Generate text context dict for Stage 2
    text_context_path = os.path.join(args.save_path, "text_context_dict.pth")
    if os.path.exists(text_context_path):
        print(f"\nLoading text_context_dict from: {text_context_path}")
        text_context_dict = torch.load(text_context_path, map_location=device)
    else:
        text_context_dict = generate_text_context_dict(model, args.dataset, device)
        torch.save(text_context_dict, text_context_path)

    model.load_text_context_dict(text_context_dict, debug=True)

    # Generate final text embeddings for loss
    with torch.no_grad():
        model.current_image_context = None
        text_embeddings = get_adapted_text_embedding(model, args.dataset, device)

    # Stage 2: Image Adapter with Cross-Attention
    model, _ = train_image_adapter_cross_attention(
        model=model,
        text_embeddings=text_embeddings,
        train_loader=image_dataloader,
        optimizer=image_optimizer,
        scheduler=image_scheduler,
        device=device,
        start_epoch=image_start_epoch,
        save_path=args.save_path,
        image_epoch=args.image_epoch,
        img_size=args.img_size,
        logger=logger,
        freeze_adapter=False,  # adapter[0] is trainable
        grad_clip=args.grad_clip,
        debug_frequency=args.debug_frequency,
    )

    print(f"\n[VARIANT B] Training Complete!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified AA-CLIP Training")

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="cross_attention",
        choices=["base", "cross_attention", "feedback", "freeze_adapters", "variant_b"],
        help="Training mode: base, cross_attention, feedback, freeze_adapters, or variant_b"
    )

    # Model
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--surgery_until_layer", type=int, default=20)
    parser.add_argument("--relu", action="store_true")

    # Training
    parser.add_argument("--dataset", type=str, default="VisA")
    parser.add_argument("--training_mode", type=str, default="full_shot", choices=["few_shot", "full_shot"])
    parser.add_argument("--shot", type=int, default=32)
    parser.add_argument("--text_batch_size", type=int, default=16)
    parser.add_argument("--image_batch_size", type=int, default=2)
    parser.add_argument("--text_epoch", type=int, default=5)
    parser.add_argument("--image_epoch", type=int, default=20)
    parser.add_argument("--text_lr", type=float, default=0.00001)
    parser.add_argument("--image_lr", type=float, default=0.0005)
    parser.add_argument("--criterion", type=str, default=["dice_loss", "focal_loss"], nargs="+")

    # Feedback loop parameters
    parser.add_argument("--num_feedback_loops", type=int, default=10, help="Maximum number of feedback loops")
    parser.add_argument("--lr_decay_per_loop", type=float, default=0.5, help="LR decay factor per loop")
    parser.add_argument("--epochs_decay_rate", type=float, default=0.8, help="Epoch decay rate per loop")
    parser.add_argument("--plateau_threshold", type=float, default=0.01,
                        help="Stop if relative loss improvement < threshold (e.g., 0.01 = 1%%)")
    parser.add_argument("--plateau_patience", type=int, default=2,
                        help="Stop after N consecutive loops below plateau_threshold")

    # Experiment
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/unified")

    # Hyper-parameters
    parser.add_argument("--text_norm_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)

    # Cross-attention
    parser.add_argument("--cross_attn_heads", type=int, default=8)
    parser.add_argument("--cross_attn_dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping (0=disabled)")
    parser.add_argument("--debug_frequency", type=int, default=100)

    # Pre-trained adapter loading
    parser.add_argument("--load_base_text_adapter", type=str, default=None,
                        help="Path to pre-trained text_adapter.pth")
    parser.add_argument("--load_base_image_adapter", type=str, default=None,
                        help="Path to pre-trained image_adapter.pth")

    args = parser.parse_args()

    # Setup
    setup_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "train.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("="*80)
    logger.info(f"UNIFIED TRAINING - MODE: {args.mode.upper()}")
    logger.info("="*80)
    logger.info("args: %s", vars(args))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Device: {device}")

    # Dispatch to appropriate training mode
    if args.mode == "base":
        train_base_mode(args, logger, device, use_cuda)
    elif args.mode == "cross_attention":
        train_cross_attention_mode(args, logger, device, use_cuda, freeze_adapters=False)
    elif args.mode == "freeze_adapters":
        train_cross_attention_mode(args, logger, device, use_cuda, freeze_adapters=True)
    elif args.mode == "variant_b":
        train_variant_b_mode(args, logger, device, use_cuda)
    elif args.mode == "feedback":
        train_feedback_mode(args, logger, device, use_cuda)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Checkpoints saved to: {args.save_path}")


if __name__ == "__main__":
    main()
