"""
Cross-Attention Enhanced Training Script for AA-CLIP

This script implements BOTH Stage 1 and Stage 2 training with cross-attention:
- Stage 1: Text adapters trained with text→image cross-attention
- Stage 2: Image adapters trained with image→text cross-attention

Cross-Attention Design:
Stage 1 (Text → Image):
    - Q: Text features after layer 0 [B, 77, 768]
    - K/V: V1 patches + CLS_24 residual [B, 1369, 768]
    - Location: After text layer 0, before text_adapter[0]

Stage 2 (Image → Text):
    - Q: Image features after ViT layer 0 [B, 1370, 1024]
    - K/V: Text features from Stage 1 [B, 1232, 768] (per-class)
    - Location: After ViT layer 0, before image_adapter[0]

Usage:
    python train_cross_attention.py --dataset MVTec --save_path ckpt/cross_attn

Author: Cross-attention implementation for AA-CLIP
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
from torch.optim.lr_scheduler import MultiStepLR
from utils import setup_seed
from model.adapter_cross_attention import AdaptedCLIPWithCrossAttention
from model.clip import create_model
from dataset import get_dataset
from dataset.constants import CLASS_NAMES, REAL_NAMES, PROMPTS
from model.tokenizer import tokenize
from forward_utils import (
    get_adapted_text_embedding,
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


# ========================================
# Prompt configuration
# ========================================
prompt = PROMPTS
prompt_normal = prompt["prompt_normal"]
prompt_abnormal = prompt["prompt_abnormal"]
prompt_state = [prompt_normal, prompt_abnormal]
prompt_templates = prompt["prompt_templates"]


# ========================================
# STAGE 1: Compute Image Context
# ========================================
def compute_image_context(image, clip_surgery, adapted_model, device, debug=False):
    """
    Compute image context for Stage 1 cross-attention.

    Returns V1 patches (layer 6) + CLS_24 residual = [B, 1369, 768]

    Args:
        image: Input images [B, 3, H, W]
        clip_surgery: CLIP model with surgery
        adapted_model: Adapted CLIP model
        device: torch device
        debug: If True, print debug info

    Returns:
        image_context: [B, 1369, 768]
    """
    with torch.no_grad():
        if debug:
            print("\n" + "="*60)
            print("COMPUTING IMAGE CONTEXT (Stage 1)")
            print("="*60)
            print(f"[INPUT] image shape: {image.shape}")

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
            print(f"[OUTPUT] image_context shape: {image_context.shape}")
            print("="*60 + "\n")

    return image_context


# ========================================
# STAGE 1: Train Text Adapter with Cross-Attention
# ========================================
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
    debug_frequency: int = 100,
):
    """
    Train text adapter with cross-attention to image features (Stage 1).

    Trainable: text_adapter + text_cross_attn
    """
    batch_count = 0

    for epoch in range(start_epoch, text_epoch):
        logger.info(f"training text epoch {epoch} (with cross-attention):")
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{text_epoch-1} - STAGE 1: TEXT CROSS-ATTENTION")
        print(f"{'='*70}")

        loss_list = []

        for input_data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            class_names = input_data["class_name"]

            debug_this_batch = (batch_count % debug_frequency == 0)

            if debug_this_batch:
                print(f"\n{'='*60}")
                print(f"BATCH {batch_count} DEBUG")
                print(f"{'='*60}")
                print(f"  image: {image.shape}, class_names: {class_names}")

            # Compute image context for cross-attention
            image_context = compute_image_context(
                image, clip_surgery, adapted_model, device, debug=debug_this_batch
            )
            adapted_model.set_image_context(image_context, debug=debug_this_batch)

            # Generate text embeddings WITH cross-attention
            if debug_this_batch:
                adapted_model.debug_cross_attn = True

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

            if debug_this_batch:
                print(f"\n[TEXT EMBEDDINGS]")
                print(f"  shape: {epoch_text_feature.shape}")
                adapted_model.debug_cross_attn = False

            # Compute patch features for loss
            with torch.no_grad():
                _, patch_features = clip_surgery.encode_image(image, [6, 12, 18, 24])
                cls_token, _ = adapted_model.clipmodel.encode_image(image, [])
                cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
                patch_features = [clip_surgery.visual.ln_post(t[:, 1:, :]) for t in patch_features]
                patch_features = [t @ clip_surgery.visual.proj for t in patch_features]
                patch_features = [t / t.norm(dim=-1, keepdim=True) for t in patch_features]
                patch_features = [t + cls_token.unsqueeze(1) for t in patch_features]

            # Calculate loss
            for f in patch_features:
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                loss = calculate_seg_loss(patch_preds, mask)
                orthogonal_loss = ((epoch_text_feature[:, :, 0] * epoch_text_feature[:, :, 1]).sum(1).mean()) ** 2
                loss += orthogonal_loss * text_norm_weight

            if debug_this_batch:
                print(f"\n[LOSS] total: {loss.item():.6f}")

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            if debug_this_batch:
                print(f"\n[GRADIENTS]")
                for i, adapter in enumerate(adapted_model.text_adapter):
                    grad_norm = sum(p.grad.norm().item() for p in adapter.parameters() if p.grad is not None)
                    print(f"  text_adapter[{i}]: {grad_norm:.6f}")
                for name, param in adapted_model.text_cross_attn.named_parameters():
                    if param.grad is not None:
                        print(f"  text_cross_attn.{name}: {param.grad.norm().item():.6f}")

            optimizer.step()
            loss_list.append(loss.item())
            adapted_model.clear_image_context()
            batch_count += 1

            if debug_this_batch:
                print(f"{'='*60}\n")

        # End of epoch
        avg_loss = np.mean(loss_list)
        logger.info(f"loss: {avg_loss}")
        print(f"\nEpoch {epoch} Complete - Average Loss: {avg_loss:.6f}")

        # Save checkpoint
        ckp_path = os.path.join(save_path, "text_adapter_cross_attn.pth")
        torch.save({
            "epoch": epoch + 1,
            "text_adapter": adapted_model.text_adapter.state_dict(),
            "text_cross_attn": adapted_model.text_cross_attn.state_dict(),
            "text_optimizer": optimizer.state_dict(),
        }, ckp_path)
        print(f"Checkpoint saved: {ckp_path}")

    return adapted_model


# ========================================
# Generate Text Context Dict for Stage 2
# ========================================
def generate_text_context_dict(model, dataset_name, device, debug=False):
    """
    Generate text context dictionary for Stage 2 cross-attention.

    For each class, get text features after layer 0 + adapter[0].

    Returns:
        text_context_dict: {class_name: [16, 77, 768]}
    """
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


# ========================================
# STAGE 2: Train Image Adapter with Cross-Attention
# ========================================
def train_image_adapter_with_cross_attention(
    model: nn.Module,
    text_embeddings: dict,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: str,
    start_epoch: int,
    save_path: str,
    image_epoch: int,
    img_size: int,
    logger: logging.Logger,
    debug_frequency: int = 100,
):
    """
    Train image adapter with cross-attention to text features (Stage 2).

    Trainable: image_adapter + image_cross_attn
    """
    batch_count = 0

    for epoch in range(start_epoch, image_epoch):
        logger.info(f"training image epoch {epoch} (with cross-attention):")
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{image_epoch-1} - STAGE 2: IMAGE CROSS-ATTENTION")
        print(f"{'='*70}")

        loss_list = []

        for input_data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            label = input_data["label"].to(device)
            class_names = input_data["class_name"]
            B = image.shape[0]

            debug_this_batch = (batch_count % debug_frequency == 0)

            if debug_this_batch:
                print(f"\n{'='*60}")
                print(f"BATCH {batch_count} DEBUG")
                print(f"{'='*60}")
                print(f"  image: {image.shape}, class_names: {class_names}")

            # Prepare text context for cross-attention (per-class)
            text_context = model.prepare_text_context_for_batch(
                class_names, device, debug=debug_this_batch
            )

            if debug_this_batch:
                print(f"\n[TEXT CONTEXT for Stage 2]")
                print(f"  shape: {text_context.shape}")
                model.debug_cross_attn = True

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

            if debug_this_batch:
                print(f"\n[IMAGE FEATURES]")
                print(f"  patch_features: {[pf.shape for pf in patch_features]}")
                print(f"  det_feature: {det_feature.shape}")
                model.debug_cross_attn = False

            # Calculate loss
            loss = 0.0

            # Classification loss
            det_feature_exp = det_feature.unsqueeze(1)
            cls_preds = torch.matmul(det_feature_exp, epoch_text_feature)[:, 0]
            cls_loss = F.cross_entropy(cls_preds, label)
            loss += cls_loss

            # Segmentation loss
            seg_loss_total = 0.0
            for f in patch_features:
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                seg_loss = calculate_seg_loss(patch_preds, mask)
                loss += seg_loss
                seg_loss_total += seg_loss.item()

            if debug_this_batch:
                print(f"\n[LOSS]")
                print(f"  cls_loss: {cls_loss.item():.6f}")
                print(f"  seg_loss: {seg_loss_total:.6f}")
                print(f"  total: {loss.item():.6f}")

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            if debug_this_batch:
                print(f"\n[GRADIENTS]")
                for i in range(len(model.image_adapter["layer_adapters"])):
                    adapter = model.image_adapter["layer_adapters"][i]
                    grad_norm = sum(p.grad.norm().item() for p in adapter.parameters() if p.grad is not None)
                    print(f"  image_adapter[{i}]: {grad_norm:.6f}")
                for name, param in model.image_cross_attn.named_parameters():
                    if param.grad is not None:
                        print(f"  image_cross_attn.{name}: {param.grad.norm().item():.6f}")

            optimizer.step()
            loss_list.append(loss.item())
            scheduler.step()
            model.clear_text_context()
            batch_count += 1

            if debug_this_batch:
                print(f"{'='*60}\n")

        # End of epoch
        avg_loss = np.mean(loss_list)
        logger.info(f"loss: {avg_loss}")
        print(f"\nEpoch {epoch} Complete - Average Loss: {avg_loss:.6f}")

        # Save checkpoint
        ckp_path = os.path.join(save_path, "image_adapter_cross_attn.pth")
        torch.save({
            "epoch": epoch + 1,
            "image_adapter": model.image_adapter.state_dict(),
            "image_cross_attn": model.image_cross_attn.state_dict(),
            "image_optimizer": optimizer.state_dict(),
        }, ckp_path)
        print(f"Checkpoint saved: {ckp_path}")

        # Epoch-specific checkpoint
        epoch_ckp = os.path.join(save_path, f"image_adapter_cross_attn_{epoch + 1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "image_adapter": model.image_adapter.state_dict(),
            "image_cross_attn": model.image_cross_attn.state_dict(),
            "image_optimizer": optimizer.state_dict(),
        }, epoch_ckp)

    return model


# ========================================
# MAIN
# ========================================
def main():
    parser = argparse.ArgumentParser(description="AA-CLIP Training with Cross-Attention")
    # model
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--surgery_until_layer", type=int, default=20)
    parser.add_argument("--relu", action="store_true")
    # training
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--training_mode", type=str, default="few_shot", choices=["few_shot", "full_shot"])
    parser.add_argument("--shot", type=int, default=32)
    parser.add_argument("--text_batch_size", type=int, default=16)
    parser.add_argument("--image_batch_size", type=int, default=2)
    parser.add_argument("--text_epoch", type=int, default=5)
    parser.add_argument("--image_epoch", type=int, default=20)
    parser.add_argument("--text_lr", type=float, default=0.00001)
    parser.add_argument("--image_lr", type=float, default=0.0005)
    parser.add_argument("--criterion", type=str, default=["dice_loss", "focal_loss"], nargs="+")
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/cross_attention")
    # hyper-parameters
    parser.add_argument("--text_norm_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)
    # Cross-attention
    parser.add_argument("--cross_attn_heads", type=int, default=8)
    parser.add_argument("--cross_attn_dropout", type=float, default=0.1)
    parser.add_argument("--debug_frequency", type=int, default=100)
    # Pre-trained base model loading
    parser.add_argument(
        "--load_base_text_adapter",
        type=str,
        default=None,
        help="Path to pre-trained base model text_adapter.pth (for fair comparison)"
    )
    parser.add_argument(
        "--load_base_image_adapter",
        type=str,
        default=None,
        help="Path to pre-trained base model image_adapter.pth (for fair comparison)"
    )

    args = parser.parse_args()

    # Setup
    setup_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "train_cross_attention.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("args: %s", vars(args))

    print("\n" + "="*70)
    print("AA-CLIP TRAINING WITH BIDIRECTIONAL CROSS-ATTENTION")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Save path: {args.save_path}")
    print(f"Stage 1 epochs: {args.text_epoch}")
    print(f"Stage 2 epochs: {args.image_epoch}")
    print(f"Cross-attention heads: {args.cross_attn_heads}")
    print("="*70 + "\n")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Device: {device}")

    # ========================================
    # Load Models
    # ========================================
    print("\nLoading CLIP models...")

    clip_surgery = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_surgery.eval()
    clip_surgery.visual.DAPM_replace(DPAM_layer=args.surgery_until_layer)
    print(f"  clip_surgery loaded")

    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    print("  clip_model loaded")

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
    print("  AdaptedCLIPWithCrossAttention loaded")

    # Count parameters
    text_adapter_params = sum(p.numel() for p in model.text_adapter.parameters())
    text_cross_params = sum(p.numel() for p in model.text_cross_attn.parameters())
    image_adapter_params = sum(p.numel() for p in model.image_adapter.parameters())
    image_cross_params = sum(p.numel() for p in model.image_cross_attn.parameters())
    print(f"\n[PARAMETERS]")
    print(f"  Stage 1: text_adapter ({text_adapter_params:,}) + text_cross_attn ({text_cross_params:,})")
    print(f"  Stage 2: image_adapter ({image_adapter_params:,}) + image_cross_attn ({image_cross_params:,})")

    # ========================================
    # Load Pre-trained Base Model Weights (Optional)
    # ========================================
    if args.load_base_text_adapter is not None:
        print(f"\n{'='*70}")
        print("LOADING PRE-TRAINED BASE TEXT ADAPTER")
        print(f"{'='*70}")
        print(f"  Path: {args.load_base_text_adapter}")
        try:
            base_ckpt = torch.load(args.load_base_text_adapter, map_location=device)
            model.text_adapter.load_state_dict(base_ckpt["text_adapter"])
            logger.info(f"Loaded base text_adapter from: {args.load_base_text_adapter}")
            print("  ✓ Base text adapter loaded successfully!")

            # Verify loading by checking weight statistics
            text_mean = sum(p.abs().mean().item() for p in model.text_adapter.parameters()) / len(list(model.text_adapter.parameters()))
            print(f"  Mean weight magnitude: {text_mean:.6f} (should be > 0.1 if trained)")
        except Exception as e:
            print(f"  ✗ Failed to load base text adapter: {e}")
            logger.warning(f"Failed to load base text adapter: {e}")
    else:
        print("\n[NOTE] Training text_adapter from scratch (random initialization)")

    if args.load_base_image_adapter is not None:
        print(f"\n{'='*70}")
        print("LOADING PRE-TRAINED BASE IMAGE ADAPTER")
        print(f"{'='*70}")
        print(f"  Path: {args.load_base_image_adapter}")
        try:
            base_ckpt = torch.load(args.load_base_image_adapter, map_location=device)
            model.image_adapter.load_state_dict(base_ckpt["image_adapter"])
            logger.info(f"Loaded base image_adapter from: {args.load_base_image_adapter}")
            print("  ✓ Base image adapter loaded successfully!")

            # Verify loading
            image_mean = sum(p.abs().mean().item() for p in model.image_adapter.parameters()) / len(list(model.image_adapter.parameters()))
            print(f"  Mean weight magnitude: {image_mean:.6f} (should be > 0.1 if trained)")
        except Exception as e:
            print(f"  ✗ Failed to load base image adapter: {e}")
            logger.warning(f"Failed to load base image adapter: {e}")
    else:
        print("\n[NOTE] Training image_adapter from scratch (random initialization)")

    # Print weight initialization summary
    print(f"\n{'='*70}")
    print("WEIGHT INITIALIZATION SUMMARY")
    print(f"{'='*70}")
    text_adapter_mean = sum(p.abs().mean().item() for p in model.text_adapter.parameters()) / len(list(model.text_adapter.parameters()))
    text_cross_mean = sum(p.abs().mean().item() for p in model.text_cross_attn.parameters()) / len(list(model.text_cross_attn.parameters()))
    image_adapter_mean = sum(p.abs().mean().item() for p in model.image_adapter.parameters()) / len(list(model.image_adapter.parameters()))
    image_cross_mean = sum(p.abs().mean().item() for p in model.image_cross_attn.parameters()) / len(list(model.image_cross_attn.parameters()))

    print(f"  text_adapter:       {text_adapter_mean:.6f} {'(pre-trained)' if args.load_base_text_adapter else '(random init)'}")
    print(f"  text_cross_attn:    {text_cross_mean:.6f} (random init - NEW)")
    print(f"  image_adapter:      {image_adapter_mean:.6f} {'(pre-trained)' if args.load_base_image_adapter else '(random init)'}")
    print(f"  image_cross_attn:   {image_cross_mean:.6f} (random init - NEW)")
    print(f"{'='*70}\n")

    logger.info(f"Weight init - text_adapter: {text_adapter_mean:.6f}, text_cross: {text_cross_mean:.6f}")
    logger.info(f"Weight init - image_adapter: {image_adapter_mean:.6f}, image_cross: {image_cross_mean:.6f}")

    # ========================================
    # Optimizers
    # ========================================
    text_optimizer = torch.optim.Adam(
        list(model.text_adapter.parameters()) + list(model.text_cross_attn.parameters()),
        lr=args.text_lr,
        betas=(0.5, 0.999),
    )
    image_optimizer = torch.optim.Adam(
        list(model.image_adapter.parameters()) + list(model.image_cross_attn.parameters()),
        lr=args.image_lr,
        betas=(0.5, 0.999),
    )
    image_scheduler = MultiStepLR(image_optimizer, milestones=[16000, 32000], gamma=0.5)

    # ========================================
    # Load Checkpoints (Resume Training)
    # NOTE: This OVERWRITES base model weights if checkpoint exists!
    #       This is correct for resuming cross-attn training.
    #       Base weights are only used for FRESH training (no checkpoint).
    # ========================================
    text_file = glob(args.save_path + "/text_adapter_cross_attn.pth")
    if len(text_file) > 0:
        print(f"\n{'='*70}")
        print("RESUMING FROM CROSS-ATTENTION CHECKPOINT (Stage 1)")
        print(f"{'='*70}")
        print(f"  Path: {text_file[0]}")
        if args.load_base_text_adapter:
            print("  [!] NOTE: Base text adapter will be OVERWRITTEN by checkpoint")
        ckpt = torch.load(text_file[0], map_location=device)
        model.text_adapter.load_state_dict(ckpt["text_adapter"])
        model.text_cross_attn.load_state_dict(ckpt["text_cross_attn"])
        text_optimizer.load_state_dict(ckpt["text_optimizer"])
        text_start_epoch = ckpt["epoch"]
        adapt_text = not (text_start_epoch >= args.text_epoch)
        print(f"  ✓ Resuming Stage 1 from epoch {text_start_epoch}")
        logger.info(f"Resumed Stage 1 from checkpoint epoch {text_start_epoch}")
    elif args.text_epoch == 0:
        adapt_text = False
        text_start_epoch = 0
        print("\n[SKIP] Stage 1: text_epoch=0, skipping text adapter training")
    else:
        text_start_epoch = 0
        adapt_text = True
        if args.load_base_text_adapter:
            print(f"\n[START] Stage 1: Fresh training with PRE-TRAINED base text adapter")
        else:
            print(f"\n[START] Stage 1: Fresh training from RANDOM initialization")

    image_file = glob(args.save_path + "/image_adapter_cross_attn.pth")
    if len(image_file) > 0:
        print(f"\n{'='*70}")
        print("RESUMING FROM CROSS-ATTENTION CHECKPOINT (Stage 2)")
        print(f"{'='*70}")
        print(f"  Path: {image_file[0]}")
        if args.load_base_image_adapter:
            print("  [!] NOTE: Base image adapter will be OVERWRITTEN by checkpoint")
        ckpt = torch.load(image_file[0], map_location=device)
        model.image_adapter.load_state_dict(ckpt["image_adapter"])
        model.image_cross_attn.load_state_dict(ckpt["image_cross_attn"])
        image_optimizer.load_state_dict(ckpt["image_optimizer"])
        image_start_epoch = ckpt["epoch"]
        print(f"  ✓ Resuming Stage 2 from epoch {image_start_epoch}")
        logger.info(f"Resumed Stage 2 from checkpoint epoch {image_start_epoch}")
    else:
        image_start_epoch = 0
        if args.load_base_image_adapter:
            print(f"\n[START] Stage 2: Fresh training with PRE-TRAINED base image adapter")
        else:
            print(f"\n[START] Stage 2: Fresh training from RANDOM initialization")

    # ========================================
    # Load Dataset
    # ========================================
    if args.training_mode == "full_shot":
        args.shot = -1
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    print(f"\nLoading dataset: {args.dataset}")
    text_dataset, image_dataset = get_dataset(
        args.dataset, args.img_size, args.training_mode, args.shot, "train", logger
    )
    text_dataloader = torch.utils.data.DataLoader(
        text_dataset, batch_size=args.text_batch_size, shuffle=True, **kwargs
    )
    image_dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=args.image_batch_size, shuffle=True, **kwargs
    )
    print(f"  Text dataset: {len(text_dataset)} samples")
    print(f"  Image dataset: {len(image_dataset)} samples")

    # ========================================
    # STAGE 1: Text Adapter Training
    # ========================================
    if adapt_text:
        print("\n" + "="*70)
        print("STAGE 1: TEXT ADAPTER TRAINING WITH CROSS-ATTENTION")
        print("="*70)
        model = train_text_adapter_with_cross_attention(
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
            debug_frequency=args.debug_frequency,
        )
        print("\nStage 1 Complete!")

    # Clean up Stage 1 resources
    del text_dataloader, text_dataset, clip_surgery, text_optimizer
    torch.cuda.empty_cache()

    # ========================================
    # Generate Text Context Dict for Stage 2
    # ========================================
    text_context_path = os.path.join(args.save_path, "text_context_dict.pth")
    if os.path.exists(text_context_path):
        print(f"\nLoading text_context_dict from: {text_context_path}")
        text_context_dict = torch.load(text_context_path, map_location=device)
    else:
        print("\nGenerating text_context_dict...")
        text_context_dict = generate_text_context_dict(model, args.dataset, device)
        torch.save(text_context_dict, text_context_path)
        print(f"Saved to: {text_context_path}")

    model.load_text_context_dict(text_context_dict, debug=True)

    # ========================================
    # Generate Final Text Embeddings for Loss
    # ========================================
    print("\n" + "="*70)
    print("GENERATING FINAL TEXT EMBEDDINGS FOR STAGE 2 LOSS")
    print("="*70)
    with torch.no_grad():
        model.current_image_context = None  # No cross-attention for final embeddings
        text_embeddings = get_adapted_text_embedding(model, args.dataset, device)
    print(f"  Generated for {len(text_embeddings)} classes")

    # ========================================
    # STAGE 2: Image Adapter Training
    # ========================================
    print("\n" + "="*70)
    print("STAGE 2: IMAGE ADAPTER TRAINING WITH CROSS-ATTENTION")
    print("="*70)
    model = train_image_adapter_with_cross_attention(
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
        debug_frequency=args.debug_frequency,
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Checkpoints saved to: {args.save_path}")
    print(f"  - text_adapter_cross_attn.pth (Stage 1)")
    print(f"  - text_context_dict.pth (for Stage 2)")
    print(f"  - image_adapter_cross_attn.pth (Stage 2)")


if __name__ == "__main__":
    main()
