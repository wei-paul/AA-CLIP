"""
Test script for Cross-Attention Feedback AA-CLIP

This script evaluates the feedback-trained model by:
1. Loading the trained text adapter, image adapter, and cross-attention modules
2. Computing quantitative metrics (Pixel AUC/AP, Image AUC/AP)
3. Generating visual heatmaps showing anomaly predictions

Usage:
    # Test latest feedback loop only (default)
    python test_cross_attention_feedback.py \
        --dataset VisA \
        --save_path ./ckpt/visa_cross_feedback

    # Test all available feedback loops
    python test_cross_attention_feedback.py \
        --dataset VisA \
        --save_path ./ckpt/visa_cross_feedback \
        --test_all_loops

    # Test specific loop
    python test_cross_attention_feedback.py \
        --dataset VisA \
        --save_path ./ckpt/visa_cross_feedback \
        --loop 2 \
        --visualize

The script will:
- Load checkpoint(s) from loop_N_final.pth
- Test on all classes in the dataset
- Save metrics to test.log
- Save visualizations to {save_path}/visualization/ (if --visualize is enabled)
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
from pandas import DataFrame, Series
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import setup_seed
from model.adapter_cross_attention import AdaptedCLIPWithCrossAttention
from model.clip import create_model
from dataset import get_dataset, DOMAINS
from forward_utils_cross_attention import (
    get_adapted_text_embedding_with_cross_attention,
)
from forward_utils import (
    calculate_similarity_map,
    metrics_eval,
    visualize,
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


def get_predictions(
    model: nn.Module,
    class_text_embeddings: torch.Tensor,
    test_loader: DataLoader,
    device: str,
    img_size: int,
    dataset: str = "VisA",
):
    """
    Get predictions for a single class.

    Args:
        model: The trained AdaptedCLIPWithCrossAttention model
        class_text_embeddings: Pre-computed text embeddings [768, 2]
        test_loader: DataLoader for test images
        device: Device to run on
        img_size: Image size for upsampling predictions
        dataset: Dataset name

    Returns:
        masks: Ground truth masks [N, 1, H, W]
        labels: Image-level labels [N]
        preds: Pixel-level predictions [N, H, W]
        preds_image: Image-level predictions [N]
        file_names: List of image file paths
    """
    masks = []
    labels = []
    preds = []
    preds_image = []
    file_names = []

    for input_data in tqdm(test_loader):
        image = input_data["image"].to(device)
        mask = input_data["mask"].cpu().numpy()
        label = input_data["label"].cpu().numpy()
        file_name = input_data["file_name"]
        class_name = input_data["class_name"]

        # Ensure single class per batch
        assert len(set(class_name)) == 1, "mixed class not supported"

        masks.append(mask)
        labels.append(label)
        file_names.extend(file_name)

        # Get text embeddings for this class
        epoch_text_feature = class_text_embeddings

        # Forward pass through image encoder
        patch_features, det_feature = model(image)

        # Image-level prediction (classification)
        pred = det_feature @ epoch_text_feature
        pred = (pred[:, 1] + 1) / 2  # Normalize to [0, 1]
        preds_image.append(pred.cpu().numpy())

        # Pixel-level prediction (segmentation)
        # Average predictions across all 4 scales
        patch_preds = []
        for f in patch_features:
            # f: [bs, patch_num, 768]
            patch_pred = calculate_similarity_map(
                f, epoch_text_feature, img_size, test=True, domain=DOMAINS[dataset]
            )
            patch_preds.append(patch_pred)

        # Combine multi-scale predictions
        patch_preds = torch.cat(patch_preds, dim=1).sum(1).cpu().numpy()
        preds.append(patch_preds)

    # Concatenate all batches
    masks = np.concatenate(masks, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    preds_image = np.concatenate(preds_image, axis=0)

    return masks, labels, preds, preds_image, file_names


def main():
    parser = argparse.ArgumentParser(description="Test Cross-Attention Feedback AA-CLIP")

    # Model architecture
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="ViT-B-16-plus-240, ViT-L-14-336",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")

    # Testing configuration
    parser.add_argument("--dataset", type=str, default="VisA", help="MVTec or VisA")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)

    # Checkpoint and visualization
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument(
        "--save_path",
        type=str,
        default="./ckpt/visa_cross_feedback",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visual heatmaps (saves to {save_path}/visualization/)",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=None,
        help="Which feedback loop checkpoint to load (1, 2, or 3). If not specified, tests latest.",
    )
    parser.add_argument(
        "--test_all_loops",
        action="store_true",
        help="Test all available feedback loop checkpoints",
    )

    # Adapter configuration (must match training)
    parser.add_argument("--text_norm_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)
    parser.add_argument("--cross_attn_heads", type=int, default=8)
    parser.add_argument("--cross_attn_dropout", type=float, default=0.1)

    args = parser.parse_args()

    # ========================================================
    # Setup
    # ========================================================
    setup_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    # Logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "test.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("=" * 80)
    logger.info("TESTING CROSS-ATTENTION FEEDBACK AA-CLIP")
    logger.info("=" * 80)
    logger.info("args: %s", vars(args))

    # Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logger.info(f"Using device: {device}")

    # ========================================================
    # Load Model
    # ========================================================
    logger.info("Loading CLIP base model...")
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()

    logger.info("Creating AdaptedCLIPWithCrossAttention...")
    model = AdaptedCLIPWithCrossAttention(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        cross_attn_heads=args.cross_attn_heads,
        cross_attn_dropout=args.cross_attn_dropout,
        relu=args.relu,
    ).to(device)
    model.eval()

    # ========================================================
    # Find Available Feedback Loop Checkpoints
    # ========================================================
    loop_ckpts = sorted(glob(os.path.join(args.save_path, "loop_*_final.pth")))

    if not loop_ckpts:
        logger.error(f"No feedback loop checkpoints found in {args.save_path}")
        print(f"ERROR: No feedback loop checkpoints found in {args.save_path}")
        print(f"\nAvailable checkpoints in {args.save_path}:")
        for f in sorted(glob(os.path.join(args.save_path, "*.pth"))):
            print(f"  - {os.path.basename(f)}")
        return

    # Determine which checkpoints to test
    if args.test_all_loops:
        # Test all available loops
        checkpoints_to_test = loop_ckpts
        print(f"\n✓ Testing ALL {len(checkpoints_to_test)} available feedback loops")
    elif args.loop is not None:
        # Test specific loop
        checkpoint_path = os.path.join(args.save_path, f"loop_{args.loop}_final.pth")
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint for loop {args.loop} not found")
            print(f"Available loops: {[int(f.split('_')[1]) for f in [os.path.basename(p) for p in loop_ckpts]]}")
            return
        checkpoints_to_test = [checkpoint_path]
        print(f"\n✓ Testing specific loop: {args.loop}")
    else:
        # Test latest loop only (default)
        checkpoints_to_test = [loop_ckpts[-1]]
        print(f"\n✓ Testing latest loop only (use --test_all_loops to test all)")

    # ========================================================
    # Pre-compute Dataset (same for all loops)
    # ========================================================
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    logger.info("Loading test dataset...")
    print(f"\nLoading test dataset: {args.dataset}")
    image_datasets = get_dataset(
        args.dataset,
        args.img_size,
        None,
        args.shot,
        "test",
        logger=logger,
    )

    # ========================================================
    # Loop Through Selected Checkpoints
    # ========================================================
    for checkpoint_path in checkpoints_to_test:
        # Load checkpoint
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        print(f"\nLoading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load all trained components
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        model.text_cross_attn.load_state_dict(checkpoint["text_cross_attn"])
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
        model.image_cross_attn.load_state_dict(checkpoint["image_cross_attn"])

        loop_num = checkpoint['loop']
        logger.info(f"Loaded feedback loop {loop_num} checkpoint")
        print(f"✓ Loaded feedback loop {loop_num} checkpoint")

        # ========================================================
        # Pre-compute Text Embeddings for This Loop
        # ========================================================
        logger.info("Pre-computing text embeddings for all classes...")
        print("Pre-computing text embeddings...")

        with torch.no_grad():
            # NOTE: For testing, we don't use cross-attention
            # (no image context available before seeing test images)
            # The model will automatically skip cross-attention when
            # current_image_context is None
            text_embeddings = get_adapted_text_embedding_with_cross_attention(
                model, args.dataset, device
            )

        print(f"✓ Text embeddings computed for {len(text_embeddings)} classes")

        logger.info("=" * 80)
        logger.info(f"Testing Feedback Loop {loop_num}")
        logger.info("=" * 80)
        print("\n" + "=" * 80)
        print(f"Testing Feedback Loop {loop_num}")
        print("=" * 80)

        # ========================================================
        # Testing Loop for This Checkpoint
        # ========================================================
        df = DataFrame(
            columns=[
                "class name",
                "pixel AUC",
                "pixel AP",
                "image AUC",
                "image AP",
            ]
        )

        for class_name, image_dataset in image_datasets.items():
            print(f"\nClass: {class_name}")
            logger.info(f"Testing class: {class_name}")

            image_dataloader = torch.utils.data.DataLoader(
                image_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
            )

            # Get predictions
            with torch.no_grad():
                class_text_embeddings = text_embeddings[class_name]
                masks, labels, preds, preds_image, file_names = get_predictions(
                    model=model,
                    class_text_embeddings=class_text_embeddings,
                    test_loader=image_dataloader,
                    device=device,
                    img_size=args.img_size,
                    dataset=args.dataset,
                )

            # Visualize if requested (only for single loop testing to avoid clutter)
            if args.visualize and len(checkpoints_to_test) == 1:
                print(f"  Generating visualizations...")
                visualize(
                    masks,
                    preds,
                    file_names,
                    args.save_path,
                    args.dataset,
                    class_name=class_name,
                )

            # Compute metrics
            class_result_dict = metrics_eval(
                masks,
                labels,
                preds,
                preds_image,
                class_name,
                domain=DOMAINS[args.dataset],
            )

            # Print class results
            print(f"  Pixel AUC: {class_result_dict['pixel AUC']:.4f}")
            print(f"  Pixel AP:  {class_result_dict['pixel AP']:.4f}")
            print(f"  Image AUC: {class_result_dict['image AUC']:.4f}")
            print(f"  Image AP:  {class_result_dict['image AP']:.4f}")

            df.loc[len(df)] = Series(class_result_dict)

        # ========================================================
        # Compute and Display Average Results for This Loop
        # ========================================================
        # Calculate average for numeric columns only
        avg_row = {"class name": "Average"}
        numeric_cols = ["pixel AUC", "pixel AP", "image AUC", "image AP"]
        for col in numeric_cols:
            if col in df.columns:
                avg_row[col] = df[col].mean()
        df.loc[len(df)] = Series(avg_row)

        logger.info("=" * 80)
        logger.info(f"RESULTS FOR FEEDBACK LOOP {loop_num}")
        logger.info("=" * 80)
        logger.info("\n" + df.to_string(index=False, justify="center"))

        print("\n" + "=" * 80)
        print(f"RESULTS FOR FEEDBACK LOOP {loop_num}")
        print("=" * 80)
        print(df.to_string(index=False, justify="center"))
        print("\n" + "=" * 80)

    # Final message
    if args.visualize and len(checkpoints_to_test) == 1:
        vis_path = os.path.join(args.save_path, "visualization", args.dataset)
        print(f"\nVisualizations saved to: {vis_path}")

    print(f"\nTest log saved to: {os.path.join(args.save_path, 'test.log')}")


if __name__ == "__main__":
    main()
