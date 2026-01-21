"""Offline evaluation script for computing action prediction error on held-out data.

Usage:
    uv run scripts/eval_offline.py --config pi05_bridge --checkpoint-dir checkpoints/pi05_bridge/bridge_run_20k/10000

This script computes:
1. Flow matching loss (the training objective)
2. Action prediction MSE (comparing predicted vs ground truth actions)
"""

import argparse
import dataclasses
from pathlib import Path

import numpy as np
from tqdm import tqdm
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Offline evaluation of a trained policy")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Training config name (e.g., 'pi05_bridge')",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., 'checkpoints/pi05_bridge/bridge_run_20k/10000')",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate on",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip action inference MSE (faster, only compute flow loss)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index in dataset (use to evaluate on different splits)",
    )
    return parser.parse_args()


def compute_metrics(pred_actions: np.ndarray, gt_actions: np.ndarray) -> dict:
    """Compute evaluation metrics between predicted and ground truth actions.

    Args:
        pred_actions: Predicted actions, shape (batch, horizon, action_dim) or (horizon, action_dim)
        gt_actions: Ground truth actions, same shape as pred_actions

    Returns:
        Dictionary of metrics
    """
    # Ensure both have same shape
    pred_actions = np.asarray(pred_actions)
    gt_actions = np.asarray(gt_actions)

    # Compute MSE
    mse = np.mean((pred_actions - gt_actions) ** 2)

    # Compute per-dimension MSE
    per_dim_mse = np.mean((pred_actions - gt_actions) ** 2, axis=(0, 1) if pred_actions.ndim == 3 else 0)

    # Compute MAE
    mae = np.mean(np.abs(pred_actions - gt_actions))

    # Compute RMSE
    rmse = np.sqrt(mse)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "per_dim_mse": per_dim_mse.tolist(),
    }


def main():
    args = parse_args()

    print(f"Loading config: {args.config}")
    config = _config.get_config(args.config)
    data_config = config.data.create(config.assets_dirs, config.model)

    print(f"Loading checkpoint from: {args.checkpoint_dir}")
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        # Try downloading if it's a GCS path
        checkpoint_dir = download.maybe_download(args.checkpoint_dir)

    # Create a trained policy for inference
    print("Creating policy...")
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    # Load raw dataset for inference evaluation
    print(f"Loading dataset: {data_config.repo_id}")
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(config.model.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    # Add prompt from task
    prompt_transform = _transforms.PromptFromLeRobotTask(dataset_meta.tasks)

    # Create repack transform for converting dataset format to policy input format
    repack_transform = _transforms.compose(data_config.repack_transforms.inputs)

    # ===== Part 1: Action Prediction MSE via Inference =====
    if not args.skip_inference:
        print(f"\nComputing action prediction MSE on {args.num_samples} samples...")

        all_mse = []
        all_mae = []
        per_dim_errors = []

        indices = range(args.start_idx, min(args.start_idx + args.num_samples, len(dataset)))

        for idx in tqdm(indices, desc="Inference"):
            # Get raw sample from dataset
            raw_sample = dataset[idx]

            # Add prompt
            sample_with_prompt = prompt_transform(raw_sample)

            # Repack to policy input format
            sample = repack_transform(sample_with_prompt)

            # Get ground truth actions (first action in the horizon)
            gt_actions = np.asarray(sample["actions"])  # Shape: (horizon, action_dim)

            # Run inference
            result = policy.infer(sample)
            pred_actions = np.asarray(result["actions"])  # Shape: (horizon, action_dim)

            # Compute metrics (only on first 7 dims - the actual action dims for Bridge)
            action_dim = min(7, gt_actions.shape[-1], pred_actions.shape[-1])
            gt_trimmed = gt_actions[:, :action_dim]
            pred_trimmed = pred_actions[:, :action_dim]

            mse = np.mean((pred_trimmed - gt_trimmed) ** 2)
            mae = np.mean(np.abs(pred_trimmed - gt_trimmed))
            per_dim = np.mean((pred_trimmed - gt_trimmed) ** 2, axis=0)

            all_mse.append(mse)
            all_mae.append(mae)
            per_dim_errors.append(per_dim)

        mean_mse = np.mean(all_mse)
        std_mse = np.std(all_mse)
        mean_mae = np.mean(all_mae)
        mean_per_dim = np.mean(per_dim_errors, axis=0)

    # ===== Part 2: Flow Matching Loss =====
    print("\nComputing flow matching loss on evaluation data...")
    import jax
    key = jax.random.key(42)

    # Load model directly
    model = config.model.load(_model.restore_params(checkpoint_dir / "params"))

    # Reduce batch size for evaluation
    eval_config = dataclasses.replace(config, batch_size=8)
    num_batches = args.num_samples // 8

    losses = []
    loader = _data_loader.create_data_loader(
        eval_config,
        num_batches=num_batches,
        shuffle=False,
    )

    for i, (obs, gt_actions) in enumerate(tqdm(loader, total=num_batches, desc="Flow loss")):
        key, subkey = jax.random.split(key)
        loss = model.compute_loss(subkey, obs, gt_actions)
        losses.append(float(np.mean(loss)))

    mean_flow_loss = np.mean(losses)
    std_flow_loss = np.std(losses)

    # ===== Print Results =====
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Num samples: {args.num_samples}")
    print("-" * 60)
    print(f"Flow Matching Loss: {mean_flow_loss:.6f} +/- {std_flow_loss:.6f}")

    if not args.skip_inference:
        print("-" * 60)
        print(f"Action MSE: {mean_mse:.6f} +/- {std_mse:.6f}")
        print(f"Action MAE: {mean_mae:.6f}")
        print(f"Action RMSE: {np.sqrt(mean_mse):.6f}")
        print("-" * 60)
        print("Per-dimension MSE:")
        dim_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        for i, (name, val) in enumerate(zip(dim_names, mean_per_dim)):
            print(f"  {name}: {val:.6f}")

    print("=" * 60)

    # Clean up
    del model
    del policy

    return mean_flow_loss


if __name__ == "__main__":
    main()
