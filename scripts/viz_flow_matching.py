"""Visualize the flow matching denoising process.

Generates images showing how actions evolve from noise to clean trajectories.

Usage:
    uv run scripts/viz_flow_matching.py --config pi05_bridge \
        --checkpoint-dir checkpoints/pi05_bridge/bridge_run1_20k/19999 \
        --output-dir viz_output

Can also be imported as a module for notebook use:
    from scripts.viz_flow_matching import load_sample_direct, load_model, ...
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import av  # For video decoding

from openpi.models import model as _model
from openpi.shared import download
from openpi.training import config as _config
import openpi.transforms as _transforms


# ============ Direct Data Loading Utilities ============
# These bypass LeRobotDataset for fast single-sample loading

def get_data_root(repo_id: str = "IPEC-COMMUNITY/bridge_orig_lerobot") -> Path:
    """Get the local data root for the dataset."""
    # Check common locations
    hf_cache = Path.home() / ".cache/huggingface/hub" / f"datasets--{repo_id.replace('/', '--')}"
    if hf_cache.exists():
        # Follow symlink if present
        snapshots = hf_cache / "snapshots"
        if snapshots.exists():
            for d in snapshots.iterdir():
                if d.is_dir() or d.is_symlink():
                    target = d.resolve() if d.is_symlink() else d
                    if (target / "meta").exists():
                        return target

    # Try direct path (for local datasets)
    local_path = Path("/mnt/efs/rohingarg/cri/bridge_orig_lerobot")
    if local_path.exists():
        return local_path

    raise FileNotFoundError(f"Could not find dataset root for {repo_id}")


def load_metadata(data_root: Path) -> dict:
    """Load dataset metadata (info.json)."""
    with open(data_root / "meta" / "info.json") as f:
        return json.load(f)


def load_tasks(data_root: Path) -> dict[int, str]:
    """Load task index -> task string mapping."""
    tasks = {}
    with open(data_root / "meta" / "tasks.jsonl") as f:
        for line in f:
            item = json.loads(line)
            tasks[item["task_index"]] = item["task"]
    return tasks


def decode_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    """Decode a single frame from a video file."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]

    # Seek to approximate position
    target_pts = int(frame_idx * stream.duration / stream.frames) if stream.frames else 0
    container.seek(target_pts, stream=stream)

    # Decode frames until we get the one we want
    for i, frame in enumerate(container.decode(video=0)):
        if i >= frame_idx or frame.pts >= target_pts:
            img = frame.to_ndarray(format='rgb24')
            container.close()
            return img

    container.close()
    raise ValueError(f"Could not decode frame {frame_idx} from {video_path}")


def load_sample_direct(
    episode_idx: int,
    frame_idx: int = 0,
    data_root: Path | None = None,
    action_horizon: int = 50,
) -> dict:
    """
    Load a single sample directly from parquet + video files.

    This is MUCH faster than using LeRobotDataset for single samples.

    Args:
        episode_idx: Episode index (0-indexed)
        frame_idx: Frame index within episode (0-indexed)
        data_root: Path to dataset root (auto-detected if None)
        action_horizon: Number of future action steps to load

    Returns:
        dict with keys: images (dict), state, actions, prompt
    """
    if data_root is None:
        data_root = get_data_root()

    info = load_metadata(data_root)
    tasks = load_tasks(data_root)

    # Calculate chunk
    chunk_size = info["chunks_size"]
    chunk_idx = episode_idx // chunk_size

    # Load parquet
    parquet_path = data_root / f"data/chunk-{chunk_idx:03d}/episode_{episode_idx:06d}.parquet"
    df = pd.read_parquet(parquet_path)

    if frame_idx >= len(df):
        raise ValueError(f"frame_idx {frame_idx} >= episode length {len(df)}")

    row = df.iloc[frame_idx]

    # Get state and actions
    state = np.array(row["observation.state"])

    # Get action sequence (current + future frames)
    actions = []
    for i in range(action_horizon):
        if frame_idx + i < len(df):
            actions.append(np.array(df.iloc[frame_idx + i]["action"]))
        else:
            # Pad with last action if we run out
            actions.append(np.array(df.iloc[-1]["action"]))
    actions = np.stack(actions)

    # Get task/prompt
    task_idx = int(row["task_index"])
    prompt = tasks.get(task_idx, "")

    # Load images from videos
    images = {}
    for key, feat in info["features"].items():
        if feat["dtype"] == "video" and key.startswith("observation.images"):
            img_key = key.replace("observation.images.", "")
            video_path = data_root / f"videos/chunk-{chunk_idx:03d}/{key}/episode_{episode_idx:06d}.mp4"
            if video_path.exists():
                img = decode_video_frame(video_path, frame_idx)
                images[img_key] = img

    return {
        "images": images,
        "state": state,
        "actions": actions,
        "prompt": prompt,
        "episode_idx": episode_idx,
        "frame_idx": frame_idx,
    }


def load_model(config_name: str, checkpoint_dir: str):
    """Load model from config and checkpoint."""
    config = _config.get_config(config_name)

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        checkpoint_path = download.maybe_download(checkpoint_dir)

    model = config.model.load(_model.restore_params(checkpoint_path / "params"))
    return model, config


def prepare_observation_from_sample(
    sample: dict,
    config_name: str = "pi05_bridge",
) -> _model.Observation:
    """
    Convert a raw sample dict to model Observation.

    Args:
        sample: Output from load_sample_direct()
        config_name: Config to use for transforms

    Returns:
        Observation ready for model inference
    """
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # Format to match what repack_transform expects:
    # - observation.images.image_X
    # - observation.state
    # - action (singular, not actions)
    # - prompt
    formatted = {
        "prompt": sample["prompt"],
        "observation.state": sample["state"],
        "action": sample["actions"],  # repack expects 'action' singular
    }

    # Add images with observation.images.X format
    for img_key, img in sample["images"].items():
        formatted[f"observation.images.{img_key}"] = img

    # Apply transforms
    repack_transform = _transforms.compose(data_config.repack_transforms.inputs)
    data_transform = _transforms.compose(data_config.data_transforms.inputs)
    model_transform = _transforms.compose(data_config.model_transforms.inputs)

    processed = repack_transform(formatted)
    processed = data_transform(processed)

    # Normalize
    if data_config.norm_stats:
        norm_transform = _transforms.Normalize(
            data_config.norm_stats,
            use_quantiles=data_config.use_quantile_norm
        )
        processed = norm_transform(processed)

    processed = model_transform(processed)

    # Convert to observation
    def add_batch_dim_and_convert(x):
        if isinstance(x, dict):
            return {k: add_batch_dim_and_convert(v) for k, v in x.items()}
        arr = np.expand_dims(np.asarray(x), 0)
        return jnp.array(arr)

    obs_dict = {k: add_batch_dim_and_convert(v) for k, v in processed.items() if k != "actions"}
    return _model.Observation.from_dict(obs_dict)


# ============ Keep original imports for backward compatibility ============
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset


# ============ Enhanced Visualization: Actions on Images ============

def visualize_action_on_image(
    sample: dict,
    actions: np.ndarray,
    title: str = "Predicted Action Trajectory",
    save_path: str | None = None,
):
    """
    Visualize predicted actions overlaid on input images.

    Shows:
    - Input camera views
    - 2D projection of XY movement (top-down view)
    - Side view (XZ) of trajectory
    - Gripper state over time

    Args:
        sample: Output from load_sample_direct()
        actions: Predicted actions [horizon, action_dim] or [batch, horizon, action_dim]
        title: Plot title
        save_path: Optional path to save figure
    """
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection

    # Handle batch dimension
    if actions.ndim == 3:
        actions = actions[0]

    # Extract action components (assuming Bridge format: x, y, z, roll, pitch, yaw, gripper)
    # Actions are deltas, so cumsum gives trajectory
    x = np.cumsum(actions[:, 0])
    y = np.cumsum(actions[:, 1])
    z = np.cumsum(actions[:, 2])
    gripper = actions[:, 6] if actions.shape[1] > 6 else np.zeros(len(actions))

    # Create figure
    fig = plt.figure(figsize=(16, 10))

    # Get images - sort keys for consistent ordering (camera_0, camera_1, camera_2)
    images = sample.get("images", {})
    sorted_keys = sorted(images.keys())
    n_images = min(len(sorted_keys), 3)

    # Top row: Camera views
    for i, key in enumerate(sorted_keys[:3]):
        img = images[key]
        ax = fig.add_subplot(2, 4, i + 1)
        ax.imshow(img)
        ax.set_title(f"Camera: {key}", fontsize=10)
        ax.axis('off')

        # Add arrow showing XY direction on image 1 (usually the main camera with content)
        if i == 1 and len(x) > 1:
            # Normalize and scale arrow for visibility
            dx_total = x[-1] - x[0]
            dy_total = y[-1] - y[0]
            magnitude = np.sqrt(dx_total**2 + dy_total**2)
            if magnitude > 0.001:
                # Draw arrow from center of image
                h, w = img.shape[:2]
                cx, cy = w // 2, h // 2
                scale = min(w, h) // 4
                ax.annotate('',
                    xy=(cx + dx_total/magnitude * scale, cy - dy_total/magnitude * scale),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3),
                )
                ax.text(cx, cy + 30, 'XY movement', color='red', fontsize=8, ha='center')

    # Add prompt
    ax = fig.add_subplot(2, 4, 4)
    ax.text(0.5, 0.5, f"Task:\n\n\"{sample.get('prompt', 'N/A')}\"",
            ha='center', va='center', fontsize=12, wrap=True,
            transform=ax.transAxes)
    ax.axis('off')
    ax.set_title("Task Description", fontsize=10)

    # Bottom left: Top-down view (XY trajectory)
    ax = fig.add_subplot(2, 4, 5)

    # Color by time (early=blue, late=red)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(segments)))
    lc = LineCollection(segments, colors=colors, linewidth=2)
    ax.add_collection(lc)

    ax.scatter(x[0], y[0], c='green', s=100, marker='o', zorder=5, label='Start')
    ax.scatter(x[-1], y[-1], c='red', s=100, marker='^', zorder=5, label='End')
    ax.set_xlabel('X (forward/back)')
    ax.set_ylabel('Y (left/right)')
    ax.set_title('Top-Down View (XY)', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Set reasonable axis limits
    margin = max(0.05, max(abs(x).max(), abs(y).max()) * 0.2)
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.min() - margin, y.max() + margin)

    # Bottom middle: Side view (XZ trajectory)
    ax = fig.add_subplot(2, 4, 6)

    points = np.array([x, z]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=colors, linewidth=2)
    ax.add_collection(lc)

    ax.scatter(x[0], z[0], c='green', s=100, marker='o', zorder=5)
    ax.scatter(x[-1], z[-1], c='red', s=100, marker='^', zorder=5)
    ax.set_xlabel('X (forward/back)')
    ax.set_ylabel('Z (up/down)')
    ax.set_title('Side View (XZ)', fontsize=10)
    ax.grid(True, alpha=0.3)

    margin = max(0.05, max(abs(x).max(), abs(z).max()) * 0.2)
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(z.min() - margin, z.max() + margin)

    # Bottom right: Gripper + action magnitudes over time
    ax = fig.add_subplot(2, 4, 7)

    time_steps = np.arange(len(actions))
    ax.plot(time_steps, gripper, 'purple', linewidth=2, label='Gripper')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Gripper (0=closed, 1=open)')
    ax.set_title('Gripper State', fontsize=10)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 3D trajectory
    ax = fig.add_subplot(2, 4, 8, projection='3d')
    ax.plot(x, y, z, 'b-', linewidth=2)
    ax.scatter(x[0], y[0], z[0], c='green', s=100, marker='o')
    ax.scatter(x[-1], y[-1], z[-1], c='red', s=100, marker='^')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory', fontsize=10)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def visualize_denoising_on_image(
    sample: dict,
    intermediates: list[dict],
    timesteps: list[float] = [1.0, 0.5, 0.2, 0.0],
    save_path: str | None = None,
):
    """
    Visualize denoising process with trajectories overlaid.

    Shows how the predicted trajectory evolves from noise to clean.
    """
    # Find closest timesteps
    times = [inter["t"] for inter in intermediates]
    selected = []
    for t in timesteps:
        idx = np.argmin(np.abs(np.array(times) - t))
        selected.append(intermediates[idx])

    n_steps = len(selected)
    fig = plt.figure(figsize=(5 * n_steps, 8))

    # Get one image for reference - prefer camera_1 which usually has content
    images = sample.get("images", {})
    sorted_keys = sorted(images.keys())
    # Use camera_1 if available, otherwise first available
    ref_key = sorted_keys[1] if len(sorted_keys) > 1 else (sorted_keys[0] if sorted_keys else None)
    ref_img = images[ref_key] if ref_key else np.zeros((256, 256, 3))

    for i, inter in enumerate(selected):
        actions = inter["x"][0] if inter["x"].ndim == 3 else inter["x"]
        t = inter["t"]

        # Trajectory
        x = np.cumsum(actions[:, 0])
        y = np.cumsum(actions[:, 1])
        z = np.cumsum(actions[:, 2])

        # Top: Image with XY arrow
        ax = fig.add_subplot(2, n_steps, i + 1)
        ax.imshow(ref_img)

        # Draw trajectory projection
        h, w = ref_img.shape[:2]
        cx, cy = w // 2, h // 2

        # Scale trajectory to image coordinates
        scale = min(w, h) // 3
        x_img = cx + x * scale * 10  # Scale factor for visibility
        y_img = cy - y * scale * 10

        # Clip to image bounds
        x_img = np.clip(x_img, 10, w - 10)
        y_img = np.clip(y_img, 10, h - 10)

        # Color by noise level
        color = plt.cm.coolwarm(1 - t)
        ax.plot(x_img, y_img, color=color, linewidth=2, alpha=0.8)
        ax.scatter(x_img[0], y_img[0], c='green', s=50, zorder=5)
        ax.scatter(x_img[-1], y_img[-1], c='red', s=50, marker='^', zorder=5)

        ax.set_title(f't={t:.2f}', fontsize=12, fontweight='bold')
        ax.axis('off')

        # Bottom: XY trajectory plot
        ax = fig.add_subplot(2, n_steps, n_steps + i + 1)
        ax.plot(x, y, color=color, linewidth=2)
        ax.scatter(x[0], y[0], c='green', s=80, marker='o')
        ax.scatter(x[-1], y[-1], c='red', s=80, marker='^')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Consistent axis limits
        all_x = np.concatenate([inter["x"][0, :, 0].cumsum() for inter in selected])
        all_y = np.concatenate([inter["x"][0, :, 1].cumsum() for inter in selected])
        margin = max(0.1, max(abs(all_x).max(), abs(all_y).max()) * 0.2)
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)

    plt.suptitle(f'Denoising: Noise → Clean\nTask: "{sample.get("prompt", "")}"',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize flow matching denoising")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="viz_output")
    parser.add_argument("--sample-idx", type=int, default=0, help="Dataset sample index")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_actions_with_intermediates(
    model,
    observation: _model.Observation,
    rng,
    num_steps: int = 10,
):
    """Modified sampling that returns intermediate x_t at each timestep."""
    from openpi.models.pi0 import make_attn_mask
    import einops

    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))

    # Fill KV cache
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    # Store intermediates
    intermediates = []
    x_t = noise
    time = 1.0

    intermediates.append({"t": float(time), "x": np.array(x_t)})

    for step in range(num_steps):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_step = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask_step, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])

        x_t = x_t + dt * v_t
        time = time + dt

        intermediates.append({"t": float(time), "x": np.array(x_t)})

    return intermediates


def plot_3d_trajectory(actions, ax, title, color='blue', alpha=1.0):
    """Plot XYZ trajectory in 3D space."""
    actions = actions[0]  # Remove batch dim

    # Cumulative sum to get positions from deltas
    x = np.cumsum(actions[:, 0])
    y = np.cumsum(actions[:, 1])
    z = np.cumsum(actions[:, 2])

    # Plot trajectory
    ax.plot(x, y, z, color=color, alpha=alpha, linewidth=2)
    ax.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='red', s=100, marker='^', label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=12, fontweight='bold')


def plot_heatmap(actions, ax, title):
    """Plot actions as a heatmap (time x dimension)."""
    actions = actions[0, :, :7]  # Remove batch, keep first 7 dims

    im = ax.imshow(actions.T, aspect='auto', cmap='RdBu_r',
                   vmin=-1.5, vmax=1.5, interpolation='nearest')

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Action Dimension')
    ax.set_yticks(range(7))
    ax.set_yticklabels(['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'grip'])
    ax.set_title(title, fontsize=12, fontweight='bold')

    return im


def plot_action_variance(intermediates, ax):
    """Plot how variance decreases during denoising."""
    times = [inter["t"] for inter in intermediates]
    variances = [np.var(inter["x"][0, :, :7]) for inter in intermediates]

    ax.plot(times, variances, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Time t (1=noise, 0=clean)')
    ax.set_ylabel('Action Variance')
    ax.set_title('Variance Reduction During Denoising', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # t goes from 1 to 0


def create_visualization(intermediates, output_dir: Path):
    """Create improved visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select key timesteps
    t1 = intermediates[0]   # t=1.0 (noise)
    t05 = intermediates[len(intermediates)//2]  # t≈0.5 (mid)
    # Find t≈0.2 (where most denoising happens)
    t02_idx = int(0.8 * len(intermediates))  # 80% through = t≈0.2
    t02 = intermediates[t02_idx]
    t0 = intermediates[-1]  # t=0.0 (clean)

    # ============ Figure 1: 3D Trajectory Evolution ============
    fig = plt.figure(figsize=(20, 5))

    ax1 = fig.add_subplot(141, projection='3d')
    plot_3d_trajectory(t1["x"], ax1, f"t=1.0: Pure Noise", color='gray')

    ax2 = fig.add_subplot(142, projection='3d')
    plot_3d_trajectory(t05["x"], ax2, f"t={t05['t']:.1f}: Intermediate", color='orange')

    ax3 = fig.add_subplot(143, projection='3d')
    plot_3d_trajectory(t02["x"], ax3, f"t={t02['t']:.1f}: Late Stage", color='purple')

    ax4 = fig.add_subplot(144, projection='3d')
    plot_3d_trajectory(t0["x"], ax4, f"t=0.0: Clean Trajectory", color='blue')

    plt.suptitle('Flow Matching: 3D End-Effector Trajectory (XYZ)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_3d.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/trajectory_3d.png", flush=True)
    plt.close()

    # ============ Figure 2: Heatmap Evolution ============
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    im = plot_heatmap(t1["x"], axes[0], "t=1.0: Noise")
    plot_heatmap(t05["x"], axes[1], f"t={t05['t']:.1f}: Intermediate")
    plot_heatmap(t02["x"], axes[2], f"t={t02['t']:.1f}: Late Stage")
    plot_heatmap(t0["x"], axes[3], "t=0.0: Clean")

    # Variance plot
    plot_action_variance(intermediates, axes[4])

    # Add colorbar
    fig.colorbar(im, ax=axes[:4], shrink=0.8, label='Action Value')

    plt.suptitle('Flow Matching: Action Heatmaps (Denoising Sharpens Structure)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_evolution.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/heatmap_evolution.png", flush=True)
    plt.close()

    # ============ Figure 3: Per-Dimension Comparison ============
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']

    for i, name in enumerate(dim_names):
        row, col = i // 4, i % 4
        ax = axes[row, col]

        horizon = t1["x"].shape[1]
        time_steps = np.arange(horizon)

        ax.plot(time_steps, t1["x"][0, :, i], 'gray', alpha=0.5, label='t=1 (noise)', linewidth=1)
        ax.plot(time_steps, t05["x"][0, :, i], 'orange', alpha=0.7, label=f't={t05["t"]:.1f}', linewidth=1.5)
        ax.plot(time_steps, t02["x"][0, :, i], 'purple', alpha=0.8, label=f't={t02["t"]:.1f}', linewidth=1.5)
        ax.plot(time_steps, t0["x"][0, :, i], 'blue', label='t=0 (clean)', linewidth=2)

        ax.set_title(f'{name.upper()}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2, 2)

        if i == 0:
            ax.legend(fontsize=8)

    # Hide the 8th subplot
    axes[1, 3].axis('off')

    plt.suptitle('Flow Matching: Per-Dimension Denoising (Gray→Orange→Purple→Blue)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "per_dimension.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/per_dimension.png", flush=True)
    plt.close()

    # ============ Figure 4: Single Combined Overview ============
    fig = plt.figure(figsize=(18, 10))

    # 3D trajectories on top row
    ax1 = fig.add_subplot(241, projection='3d')
    plot_3d_trajectory(t1["x"], ax1, "t=1.0: Noise", color='gray')

    ax2 = fig.add_subplot(242, projection='3d')
    plot_3d_trajectory(t05["x"], ax2, f"t={t05['t']:.1f}: Mid", color='orange')

    ax3 = fig.add_subplot(243, projection='3d')
    plot_3d_trajectory(t02["x"], ax3, f"t={t02['t']:.1f}: Late", color='purple')

    ax4 = fig.add_subplot(244, projection='3d')
    plot_3d_trajectory(t0["x"], ax4, "t=0.0: Clean", color='blue')

    # Heatmaps on bottom row
    ax5 = fig.add_subplot(245)
    plot_heatmap(t1["x"], ax5, "Noise")

    ax6 = fig.add_subplot(246)
    plot_heatmap(t05["x"], ax6, "Intermediate")

    ax7 = fig.add_subplot(247)
    plot_heatmap(t02["x"], ax7, "Late Stage")

    ax8 = fig.add_subplot(248)
    im = plot_heatmap(t0["x"], ax8, "Clean")

    plt.suptitle('Flow Matching Denoising: Noise → Clean Action Trajectory', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "overview.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/overview.png", flush=True)
    plt.close()

    # ============ GIF: 3D trajectory animation ============
    gif_frames = []
    for inter in intermediates:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        t = inter["t"]
        color = plt.cm.coolwarm(1 - t)  # Blue at t=0, Red at t=1
        plot_3d_trajectory(inter["x"], ax, f"Denoising: t={t:.2f}", color=color)

        # Keep consistent axis limits
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)

        plt.tight_layout()
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        gif_frames.append(img)
        plt.close(fig)

    gif_path = output_dir / "denoising_3d.gif"
    imageio.mimsave(gif_path, gif_frames, fps=2, loop=0)
    print(f"Saved: {gif_path}", flush=True)


def main():
    args = parse_args()

    print(f"Loading config: {args.config}", flush=True)
    config = _config.get_config(args.config)
    data_config = config.data.create(config.assets_dirs, config.model)

    print(f"Loading checkpoint: {args.checkpoint_dir}", flush=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        checkpoint_dir = download.maybe_download(args.checkpoint_dir)

    # Load model
    print("Loading model...", flush=True)
    model = config.model.load(_model.restore_params(checkpoint_dir / "params"))

    # Load a sample observation
    print(f"Loading sample {args.sample_idx} from dataset...", flush=True)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(config.model.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    # Get sample and transform
    raw_sample = dataset[args.sample_idx]
    prompt_transform = _transforms.PromptFromLeRobotTask(dataset_meta.tasks)
    sample = prompt_transform(raw_sample)
    repack_transform = _transforms.compose(data_config.repack_transforms.inputs)
    sample = repack_transform(sample)

    # Apply data transforms and model transforms to get observation
    data_transform = _transforms.compose(data_config.data_transforms.inputs)
    sample = data_transform(sample)

    # Normalize
    if data_config.norm_stats:
        norm_transform = _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
        sample = norm_transform(sample)

    # Apply model transforms
    model_transform = _transforms.compose(data_config.model_transforms.inputs)
    sample = model_transform(sample)

    # Create observation object - need to handle nested dicts properly and convert to JAX arrays
    def add_batch_dim_and_convert(x):
        if isinstance(x, dict):
            return {k: add_batch_dim_and_convert(v) for k, v in x.items()}
        arr = np.expand_dims(np.asarray(x), 0)
        return jnp.array(arr)  # Convert to JAX array

    obs_dict = {k: add_batch_dim_and_convert(v) for k, v in sample.items() if k != "actions"}
    observation = _model.Observation.from_dict(obs_dict)

    # Run sampling with intermediates
    print(f"Running flow matching with {args.num_steps} steps...", flush=True)
    rng = jax.random.key(args.seed)
    intermediates = sample_actions_with_intermediates(model, observation, rng, num_steps=args.num_steps)

    print(f"Captured {len(intermediates)} intermediate states", flush=True)

    # Create visualizations
    output_dir = Path(args.output_dir)
    create_visualization(intermediates, output_dir)

    print(f"\nVisualization complete! Check {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
