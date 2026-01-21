"""Visualize the flow matching denoising process.

Generates a GIF or images showing how actions evolve from noise to clean trajectories.

Usage:
    uv run scripts/viz_flow_matching.py --config pi05_bridge \
        --checkpoint-dir checkpoints/pi05_bridge/bridge_run_20k/20000 \
        --output-dir viz_output
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import imageio

from openpi.models import model as _model
from openpi.shared import download
from openpi.training import config as _config
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import openpi.transforms as _transforms


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


def plot_action_trajectory(actions, title, ax, action_dim=7):
    """Plot action trajectory as a line plot or arrows."""
    actions = actions[0]  # Remove batch dim
    horizon, dim = actions.shape
    dim = min(dim, action_dim)  # Only plot first 7 dims (actual actions)

    # Plot each dimension as a line
    colors = plt.cm.tab10(np.linspace(0, 1, dim))
    dim_names = ["x", "y", "z", "roll", "pitch", "yaw", "grip"][:dim]

    for d in range(dim):
        ax.plot(range(horizon), actions[:, d], color=colors[d], label=dim_names[d], linewidth=2, alpha=0.8)

    ax.set_xlabel("Time Step (action horizon)")
    ax.set_ylabel("Action Value")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, horizon - 1)


def plot_2d_trajectory(actions, title, ax):
    """Plot XY trajectory as arrows showing movement direction."""
    actions = actions[0]  # Remove batch dim
    horizon = actions.shape[0]

    # Use x, y actions (first 2 dims)
    x = np.cumsum(actions[:, 0])  # Cumulative for position
    y = np.cumsum(actions[:, 1])

    # Color by time
    colors = plt.cm.viridis(np.linspace(0, 1, horizon))

    # Plot trajectory
    for i in range(horizon - 1):
        ax.annotate(
            '', xy=(x[i + 1], y[i + 1]), xytext=(x[i], y[i]),
            arrowprops=dict(arrowstyle='->', color=colors[i], lw=2)
        )

    ax.scatter(x[0], y[0], c='green', s=100, zorder=5, label='Start')
    ax.scatter(x[-1], y[-1], c='red', s=100, zorder=5, label='End')

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def create_visualization(intermediates, output_dir: Path, create_gif=True):
    """Create visualization images and optionally a GIF."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select key timesteps: t=1.0, t=0.5, t=0.0
    key_steps = []
    for inter in intermediates:
        t = inter["t"]
        if abs(t - 1.0) < 0.05 or abs(t - 0.5) < 0.1 or abs(t - 0.0) < 0.05:
            key_steps.append(inter)

    # If we don't have exactly 3, select evenly spaced
    if len(key_steps) < 3:
        indices = [0, len(intermediates) // 2, -1]
        key_steps = [intermediates[i] for i in indices]

    # Create individual images for key timesteps
    fig_images = []
    titles = ["t=1.0: Pure Noise", "t=0.5: Intermediate (Drift Phase)", "t=0.0: Clean Trajectory"]

    for i, (step, title_suffix) in enumerate(zip(key_steps[:3], titles)):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        t = step["t"]
        x = step["x"]

        # Left: Line plot of all action dimensions
        plot_action_trajectory(x, f"{title_suffix}", axes[0])

        # Right: 2D XY trajectory
        plot_2d_trajectory(x, f"XY Trajectory @ t={t:.2f}", axes[1])

        plt.tight_layout()

        # Save individual image
        img_path = output_dir / f"step_{i}_t{t:.2f}.png"
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {img_path}", flush=True)

        # For GIF
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]  # RGB only
        fig_images.append(img)

        plt.close(fig)

    # Create combined figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, (step, title) in enumerate(zip(key_steps[:3], titles)):
        t = step["t"]
        x = step["x"]
        plot_action_trajectory(x, title, axes[0, i])
        plot_2d_trajectory(x, f"XY @ t={t:.2f}", axes[1, i])

    plt.suptitle("Flow Matching Denoising Process: Noise → Action", fontsize=16, fontweight='bold')
    plt.tight_layout()
    combined_path = output_dir / "denoising_combined.png"
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined: {combined_path}", flush=True)
    plt.close()

    # Create GIF from all intermediates
    if create_gif:
        gif_frames = []
        for step in intermediates:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            t = step["t"]
            x = step["x"]

            plot_action_trajectory(x, f"Denoising: t={t:.2f}", axes[0])
            plot_2d_trajectory(x, f"XY Trajectory @ t={t:.2f}", axes[1])

            plt.tight_layout()

            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]  # RGB only
            gif_frames.append(img)
            plt.close(fig)

        gif_path = output_dir / "denoising_process.gif"
        imageio.mimsave(gif_path, gif_frames, fps=2, loop=0)
        print(f"Saved GIF: {gif_path}", flush=True)


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
