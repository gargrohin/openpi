"""
Evaluate Pi0.5 policy in PyBullet simulation.

This provides a basic sanity check for the policy - not as accurate as SIMPLER
but works on headless servers.

Usage:
    uv run scripts/eval_pybullet.py --config pi05_bridge \
        --checkpoint-dir checkpoints/pi05_bridge/bridge_run1_20k/19999 \
        --num-episodes 5
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pybullet_widowx_env import WidowXEnv
from viz_flow_matching import (
    load_model,
    visualize_action_on_image,
)
from openpi.models import model as _model
from openpi.training import config as _config
import openpi.transforms as _transforms


def create_observation_from_pybullet(
    pybullet_obs: dict,
    prompt: str,
    config_name: str = "pi05_bridge",
) -> _model.Observation:
    """
    Convert PyBullet observation to OpenPi Observation format.

    Note: This is a simplified version - real deployment would need
    proper camera calibration and image preprocessing.
    """
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # Get state
    state = pybullet_obs['state']

    # Get image (if available)
    image = pybullet_obs.get('image', np.zeros((256, 256, 3), dtype=np.uint8))

    # Create sample dict matching expected format
    sample = {
        "prompt": prompt,
        "observation.state": state,
        "action": np.zeros((50, 7)),  # Dummy, not used for inference
        "observation.images.image_0": image,
        "observation.images.image_1": image,  # Duplicate for required cameras
        "observation.images.image_2": image,
    }

    # Apply transforms
    repack_transform = _transforms.compose(data_config.repack_transforms.inputs)
    data_transform = _transforms.compose(data_config.data_transforms.inputs)
    model_transform = _transforms.compose(data_config.model_transforms.inputs)

    processed = repack_transform(sample)
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


def run_episode(
    env: WidowXEnv,
    model,
    prompt: str,
    config_name: str,
    max_steps: int = 100,
    action_chunk_size: int = 5,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Run a single episode with the policy.

    Args:
        env: PyBullet environment
        model: Loaded Pi0.5 model
        prompt: Task description
        config_name: Config name for transforms
        max_steps: Maximum steps per episode
        action_chunk_size: How many actions to execute from each prediction
        seed: Random seed
        verbose: Print progress

    Returns:
        dict with episode statistics
    """
    rng = jax.random.key(seed)

    obs, _ = env.reset(seed=seed)
    total_reward = 0
    actions_taken = []
    observations = []

    step = 0
    while step < max_steps:
        # Convert PyBullet obs to model observation
        model_obs = create_observation_from_pybullet(obs, prompt, config_name)

        # Get action from model
        rng, sample_rng = jax.random.split(rng)
        actions = model.sample_actions(sample_rng, model_obs, num_steps=10)
        actions = np.array(actions[0])  # Remove batch dim [horizon, action_dim]

        # Execute action chunk
        for i in range(min(action_chunk_size, max_steps - step)):
            action = actions[i]
            actions_taken.append(action)

            obs, reward, done, truncated, info = env.step(action)
            observations.append(obs.copy())
            total_reward += reward
            step += 1

            if verbose and step % 20 == 0:
                print(f"  Step {step}: reward={reward:.3f}, distance={info['distance']:.3f}")

            if done or truncated:
                break

        if done or truncated:
            break

    return {
        'total_reward': total_reward,
        'num_steps': step,
        'actions': np.array(actions_taken),
        'final_distance': info.get('distance', 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pi0.5 in PyBullet")
    parser.add_argument("--config", type=str, default="pi05_bridge")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--prompt", type=str, default="pick up the red cube")
    parser.add_argument("--output-dir", type=str, default="viz_output/pybullet_eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint_dir}...")
    model, config = load_model(args.config, args.checkpoint_dir)
    print("Model loaded!")

    # Create environment
    print("Creating PyBullet environment...")
    env = WidowXEnv(render=False, use_camera=True)
    print(f"Environment created with {env.num_joints} joints")

    # Run episodes
    results = []
    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")
        print(f"Prompt: {args.prompt}")

        result = run_episode(
            env=env,
            model=model,
            prompt=args.prompt,
            config_name=args.config,
            max_steps=args.max_steps,
            seed=args.seed + ep,
        )

        results.append(result)
        print(f"Episode {ep + 1}: reward={result['total_reward']:.2f}, "
              f"steps={result['num_steps']}, distance={result['final_distance']:.3f}")

        # Save visualization of actions
        if ep == 0:
            # Create a fake sample for visualization
            obs = env._get_observation()
            sample = {
                'images': {'camera': env._get_camera_image()},
                'state': obs['state'],
                'prompt': args.prompt,
            }
            fig = visualize_action_on_image(
                sample, result['actions'],
                title=f"Episode {ep + 1}: {args.prompt}"
            )
            fig.savefig(output_dir / f"episode_{ep + 1}_actions.png", dpi=150)
            plt.close(fig)
            print(f"Saved: {output_dir}/episode_{ep + 1}_actions.png")

    # Summary
    print("\n=== Summary ===")
    rewards = [r['total_reward'] for r in results]
    distances = [r['final_distance'] for r in results]
    print(f"Mean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Mean final distance: {np.mean(distances):.3f} +/- {np.std(distances):.3f}")

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
