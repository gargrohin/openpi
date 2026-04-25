"""
Simple PyBullet WidowX environment for testing Pi0.5 policies.

This is a basic simulation for sanity-checking - not as accurate as SIMPLER
but works on headless servers without Vulkan.

Usage:
    from scripts.pybullet_widowx_env import WidowXEnv

    env = WidowXEnv()
    obs = env.reset()
    for _ in range(100):
        action = policy.predict(obs)  # Your policy
        obs, reward, done, info = env.step(action)
    env.close()
"""

import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path
import urllib.request
import os


class WidowXEnv:
    """
    Simple WidowX robot environment in PyBullet.

    Action space: 7D (dx, dy, dz, droll, dpitch, dyaw, gripper)
    Observation: dict with 'state' (joint positions) and optionally 'image'
    """

    # WidowX joint limits (approximate)
    JOINT_LIMITS = {
        'lower': [-3.14, -1.88, -1.6, -1.75, -2.15, -3.14],
        'upper': [3.14, 1.99, 1.6, 1.75, 2.15, 3.14],
    }

    def __init__(
        self,
        render: bool = False,
        control_freq: int = 5,
        action_scale: float = 0.1,
        use_camera: bool = False,
        camera_width: int = 256,
        camera_height: int = 256,
    ):
        """
        Args:
            render: Whether to render GUI (set False for headless)
            control_freq: Control frequency in Hz
            action_scale: Scale factor for actions
            use_camera: Whether to return camera images in observations
            camera_width: Camera image width
            camera_height: Camera image height
        """
        self.render_mode = render
        self.control_freq = control_freq
        self.action_scale = action_scale
        self.use_camera = use_camera
        self.camera_width = camera_width
        self.camera_height = camera_height

        # Connect to PyBullet
        if render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)  # 240 Hz simulation

        self.robot = None
        self.objects = []
        self._setup_scene()

    def _setup_scene(self):
        """Set up the scene with table and robot."""
        # Ground plane
        p.loadURDF("plane.urdf")

        # Table
        table_pos = [0.5, 0, 0]
        table_size = [0.4, 0.6, 0.4]
        table_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=table_size, rgbaColor=[0.6, 0.4, 0.2, 1]
        )
        table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_size)
        self.table = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[table_pos[0], table_pos[1], table_size[2]],
        )

        # Load WidowX robot (use Kuka as stand-in if WidowX URDF not available)
        try:
            self.robot = self._load_widowx()
        except Exception as e:
            print(f"Could not load WidowX URDF: {e}")
            print("Using Kuka IIWA as stand-in robot")
            self.robot = p.loadURDF(
                "kuka_iiwa/model.urdf",
                basePosition=[0, 0, 0],
                useFixedBase=True
            )

        self.num_joints = p.getNumJoints(self.robot)
        self.arm_joints = list(range(min(6, self.num_joints)))  # First 6 joints for arm

        # Get end-effector link
        self.ee_link = self.num_joints - 1

        # Add some objects on the table
        self._add_objects()

    def _load_widowx(self):
        """Try to load WidowX URDF."""
        # Check for local URDF
        urdf_paths = [
            Path("/mnt/efs/rohingarg/cri/widowx_urdf/widowx.urdf"),
            Path("assets/widowx/widowx.urdf"),
            Path.home() / ".cache/widowx/widowx.urdf",
        ]

        for path in urdf_paths:
            if path.exists():
                return p.loadURDF(str(path), basePosition=[0, 0, 0], useFixedBase=True)

        # Download WidowX URDF from interbotix
        urdf_url = "https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/wx250s.urdf"
        cache_dir = Path.home() / ".cache/widowx"
        cache_dir.mkdir(parents=True, exist_ok=True)
        urdf_path = cache_dir / "wx250s.urdf"

        if not urdf_path.exists():
            print(f"Downloading WidowX URDF to {urdf_path}...")
            try:
                urllib.request.urlretrieve(urdf_url, urdf_path)
            except Exception as e:
                raise RuntimeError(f"Could not download URDF: {e}")

        return p.loadURDF(str(urdf_path), basePosition=[0, 0, 0], useFixedBase=True)

    def _add_objects(self):
        """Add some simple objects to the scene."""
        # Red cube
        cube_size = 0.03
        cube_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[cube_size]*3, rgbaColor=[1, 0, 0, 1]
        )
        cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_size]*3)
        cube = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=[0.5, 0.1, 0.85],
        )
        self.objects.append(('red_cube', cube))

        # Green sphere
        sphere_radius = 0.025
        sphere_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=[0, 1, 0, 1]
        )
        sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius)
        sphere = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=sphere_collision,
            baseVisualShapeIndex=sphere_visual,
            basePosition=[0.5, -0.1, 0.85],
        )
        self.objects.append(('green_sphere', sphere))

        # Blue bowl (as a cylinder)
        bowl_visual = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.05, length=0.03, rgbaColor=[0, 0, 1, 1]
        )
        bowl_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=0.03)
        bowl = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=bowl_collision,
            baseVisualShapeIndex=bowl_visual,
            basePosition=[0.4, 0, 0.82],
        )
        self.objects.append(('blue_bowl', bowl))

    def reset(self, seed=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        # Reset robot to home position
        home_positions = [0, -0.5, 0.5, 0, 0.5, 0][:len(self.arm_joints)]
        for i, joint_idx in enumerate(self.arm_joints):
            p.resetJointState(self.robot, joint_idx, home_positions[i])

        # Reset objects
        p.resetBasePositionAndOrientation(
            self.objects[0][1], [0.5, 0.1, 0.85], [0, 0, 0, 1]
        )
        p.resetBasePositionAndOrientation(
            self.objects[1][1], [0.5, -0.1, 0.85], [0, 0, 0, 1]
        )

        # Step simulation to settle
        for _ in range(100):
            p.stepSimulation()

        return self._get_observation(), {}

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: 7D array [dx, dy, dz, droll, dpitch, dyaw, gripper]
                   Actions are delta end-effector movements

        Returns:
            observation, reward, terminated, truncated, info
        """
        action = np.array(action).flatten()

        # Scale action
        delta_pos = action[:3] * self.action_scale
        delta_rot = action[3:6] * self.action_scale * 0.5
        gripper = action[6] if len(action) > 6 else 0.5

        # Get current end-effector pose
        ee_state = p.getLinkState(self.robot, self.ee_link)
        current_pos = np.array(ee_state[0])
        current_orn = np.array(ee_state[1])

        # Compute target pose
        target_pos = current_pos + delta_pos

        # Simple IK to get joint targets
        target_joints = p.calculateInverseKinematics(
            self.robot,
            self.ee_link,
            target_pos,
            maxNumIterations=50,
        )

        # Apply joint targets
        for i, joint_idx in enumerate(self.arm_joints):
            if i < len(target_joints):
                p.setJointMotorControl2(
                    self.robot,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=target_joints[i],
                    force=100,
                )

        # Step simulation
        steps_per_control = int(240 / self.control_freq)
        for _ in range(steps_per_control):
            p.stepSimulation()

        # Get observation
        obs = self._get_observation()

        # Simple reward: negative distance to red cube
        cube_pos, _ = p.getBasePositionAndOrientation(self.objects[0][1])
        ee_pos = np.array(p.getLinkState(self.robot, self.ee_link)[0])
        distance = np.linalg.norm(ee_pos - np.array(cube_pos))
        reward = -distance

        # Check termination
        terminated = False
        truncated = False
        info = {'ee_pos': ee_pos, 'cube_pos': cube_pos, 'distance': distance}

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Get current observation."""
        # Joint positions
        joint_states = [p.getJointState(self.robot, i)[0] for i in self.arm_joints]

        # End-effector pose
        ee_state = p.getLinkState(self.robot, self.ee_link)
        ee_pos = np.array(ee_state[0])
        ee_orn = np.array(p.getEulerFromQuaternion(ee_state[1]))

        # Gripper state (dummy for now)
        gripper_state = 0.5

        # State vector (similar to Bridge format)
        state = np.concatenate([
            joint_states,
            [gripper_state, gripper_state]  # Pad to 8D like Bridge
        ])[:8]

        obs = {'state': state.astype(np.float32)}

        # Camera image if requested
        if self.use_camera:
            obs['image'] = self._get_camera_image()

        return obs

    def _get_camera_image(self):
        """Render camera image."""
        # Camera positioned to look at table
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.8, 0, 1.2],
            cameraTargetPosition=[0.5, 0, 0.8],
            cameraUpVector=[0, 0, 1],
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.camera_width / self.camera_height,
            nearVal=0.1,
            farVal=2.0,
        )

        _, _, rgb, _, _ = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,
        )

        # Convert to numpy array (H, W, 4) -> (H, W, 3)
        rgb = np.array(rgb, dtype=np.uint8).reshape(
            self.camera_height, self.camera_width, 4
        )[:, :, :3]

        return rgb

    def render(self):
        """Render the environment (only works in GUI mode)."""
        if self.use_camera:
            return self._get_camera_image()
        return None

    def close(self):
        """Close the environment."""
        p.disconnect(self.client)

    @property
    def action_space(self):
        """Return action space info."""
        class ActionSpace:
            shape = (7,)
            low = -np.ones(7)
            high = np.ones(7)

            @staticmethod
            def sample():
                return np.random.uniform(-1, 1, 7)

        return ActionSpace()


def test_env():
    """Test the environment."""
    print("Creating WidowX environment...")
    env = WidowXEnv(render=False, use_camera=True)

    print(f"Robot has {env.num_joints} joints")
    print(f"Action space: {env.action_space.shape}")

    obs, _ = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"State shape: {obs['state'].shape}")
    if 'image' in obs:
        print(f"Image shape: {obs['image'].shape}")

    print("\nRunning 50 random steps...")
    total_reward = 0
    for i in range(50):
        action = env.action_space.sample() * 0.5
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if i % 10 == 0:
            print(f"  Step {i}: reward={reward:.3f}, distance={info['distance']:.3f}")

    print(f"\nTotal reward: {total_reward:.3f}")

    # Save a camera image
    if env.use_camera:
        import matplotlib.pyplot as plt
        img = env._get_camera_image()
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title("PyBullet WidowX Environment")
        plt.axis('off')
        plt.savefig('viz_output/pybullet_widowx_test.png', dpi=100)
        print("Saved: viz_output/pybullet_widowx_test.png")

    env.close()
    print("\nTest passed!")


if __name__ == "__main__":
    test_env()
