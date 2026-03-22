#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trajectory Visualization Tool for Autonomous Surgery (ACT/DETR-VAE models)

This module provides a robust set of tools to compare ground truth kinematics
trajectories with model-predicted trajectories.

Assumptions:
- Trajectories are represented as sequences of 8-DoF vectors per arm:
  [x, y, z, qx, qy, qz, qw, jaw].
- If action_dim == 16, it is assumed to be two arms concatenated (PSM1, PSM2).
- Data is provided as numpy arrays or torch tensors of shape (B, T, D) or (T, D).
  where B=batch size, T=sequence length, D=action dimension.

Features:
- Validates and un-pads mismatched sequences if necessary.
- Ensures quaternion continuity (handles sign flips q == -q).
- Plots individual dimension time-series.
- Plots 3D spatial paths.
- Computes and plots error metrics (L2 position error, angular orientation error).

Requirements:
    pip install numpy matplotlib scipy torch
"""

from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation as R
import typing
from typing import Optional, Union, Tuple, List


# -----------------------------------------------------------------------------
# Data Processing & Math Helpers
# -----------------------------------------------------------------------------

def ensure_numpy(data: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
    """Safely converts input data to a numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Normalizes a quaternion array of shape (..., 4) to unit length.
    Expects format [qx, qy, qz, qw].
    """
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    # prevent division by zero
    norm = np.where(norm == 0, 1.0, norm)
    return q / norm

def enforce_quaternion_continuity(quats: np.ndarray) -> np.ndarray:
    """
    Given a sequence of quaternions of shape (T, 4), ensures continuous
    shortest-path rotations by flipping signs of quaternions if the dot
    product with the previous quaternion is negative.
    """
    if quats.shape[0] < 2:
        return quats
    
    continuous_q = quats.copy()
    for i in range(1, len(continuous_q)):
        dot = np.sum(continuous_q[i-1] * continuous_q[i])
        if dot < 0:
            continuous_q[i] = -continuous_q[i]
    return continuous_q

def compute_position_error(gt_pos: np.ndarray, pred_pos: np.ndarray) -> np.ndarray:
    """Computes L2 distance between position sequences of shape (T, 3)."""
    return np.linalg.norm(gt_pos - pred_pos, axis=-1)

def compute_angular_error(gt_quat: np.ndarray, pred_quat: np.ndarray) -> np.ndarray:
    """
    Computes angle difference in radians between two quaternion sequences of shape (T, 4).
    """
    gt_q = normalize_quaternion(gt_quat)
    pr_q = normalize_quaternion(pred_quat)
    
    # Dot product clamped to [-1, 1] for safety
    dots = np.sum(gt_q * pr_q, axis=-1)
    dots = np.clip(abs(dots), -1.0, 1.0)  # Use built-in abs or np.abs to prevent overflow
    
    # Angle difference
    angles = 2 * np.arccos(dots)
    return angles


# -----------------------------------------------------------------------------
# Visualization Class
# -----------------------------------------------------------------------------

class TrajectoryVisualizer:
    def __init__(
        self,
        gt_trajectory: Union[np.ndarray, torch.Tensor],
        pred_trajectory: Union[np.ndarray, torch.Tensor],
        dt: float = 0.1,
        arm_names: Optional[List[str]] = None
    ):
        """
        Initializes the visualizer with ground truth and predicted trajectories.
        
        Args:
            gt_trajectory: Ground truth kinematics, shape (T, 8) or (T, 16).
            pred_trajectory: Predicted kinematics, shape (T, 8) or (T, 16).
            dt: Timestep duration (seconds) used for time axes.
            arm_names: Labels for the arms. Defaults to PSM1, PSM2.
        """
        self.gt_traj = ensure_numpy(gt_trajectory)
        self.pr_traj = ensure_numpy(pred_trajectory)
        self.dt = dt
        
        # Validation and Alignment
        if self.gt_traj.ndim != 2 or self.pr_traj.ndim != 2:
            raise ValueError(f"Trajectories must be 2D (T, D), got gt: {self.gt_traj.shape}, pr: {self.pr_traj.shape}")
        
        self.T = min(self.gt_traj.shape[0], self.pr_traj.shape[0])
        self.gt_traj = self.gt_traj[:self.T]
        self.pr_traj = self.pr_traj[:self.T]
        self.time_axis = np.arange(self.T) * self.dt
        
        self.D = self.gt_traj.shape[1]
        if self.D == 8:
            self.num_arms = 1
            self.arm_names = arm_names or ["PSM1"]
        elif self.D == 16:
            self.num_arms = 2
            self.arm_names = arm_names or ["PSM1", "PSM2"]
        else:
            print(f"Warning: Unexpected feature dimension D={self.D}. Defaulting to generic plotting.")
            self.num_arms = max(1, self.D // 8)
            self.arm_names = arm_names or [f"Arm_{i}" for i in range(self.num_arms)]

        self._preprocess()

    def _preprocess(self):
        """Applies normalization and continuity fixes to the quaternions."""
        # Process each arm's quaternions independently
        for arm_idx in range(self.num_arms):
            base_idx = arm_idx * 8
            if base_idx + 7 <= self.D:
                # Ground truth
                gt_quat = self.gt_traj[:, base_idx+3 : base_idx+7]
                self.gt_traj[:, base_idx+3 : base_idx+7] = enforce_quaternion_continuity(
                    normalize_quaternion(gt_quat)
                )
                
                # Predictions
                pr_quat = self.pr_traj[:, base_idx+3 : base_idx+7]
                self.pr_traj[:, base_idx+3 : base_idx+7] = enforce_quaternion_continuity(
                    normalize_quaternion(pr_quat)
                )

    def plot_dimension_timeseries(self, arm_idx: int = 0, save_path: Optional[str] = None):
        """
        Plots a time-series comparison of all components (XYZ, Qxyzw, Jaw) for a specific arm.
        """
        if arm_idx >= self.num_arms:
            raise ValueError(f"Invalid arm_idx {arm_idx} for num_arms {self.num_arms}")
            
        base = arm_idx * 8
        labels = ['X (m)', 'Y (m)', 'Z (m)', 'Qx', 'Qy', 'Qz', 'Qw', 'Jaw']
        
        fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True)
        fig.suptitle(f'Kinematics Time-Series Comparison: {self.arm_names[arm_idx]}', fontsize=16)
        
        for i, ax in enumerate(axes.flatten()):
            if base + i >= self.D:
                break
            
            gt_dim = self.gt_traj[:, base + i]
            pr_dim = self.pr_traj[:, base + i]
            
            ax.plot(self.time_axis, gt_dim, 'k-', linewidth=2, label='Ground Truth')
            ax.plot(self.time_axis, pr_dim, 'r--', linewidth=2, label='Predicted')
            
            ax.set_ylabel(labels[i] if i < len(labels) else f'Dim {i}')
            ax.grid(True, alpha=0.3)
            if i in [6, 7]:  # bottom row
                ax.set_xlabel('Time (s)')
            if i == 0:
                ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved timeseries plot to {save_path}")
        else:
            plt.show()

    def plot_3d_paths(self, save_path: Optional[str] = None):
        """
        Plots the 3D XYZ spatial paths for both arms (if available).
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('3D Cartesian Trajectory Paths', fontsize=16)
        
        colors = ['blue', 'green', 'magenta', 'cyan']
        
        for arm_idx in range(self.num_arms):
            base = arm_idx * 8
            if base + 3 > self.D:
                continue
                
            gt_pos = self.gt_traj[:, base:base+3]
            pr_pos = self.pr_traj[:, base:base+3]
            
            ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 
                    color=colors[arm_idx * 2 % len(colors)], linestyle='-', linewidth=2, 
                    label=f'{self.arm_names[arm_idx]} GT')
            
            ax.plot(pr_pos[:, 0], pr_pos[:, 1], pr_pos[:, 2], 
                    color=colors[(arm_idx * 2 + 1) % len(colors)], linestyle='--', linewidth=2, 
                    label=f'{self.arm_names[arm_idx]} Pred')
            
            # Start/End markers
            ax.scatter(*gt_pos[0], color='black', marker='o', s=50, label='Start' if arm_idx==0 else "")
            ax.scatter(*gt_pos[-1], color='black', marker='x', s=50, label='End' if arm_idx==0 else "")
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved 3D path plot to {save_path}")
        else:
            plt.show()

    def plot_metrics_overlay(self, save_path: Optional[str] = None):
        """
        Plots Position Error (L2) and Orientation Error (Angular diff) over time.
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('Trajectory Prediction Errors Over Time', fontsize=16)
        
        colors = ['blue', 'green']
        
        for arm_idx in range(self.num_arms):
            base = arm_idx * 8
            if base + 7 > self.D:
                continue
                
            # Position Error
            gt_pos = self.gt_traj[:, base:base+3]
            pr_pos = self.pr_traj[:, base:base+3]
            pos_err = compute_position_error(gt_pos, pr_pos)
            
            # Orientation Error
            gt_quat = self.gt_traj[:, base+3:base+7]
            pr_quat = self.pr_traj[:, base+3:base+7]
            ang_err = compute_angular_error(gt_quat, pr_quat)
            
            # Subplot 1: Position Error
            axes[0].plot(self.time_axis, pos_err, color=colors[arm_idx % len(colors)], 
                         label=f'{self.arm_names[arm_idx]} Pos Error', linewidth=2)
            
            # Subplot 2: Angular Error
            axes[1].plot(self.time_axis, np.degrees(ang_err), color=colors[arm_idx % len(colors)], 
                         label=f'{self.arm_names[arm_idx]} Ang Error', linewidth=2)
            
        axes[0].set_ylabel('L2 Distance (m)')
        axes[0].set_title('Absolute Position Error')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].set_ylabel('Angular Diff (Degrees)')
        axes[1].set_title('Absolute Orientation Error')
        axes[1].set_xlabel('Time (s)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved metrics plot to {save_path}")
        else:
            plt.show()

    def generate_all_plots(self, output_dir: str):
        """Generates and saves all available visualizations to a directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        for i in range(self.num_arms):
            self.plot_dimension_timeseries(arm_idx=i, save_path=os.path.join(output_dir, f'timeseries_arm_{i}.png'))
            
        self.plot_3d_paths(save_path=os.path.join(output_dir, '3d_paths.png'))
        self.plot_metrics_overlay(save_path=os.path.join(output_dir, 'error_metrics.png'))


# -----------------------------------------------------------------------------
# Demonstration & Example Usage
# -----------------------------------------------------------------------------

def main():
    """
    Example demonstrating how to use the TrajectoryVisualizer.
    In a real ML pipeline, fetch ground truth and predicted trajectories from
    your dataloader and normalized model outputs.
    """
    print("Running Trajectory Visualization Tool Example...")

    # 1. Generate Synthetic Dummy Data mimicking an 8-DOF arm
    # [x, y, z, qx, qy, qz, qw, jaw]
    T, D = 50, 8
    
    time_steps = np.linspace(0, 2*np.pi, T)
    gt_trajectory = np.zeros((T, D))
    
    # Position: Spiral
    gt_trajectory[:, 0] = np.sin(time_steps) * 0.1
    gt_trajectory[:, 1] = np.cos(time_steps) * 0.1
    gt_trajectory[:, 2] = np.linspace(0, 0.2, T)
    
    # Orientation: Constant identity quaternion [0, 0, 0, 1]
    gt_trajectory[:, 6] = 1.0  
    
    # Jaw: Sine wave opening
    gt_trajectory[:, 7] = np.abs(np.sin(time_steps * 2))
    
    # 2. Simulate model predictions with noise and drift
    pred_trajectory = gt_trajectory.copy()
    
    # Add random noise to position
    pred_trajectory[:, 0:3] += np.random.normal(0, 0.005, size=(T, 3))
    # Drift
    pred_trajectory[:, 0:3] += np.linspace(0, 0.02, T)[:, None]
    
    # Small angular noise
    rot_noise = R.from_euler('xyz', np.random.normal(0, 0.05, size=(T, 3))).as_quat()
    pred_trajectory[:, 3:7] = rot_noise  # applying identity * noise
    
    # 3. Instantiate Visualizer
    print("Instantiating Trajectory Visualizer...")
    visualizer = TrajectoryVisualizer(
        gt_trajectory=gt_trajectory,
        pred_trajectory=pred_trajectory,
        dt=0.1,
        arm_names=["PSM1_Synthetic"]
    )
    
    # 4. Generate Output
    output_dir = "autonomous_surgery/trajectory_plots"
    print(f"Generating plots to {output_dir}")
    visualizer.generate_all_plots(output_dir)
    print("Done! Check the plots in the output directory.")


if __name__ == "__main__":
    main()
