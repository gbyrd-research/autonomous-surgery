# ROS1 Noetic Workspace

This workspace contains `representation_policy_ros`, a ROS1 bridge package that:

- Subscribes to input topics for endoscope image, wrist images, robot state, and instruction text.
- Runs `RepresentationPolicyInference`.
- Publishes next-step PSM setpoints as `geometry_msgs/PoseStamped` and jaw setpoints as `sensor_msgs/JointState`.
- Converts model pose output `[dxyz_camera, q_delta_tool]` into absolute `setpoint_cp` in the camera frame using `measured_cp`.
- Publishes model jaw output directly as absolute jaw position (no measured jaw reference conversion).
- Denormalizes predicted action chunks using model `action_std` and `action_mean` before publishing.

## Run

```bash
source /opt/ros/noetic/setup.bash
source /ros1_ws/devel/setup.bash
roslaunch representation_policy_ros representation_policy_inference.launch \
  config_path:=/workspace/autonomous_surgery/config/train_representation_policy.yaml \
  checkpoint_path:=/workspace/checkpoints/last_checkpoint.pth \
  robot_state_dim:=16 \
  action_dim:=16 \
  device:=cuda
```
