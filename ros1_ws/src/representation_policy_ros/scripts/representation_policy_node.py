#!/usr/bin/env python3

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import rospy
import torch
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState

from autonomous_surgery.tools.inference_representation_policy import RepresentationPolicyInference


class RepresentationPolicyNode:
    def __init__(self) -> None:
        self.config_path = rospy.get_param("~config_path", "/checkpoints/grasp_only_demo/.hydra/config.yaml")
        self.checkpoint_path = rospy.get_param("~checkpoint_path", "/checkpoints/grasp_only_demo/last_model.pth")
        self.only_psm1 = self._get_bool_param("~only_psm1", False)
        self.only_psm2 = self._get_bool_param("~only_psm2", False)
        if self.only_psm1 and self.only_psm2:
            raise ValueError("Parameters only_psm1 and only_psm2 cannot both be true.")

        self.robot_state_dim = int(rospy.get_param("~robot_state_dim", 16))
        expected_robot_state_dim = 8 if (self.only_psm1 or self.only_psm2) else 16
        if self.robot_state_dim != expected_robot_state_dim:
            raise ValueError(
                "robot_state_dim mismatch: expected {} for only_psm1={} only_psm2={}, got {}".format(
                    expected_robot_state_dim,
                    self.only_psm1,
                    self.only_psm2,
                    self.robot_state_dim,
                )
            )

        configured_action_dim = int(rospy.get_param("~action_dim", self.robot_state_dim))
        if configured_action_dim != self.robot_state_dim:
            raise ValueError(
                "action_dim mismatch: action_dim must equal robot_state_dim ({}), got {}".format(
                    self.robot_state_dim,
                    configured_action_dim,
                )
            )
        self.action_dim = self.robot_state_dim
        self.device = rospy.get_param("~device", "cuda")
        self.inference_rate_hz = float(rospy.get_param("~inference_rate_hz", 5.0))
        self.default_text = rospy.get_param("~default_text", "perform the next surgical action")

        self.endoscope_topic = rospy.get_param(
            "~endoscope_topic",
            "/PSM2/endoscope_img",
        )
        self.wrist_left_topic = rospy.get_param(
            "~wrist_left_topic",
            "/jhu_daVinci/left/image_raw",
        )
        self.wrist_right_topic = rospy.get_param(
            "~wrist_right_topic",
            "/jhu_daVinci/right/image_raw",
        )
        self.robot_psm1_state_topic = rospy.get_param(
            "~robot_psm1_state_topic",
            "/PSM1/measured_cp",
        )
        self.robot_psm2_state_topic = rospy.get_param(
            "~robot_psm2_state_topic",
            "/PSM2/measured_cp",
        )
        self.robot_psm1_jaw_topic = rospy.get_param(
            "~robot_psm1_jaw_topic",
            "/PSM1/jaw/measured_js",
        )
        self.robot_psm2_jaw_topic = rospy.get_param(
            "~robot_psm2_jaw_topic",
            "/PSM2/jaw/measured_js",
        )
        self.psm1_setpoint_topic = rospy.get_param(
            "~psm1_setpoint_topic",
            "/PSM1/setpoint_cp_test",
        )
        self.psm2_setpoint_topic = rospy.get_param(
            "~psm2_setpoint_topic",
            "/PSM2/setpoint_cp_test",
        )
        self.psm1_jaw_setpoint_topic = rospy.get_param(
            "~psm1_jaw_setpoint_topic",
            "/PSM1/jaw/setpoint_cp_test",
        )
        self.psm2_jaw_setpoint_topic = rospy.get_param(
            "~psm2_jaw_setpoint_topic",
            "/PSM2/jaw/setpoint_cp_test",
        )

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.inference_lock = threading.Lock()

        self.latest_endoscope: Optional[Image] = None
        self.latest_wrist_left: Optional[Image] = None
        self.latest_wrist_right: Optional[Image] = None
        self.latest_robot_psm1_state: Optional[PoseStamped] = None
        self.latest_robot_psm2_state: Optional[PoseStamped] = None
        self.latest_robot_psm1_jaw: Optional[JointState] = None
        self.latest_robot_psm2_jaw: Optional[JointState] = None

        self.inference_runner = RepresentationPolicyInference(
            config=self.config_path,
            checkpoint_path=self.checkpoint_path,
            robot_state_dim=self.robot_state_dim,
            action_dim=self.action_dim,
            device=self.device,
        )

        self.psm1_setpoint_pub = rospy.Publisher(self.psm1_setpoint_topic, PoseStamped, queue_size=2)
        self.psm2_setpoint_pub = rospy.Publisher(self.psm2_setpoint_topic, PoseStamped, queue_size=2)
        self.psm1_jaw_setpoint_pub = rospy.Publisher(self.psm1_jaw_setpoint_topic, JointState, queue_size=2)
        self.psm2_jaw_setpoint_pub = rospy.Publisher(self.psm2_jaw_setpoint_topic, JointState, queue_size=2)

        self.endoscope_sub = rospy.Subscriber(self.endoscope_topic, Image, self._endoscope_cb, queue_size=1)
        self.wrist_left_sub = rospy.Subscriber(self.wrist_left_topic, Image, self._wrist_left_cb, queue_size=1)
        self.wrist_right_sub = rospy.Subscriber(self.wrist_right_topic, Image, self._wrist_right_cb, queue_size=1)
        self.robot_psm1_state_sub = rospy.Subscriber(
            self.robot_psm1_state_topic,
            PoseStamped,
            self._robot_psm1_state_cb,
            queue_size=1,
        )
        self.robot_psm2_state_sub = rospy.Subscriber(
            self.robot_psm2_state_topic,
            PoseStamped,
            self._robot_psm2_state_cb,
            queue_size=1,
        )
        self.robot_psm1_jaw_sub = rospy.Subscriber(
            self.robot_psm1_jaw_topic,
            JointState,
            self._robot_psm1_jaw_cb,
            queue_size=1,
        )
        self.robot_psm2_jaw_sub = rospy.Subscriber(
            self.robot_psm2_jaw_topic,
            JointState,
            self._robot_psm2_jaw_cb,
            queue_size=1,
        )

        period = 1.0 / max(self.inference_rate_hz, 1e-3)
        self.timer = rospy.Timer(rospy.Duration(period), self._run_inference)

        rospy.loginfo("representation_policy_node is ready.")
        rospy.loginfo("Publishing PSM1 pose actions to: %s", self.psm1_setpoint_topic)
        rospy.loginfo("Publishing PSM2 pose actions to: %s", self.psm2_setpoint_topic)
        rospy.loginfo("Publishing PSM1 jaw actions to: %s", self.psm1_jaw_setpoint_topic)
        rospy.loginfo("Publishing PSM2 jaw actions to: %s", self.psm2_jaw_setpoint_topic)
        rospy.loginfo(
            "Robot state mode: only_psm1=%s, only_psm2=%s, robot_state_dim=%d",
            str(self.only_psm1),
            str(self.only_psm2),
            self.robot_state_dim,
        )

    def _endoscope_cb(self, msg: Image) -> None:
        with self.lock:
            self.latest_endoscope = msg

    def _wrist_left_cb(self, msg: Image) -> None:
        with self.lock:
            self.latest_wrist_left = msg

    def _wrist_right_cb(self, msg: Image) -> None:
        with self.lock:
            self.latest_wrist_right = msg

    def _robot_psm1_state_cb(self, msg: PoseStamped) -> None:
        with self.lock:
            self.latest_robot_psm1_state = msg

    def _robot_psm2_state_cb(self, msg: PoseStamped) -> None:
        with self.lock:
            self.latest_robot_psm2_state = msg

    def _robot_psm1_jaw_cb(self, msg: JointState) -> None:
        with self.lock:
            self.latest_robot_psm1_jaw = msg

    def _robot_psm2_jaw_cb(self, msg: JointState) -> None:
        with self.lock:
            self.latest_robot_psm2_jaw = msg

    def _run_inference(self, _event: rospy.timer.TimerEvent) -> None:
        if not self.inference_lock.acquire(blocking=False):
            return

        with self.lock:
            endoscope_msg = self.latest_endoscope
            wrist_left_msg = self.latest_wrist_left
            wrist_right_msg = self.latest_wrist_right
            robot_psm1_state_msg = self.latest_robot_psm1_state
            robot_psm2_state_msg = self.latest_robot_psm2_state
            robot_psm1_jaw_msg = self.latest_robot_psm1_jaw
            robot_psm2_jaw_msg = self.latest_robot_psm2_jaw
            text = self.default_text

        try:
            if endoscope_msg is None or wrist_left_msg is None or wrist_right_msg is None:
                return

            if (not self.only_psm2) and (robot_psm1_state_msg is None or robot_psm1_jaw_msg is None):
                return

            if (not self.only_psm1) and (robot_psm2_state_msg is None or robot_psm2_jaw_msg is None):
                return
        

            robot_state_tensor = self._robot_states_to_tensor(
                robot_psm1_state_msg,
                robot_psm2_state_msg,
                robot_psm1_jaw_msg,
                robot_psm2_jaw_msg,
            )
            if robot_state_tensor is None:
                return

            endoscope_tensor = self._image_msg_to_tensor(endoscope_msg)
            wrist_left_tensor = self._image_msg_to_tensor(wrist_left_msg)
            wrist_right_tensor = self._image_msg_to_tensor(wrist_right_msg)

            actions, _ = self.inference_runner.infer(
                endoscope_image=endoscope_tensor,
                wrist_l=wrist_left_tensor,
                wrist_r=wrist_right_tensor,
                robot_states=robot_state_tensor,
                texts=[text],
            )

            # Convert model predictions from normalized space to real action space
            # using action_mean/action_std stored in the checkpointed model buffers.
            actions_np = actions.detach().float().cpu().numpy()
            if actions_np.ndim != 3 or actions_np.shape[0] != 1:
                rospy.logwarn_throttle(5.0, "Unexpected action tensor shape: %s", str(actions_np.shape))
                return

            self._publish_split_actions(
                action_chunk=actions_np[0],
                psm1_state_reference=robot_psm1_state_msg,
                psm2_state_reference=robot_psm2_state_msg,
            )

        except (CvBridgeError, RuntimeError, ValueError) as exc:
            rospy.logerr_throttle(2.0, "Inference failed: %s", str(exc))
        finally:
            self.inference_lock.release()

    def _robot_states_to_tensor(
        self,
        psm1_msg: Optional[PoseStamped],
        psm2_msg: Optional[PoseStamped],
        psm1_jaw_msg: Optional[JointState],
        psm2_jaw_msg: Optional[JointState],
    ) -> Optional[torch.Tensor]:
        state: Optional[np.ndarray] = None

        if self.only_psm1:
            if psm1_msg is None or psm1_jaw_msg is None:
                return None
            state = self._pose_and_jaw_to_state(psm1_msg, psm1_jaw_msg, arm_label="PSM1")

        elif self.only_psm2:
            if psm2_msg is None or psm2_jaw_msg is None:
                return None
            state = self._pose_and_jaw_to_state(psm2_msg, psm2_jaw_msg, arm_label="PSM2")

        else:
            if psm1_msg is None or psm2_msg is None or psm1_jaw_msg is None or psm2_jaw_msg is None:
                return None

            psm1_state = self._pose_and_jaw_to_state(psm1_msg, psm1_jaw_msg, arm_label="PSM1")
            psm2_state = self._pose_and_jaw_to_state(psm2_msg, psm2_jaw_msg, arm_label="PSM2")
            if psm1_state is None or psm2_state is None:
                return None
            # State ordering is always [PSM1 pose+jaw (8), PSM2 pose+jaw (8)].
            state = np.concatenate((psm1_state, psm2_state), axis=0)

        if state is None:
            return None

        if state.size != self.robot_state_dim:
            raise ValueError(
                "Robot state size mismatch: expected {}, got {}".format(
                    self.robot_state_dim,
                    state.size,
                )
            )

        return torch.from_numpy(state).view(1, self.robot_state_dim)

    def _pose_and_jaw_to_state(
        self,
        pose_msg: PoseStamped,
        jaw_msg: JointState,
        arm_label: str,
    ) -> Optional[np.ndarray]:
        jaw_position = self._extract_jaw_position(jaw_msg, arm_label)
        if jaw_position is None:
            return None

        pose_state = self._pose_stamped_to_state(pose_msg)
        return np.concatenate((pose_state, np.array([jaw_position], dtype=np.float32)), axis=0)

    def _extract_jaw_position(self, msg: JointState, arm_label: str) -> Optional[float]:
        if len(msg.position) == 0:
            rospy.logwarn_throttle(5.0, "%s jaw JointState has empty position array", arm_label)
            return None

        jaw_index = 0
        if len(msg.name) > 0:
            for idx, name in enumerate(msg.name):
                if "jaw" in name.lower():
                    jaw_index = idx
                    break

        if jaw_index >= len(msg.position):
            rospy.logwarn_throttle(
                5.0,
                "%s jaw JointState index %d out of range for position length %d",
                arm_label,
                jaw_index,
                len(msg.position),
            )
            return None

        return float(msg.position[jaw_index])

    def _pose_stamped_to_state(self, msg: PoseStamped) -> np.ndarray:
        pose = msg.pose
        return np.array(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float32,
        )

    def _get_bool_param(self, name: str, default: bool) -> bool:
        value = rospy.get_param(name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
        return bool(value)

    def _image_msg_to_tensor(self, msg: Image) -> torch.Tensor:
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got shape {image.shape}")

        encoding = msg.encoding.lower()
        if encoding in {"bgr8", "bgr16"}:
            image = image[:, :, ::-1]

        if image.dtype == np.uint16:
            image = (image / 257.0).astype(np.uint8)
        elif np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)

        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def _publish_split_actions(
        self,
        action_chunk: np.ndarray,
        psm1_state_reference: Optional[PoseStamped],
        psm2_state_reference: Optional[PoseStamped],
    ) -> None:
        if action_chunk.ndim != 2:
            raise ValueError(f"Expected action chunk rank-2, got shape {action_chunk.shape}")

        action_dim = action_chunk.shape[1]
        psm1_actions: Optional[np.ndarray] = None
        psm2_actions: Optional[np.ndarray] = None

        if action_dim == 16:
            psm1_actions = action_chunk[:, :8]
            psm2_actions = action_chunk[:, 8:16]
        elif action_dim == 8:
            if self.only_psm1 == self.only_psm2:
                raise ValueError(
                    "8D action output requires exactly one of only_psm1/only_psm2 to be true."
                )
            if self.only_psm1:
                psm1_actions = action_chunk
            else:
                psm2_actions = action_chunk
        else:
            raise ValueError(f"Unsupported model action_dim={action_dim}. Expected 8 or 16.")

        if psm1_actions is not None:
            self._publish_arm_actions(
                arm_actions=psm1_actions,
                setpoint_pub=self.psm1_setpoint_pub,
                jaw_setpoint_pub=self.psm1_jaw_setpoint_pub,
                pose_reference=psm1_state_reference,
                arm_label="PSM1",
            )
        if psm2_actions is not None:
            self._publish_arm_actions(
                arm_actions=psm2_actions,
                setpoint_pub=self.psm2_setpoint_pub,
                jaw_setpoint_pub=self.psm2_jaw_setpoint_pub,
                pose_reference=psm2_state_reference,
                arm_label="PSM2",
            )

    def _publish_arm_actions(
        self,
        arm_actions: np.ndarray,
        setpoint_pub: rospy.Publisher,
        jaw_setpoint_pub: rospy.Publisher,
        pose_reference: Optional[PoseStamped],
        arm_label: str,
    ) -> None:
        if arm_actions.ndim != 2 or arm_actions.shape[1] != 8:
            raise ValueError(
                f"{arm_label} action block must be [horizon, 8], got shape {arm_actions.shape}"
            )
        if arm_actions.shape[0] < 1:
            raise ValueError(f"{arm_label} action block must include at least one horizon step")

        next_action = arm_actions[0]
        pose_msg = self._build_pose_setpoint_msg(next_action[:7], pose_reference)
        jaw_msg = self._build_jaw_setpoint_msg(next_action[7], arm_label)
        setpoint_pub.publish(pose_msg)
        jaw_setpoint_pub.publish(jaw_msg)

    def _build_pose_setpoint_msg(
        self,
        pose_values: np.ndarray,
        reference_msg: PoseStamped,
    ) -> PoseStamped:
        if pose_values.shape[0] != 7:
            raise ValueError(f"Pose action must contain 7 values, got shape {pose_values.shape}")

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = reference_msg.header.frame_id

        # Model output is [dxyz_camera, q_delta_tool]. Convert to absolute setpoint in camera frame.
        msg.pose.position.x = float(reference_msg.pose.position.x + pose_values[0])
        msg.pose.position.y = float(reference_msg.pose.position.y + pose_values[1])
        msg.pose.position.z = float(reference_msg.pose.position.z + pose_values[2])

        current_quat = np.array(
            [
                reference_msg.pose.orientation.x,
                reference_msg.pose.orientation.y,
                reference_msg.pose.orientation.z,
                reference_msg.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        delta_quat = np.array(
            [pose_values[3], pose_values[4], pose_values[5], pose_values[6]],
            dtype=np.float64,
        )
        current_quat = self._normalize_quaternion(current_quat, "measured_cp orientation")
        delta_quat = self._normalize_quaternion(delta_quat, "predicted tool-frame orientation delta")
        target_quat = self._quat_multiply(current_quat, delta_quat)
        target_quat = self._normalize_quaternion(target_quat, "composed setpoint orientation")

        msg.pose.orientation.x = float(target_quat[0])
        msg.pose.orientation.y = float(target_quat[1])
        msg.pose.orientation.z = float(target_quat[2])
        msg.pose.orientation.w = float(target_quat[3])
        return msg

    def _normalize_quaternion(self, quat: np.ndarray, label: str) -> np.ndarray:
        norm = float(np.linalg.norm(quat))
        if norm < 1e-8:
            raise ValueError(f"{label} has near-zero norm; cannot construct valid quaternion.")
        return quat / norm

    def _quat_multiply(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = left
        x2, y2, z2, w2 = right
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float64,
        )

    def _build_jaw_setpoint_msg(
        self,
        jaw_value: float,
        arm_label: str,
    ) -> JointState:
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        if arm_label:
            msg.name = [f"{arm_label.lower()}_jaw_joint"]

        msg.position = [float(jaw_value)]
        return msg


def main() -> None:
    rospy.init_node("representation_policy_node")
    RepresentationPolicyNode()
    rospy.spin()


if __name__ == "__main__":
    main()
