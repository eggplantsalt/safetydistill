import collections
import dataclasses
import logging
import math
import pathlib
import sys
import warnings
from typing import List

import cvxpy as cp
import imageio
import mujoco
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from scipy.spatial.transform import Rotation as R

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kkt_sense.aegis_adapter import build_aegis_step_record
from kkt_sense.aegis_adapter import extract_cvxpy_qp_certificate
from kkt_sense.aegis_adapter import make_episode_output_path
from kkt_sense.label_exporter import export_episode
from kkt_sense.openvla_sample_exporter import build_openvla_sample_manifest
from kkt_sense.openvla_sample_exporter import build_state_fields
from kkt_sense.openvla_sample_exporter import make_openvla_episode_dir
from kkt_sense.openvla_sample_exporter import save_openvla_step_sample
from utils import (
    compute_h_coeffs_3d,
    filtering_points,
    fit_ellipse,
    get_point_cloud,
    obstacle_detection,
)

warnings.filterwarnings("ignore")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 1024
ALPHA = 1.0
MAX_VEL = 1.0


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "safelibero_spatial"
    safety_level: str = "I"
    task_index: List[int] = dataclasses.field(default_factory=lambda: [0])
    episode_index: List[int] = dataclasses.field(default_factory=lambda: [0])
    num_steps_wait: int = 20
    num_trials_per_task: int = 50

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "results_new"

    #################################################################################################################
    # Debug / smoke-test switches
    #################################################################################################################
    disable_groundingdino: bool = False
    debug_synthetic_safety_obstacle: bool = False

    #################################################################################################################
    # KKT-SenseVLA label export
    #################################################################################################################
    enable_kkt_label_export: bool = False
    kkt_label_output_dir: str = "data/kkt_safelibero_labels"

    #################################################################################################################
    # OpenVLA-style sample export
    #################################################################################################################
    enable_openvla_sample_export: bool = False
    openvla_sample_output_dir: str = "data/kkt_openvla_samples"

    seed: int = 7


def _action_to_numpy(action):
    """Convert numpy/list/tuple/tensor-like action to a float numpy array."""
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    elif hasattr(action, "cpu") and hasattr(action, "numpy"):
        action = action.cpu().numpy()

    if isinstance(action, np.ndarray):
        return action.astype(float, copy=False)

    if isinstance(action, (list, tuple)):
        return np.asarray(action, dtype=float)

    raise TypeError("Unsupported action type: {}".format(type(action)))


def _action_to_list(action):
    """Convert numpy/list/tuple/tensor-like action to a JSON-friendly float list."""
    return _action_to_numpy(action).astype(float).tolist()


def _load_groundingdino_if_needed(args):
    """Load GroundingDINO unless an explicit debug mode disables perception."""
    if args.disable_groundingdino or args.debug_synthetic_safety_obstacle:
        logging.warning("GroundingDINO disabled; using debug fallback obstacle perception.")
        return None

    from groundingdino.util.inference import load_model

    config_path = "GroundingDINO/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "GroundingDINO/groundingdino_swint_ogc.pth"
    return load_model(config_path, checkpoint_path)


def _merge_debug_flags(base_debug_flags, extra_debug):
    merged = {}
    if extra_debug:
        merged.update(extra_debug)
    if base_debug_flags:
        for key, value in base_debug_flags.items():
            merged.setdefault(key, value)
    return merged


def _normalize_vector(vec, fallback=None):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        if fallback is None:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return fallback
    return vec / norm


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    safety_level = args.safety_level
    task_index = args.task_index
    episode_index = args.episode_index
    enable_kkt_label_export = args.enable_kkt_label_export
    kkt_label_output_dir = args.kkt_label_output_dir
    enable_openvla_sample_export = args.enable_openvla_sample_export
    openvla_sample_output_dir = args.openvla_sample_output_dir

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name](safety_level=safety_level)
    logging.info("Task suite: %s, safety level: %s", args.task_suite_name, safety_level)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "safelibero_spatial":
        max_steps = 300
    elif args.task_suite_name == "safelibero_object":
        max_steps = 300
    elif args.task_suite_name == "safelibero_goal":
        max_steps = 300
    elif args.task_suite_name == "safelibero_10":
        max_steps = 550
    elif args.task_suite_name == "safelibero_90":
        max_steps = 400
    else:
        raise ValueError("Unknown task suite: {}".format(args.task_suite_name))

    print("OK")
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    model_groundingdino = _load_groundingdino_if_needed(args)

    total_episodes, total_successes = 0, 0

    for task_id in task_index:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        env, task_description = _get_libero_env(task, safety_level, LIBERO_ENV_RESOLUTION, args.seed)
        model = env.sim.model
        data = env.sim.data

        collides = 0
        time_steps = []

        task_episodes, task_successes = 0, 0
        task_segment = task_description.replace(" ", "_")

        _out_dir = pathlib.Path(args.video_out_path) / "{}".format(task_segment)
        _out_dir.mkdir(parents=True, exist_ok=True)
        out_dir = _out_dir / "ours_{}".format(safety_level)
        out_dir.mkdir(parents=True, exist_ok=True)

        for episode_idx in episode_index:
            logging.info("\nTask: %s", task_description)

            env.reset()
            action_plan = collections.deque()
            step_records = [] if enable_kkt_label_export else None

            openvla_episode_dir = None
            if enable_openvla_sample_export:
                openvla_episode_dir = make_openvla_episode_dir(
                    openvla_sample_output_dir,
                    args.task_suite_name,
                    safety_level,
                    task_id,
                    episode_idx,
                )

            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            model = env.sim.model
            data = env.sim.data
            eef_body_id = model.body_name2id("eef_marker")

            eef_pos = obs["robot0_eef_pos"]
            eef_quat = obs["robot0_eef_quat"]
            R1 = R.from_quat(eef_quat).as_matrix()
            offset_local = np.array([0, 0, -0.08])
            offset_world = R1 @ offset_local
            ball_pos = eef_pos + offset_world
            p1 = ball_pos

            env.sim.model.body_pos[eef_body_id] = ball_pos
            env.sim.model.body_quat[eef_body_id] = eef_quat[[3, 0, 1, 2]]

            if "orange juice" in task_description or "milk" in task_description or "alphabet soup" in task_description:
                Q1_diag = np.array([0.06, 0.12, 0.2])
            else:
                Q1_diag = np.array([0.06, 0.12, 0.11])

            while t < args.num_steps_wait:
                try:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                except Exception as e:
                    logging.error("Caught exception: %s", e)
                    break

            img_out_dir = out_dir / "{}".format(episode_idx)
            img_out_dir.mkdir(parents=True, exist_ok=True)

            base_debug_flags = {}

            if args.debug_synthetic_safety_obstacle:
                obstacle_infromation = "debug_synthetic_safety_obstacle"
                flag_safety_control = True

                p2 = p1 + np.array([0.03, 0.0, 0.0], dtype=np.float64)
                R2 = np.eye(3)
                Q2_diag = np.array([0.06, 0.08, 0.08], dtype=np.float64)

                z_fixed = _normalize_vector(p2 - p1)
                p_target = np.array([-0.05, 0.15, 1.05])
                Kp_pos = 1
                dt = 0.05

                base_debug_flags.update(
                    {
                        "synthetic_safety_obstacle": True,
                        "groundingdino_disabled": True,
                        "obstacle_information": obstacle_infromation,
                    }
                )
                logging.warning("Using synthetic safety obstacle; GroundingDINO and fit_ellipse are skipped.")

            elif args.disable_groundingdino:
                obstacle_infromation = "debug_dummy_obstacle"
                flag_safety_control = False

                base_debug_flags.update(
                    {
                        "groundingdino_disabled": True,
                        "obstacle_information": obstacle_infromation,
                    }
                )
                logging.warning(
                    "GroundingDINO disabled; skipping point filtering, ellipsoid fitting, and safety control."
                )

            else:
                agentview_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                agentview_depth = np.ascontiguousarray(obs["agentview_depth"][::-1, ::-1])

                obstacle_infromation = obstacle_detection(agentview_img, task_description, args.task_suite_name)
                agent_view_points = get_point_cloud(
                    agentview_img,
                    agentview_depth,
                    env,
                    "agentview",
                    obstacle_infromation,
                    model_groundingdino,
                    img_out_dir,
                )

                backview_img = np.ascontiguousarray(obs["backview_image"][::-1, ::-1])
                backview_depth = np.ascontiguousarray(obs["backview_depth"][::-1, ::-1])
                back_view_points = get_point_cloud(
                    backview_img,
                    backview_depth,
                    env,
                    "backview",
                    obstacle_infromation,
                    model_groundingdino,
                    img_out_dir,
                )

                if agent_view_points.shape[1] > 0 and back_view_points.shape[1] > 0:
                    full_points = np.vstack([agent_view_points, back_view_points])
                elif agent_view_points.shape[1] == 0 and back_view_points.shape[1] > 0:
                    full_points = back_view_points
                elif agent_view_points.shape[1] > 0 and back_view_points.shape[1] == 0:
                    full_points = agent_view_points
                else:
                    full_points = np.array([[]])

                filter_points = filtering_points(full_points, args.task_suite_name)
                flag_safety_control = True
                if filter_points.shape[0] == 0:
                    flag_safety_control = False

                if flag_safety_control:
                    p2, R2, Q2_diag = fit_ellipse(filter_points, plot=True, save_path=img_out_dir)
                    z_fixed = _normalize_vector(p2 - p1)
                    p_target = np.array([-0.05, 0.15, 1.05])
                    Kp_pos = 1
                    dt = 0.05

            t = 0

            obstacle_names = [n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n]

            obstacle_name = " "
            for i in obstacle_names:
                p = obs["{}_pos".format(i)]
                if p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
                    obstacle_name = i
                    print("Obstacle name:", i)
                    break

            initial_obstacle_pos = obs[obstacle_name + "_pos"]
            collide_flag = False
            collide_time = 0
            done = False

            logging.info("Starting episode %s...", task_episodes + 1)
            while t < max_steps:
                try:
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    replay_images.append(img)

                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            "We want to replan every {} steps, but policy only predicts {} steps.".format(
                                args.replan_steps,
                                len(action_chunk),
                            )
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    action_nominal = _action_to_list(action)
                    action_array = _action_to_numpy(action)
                    action_delta = None

                    if flag_safety_control:
                        gripper_qpos_current = obs.get("robot0_gripper_qpos")
                        action_movement = np.zeros_like(action_array)
                        action_movement[:3] = action_array[:3]
                        action_movement[6] = action_array[6]

                        v_ref = Kp_pos * R1.T @ action_movement[:3]
                        u_v_ref = 5 * v_ref

                        a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(
                            p1,
                            Q1_diag,
                            R1,
                            p2,
                            Q2_diag,
                            R2,
                            z_fixed,
                        )
                        a_u_v = 0.2 * a_v

                        u_z_nom = 10 * mu_row
                        u = cp.Variable(6)

                        W = np.diag([1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0, 1.0, 1.0])
                        u_ref_vec = np.hstack([u_v_ref, u_z_nom])
                        objective = cp.Minimize(cp.quad_form(u - u_ref_vec, W))
                        constraints = [a_u_v @ u[:3] + a_uz @ u[3:6] + 10 * h >= 0]

                        prob = cp.Problem(objective, constraints)
                        prob.solve(solver=cp.OSQP)

                        u_value = u.value if u.value is not None else None
                        certificate = extract_cvxpy_qp_certificate(
                            prob=prob,
                            constraints=constraints,
                            h=h,
                            a_u_v=a_u_v,
                            a_uz=a_uz,
                            u_ref_vec=u_ref_vec,
                            u_value=u_value,
                        )

                        if u.value is not None:
                            u_v = u.value[:3]
                            u_z = u.value[3:]
                            certificate["extra_debug"]["used_nominal_fallback"] = False
                        else:
                            u_v = u_v_ref
                            u_z = u_z_nom
                            print("QP infeasible or no solution; fallback to nominal reference")
                            certificate["extra_debug"]["used_nominal_fallback"] = True
                            certificate["qp_status"] = "{}_fallback_nominal".format(certificate["qp_status"])

                        identity = np.eye(len(z_fixed))
                        dz = (identity - np.outer(z_fixed, z_fixed)) @ u_z
                        z_fixed = z_fixed + dz * dt
                        z_fixed = z_fixed / np.linalg.norm(z_fixed)

                        action_input = np.zeros(7)
                        action_input[:3] = 0.2 * R1 @ u_v
                        action_input[6] = action_array[6]
                        action_safe = action_input.tolist()
                        action_delta = (
                            None
                            if action_nominal is None
                            else [safe - nominal for safe, nominal in zip(action_safe, action_nominal)]
                        )

                        step_extra_debug = _merge_debug_flags(
                            base_debug_flags,
                            certificate.get("extra_debug", {}),
                        )

                        if enable_kkt_label_export:
                            step_records.append(
                                build_aegis_step_record(
                                    task_suite_name=args.task_suite_name,
                                    safety_level=safety_level,
                                    task_index=task_id,
                                    episode_index=episode_idx,
                                    step_index=t,
                                    instruction=task_description,
                                    observation_metadata={"timestep": t},
                                    action_nominal=action_nominal,
                                    action_safe=action_safe,
                                    constraint_values=certificate["constraint_values"],
                                    constraint_gradients=certificate["constraint_gradients"],
                                    dual_variables=certificate["dual_variables"],
                                    active_set=certificate["active_set"],
                                    qp_status=certificate["qp_status"],
                                    extra_debug=step_extra_debug,
                                )
                            )

                        obs, reward, done, info = env.step(action_input.tolist())

                        if enable_openvla_sample_export and openvla_episode_dir is not None:
                            state_vec = element["observation/state"].astype(float).tolist()
                            gripper_qpos = gripper_qpos_current
                            state_fields = build_state_fields(gripper_qpos)
                            save_openvla_step_sample(
                                episode_dir=openvla_episode_dir,
                                step_index=t,
                                agentview_img=img,
                                wrist_img=wrist_img,
                                task_suite_name=args.task_suite_name,
                                safety_level=safety_level,
                                task_index=task_id,
                                episode_index=episode_idx,
                                instruction=task_description,
                                state=state_vec,
                                state_fields=state_fields,
                                action_nominal=action_nominal,
                                action_safe=action_safe,
                                action_delta=action_delta,
                                dual_variables=certificate.get("dual_variables"),
                                active_set=certificate.get("active_set"),
                                constraint_values=certificate.get("constraint_values"),
                                constraint_gradients=certificate.get("constraint_gradients"),
                                qp_status=certificate.get("qp_status"),
                                extra_debug=step_extra_debug,
                            )

                    else:
                        gripper_qpos_current = obs.get("robot0_gripper_qpos")
                        action_safe = action_nominal
                        action_delta = (
                            None
                            if action_nominal is None
                            else [safe - nominal for safe, nominal in zip(action_safe, action_nominal)]
                        )
                        if enable_kkt_label_export:
                            step_records.append(
                                build_aegis_step_record(
                                    task_suite_name=args.task_suite_name,
                                    safety_level=safety_level,
                                    task_index=task_id,
                                    episode_index=episode_idx,
                                    step_index=t,
                                    instruction=task_description,
                                    observation_metadata={"timestep": t},
                                    action_nominal=action_nominal,
                                    action_safe=action_safe,
                                    qp_status="no_safety_control",
                                    extra_debug=dict(base_debug_flags),
                                )
                            )

                        obs, reward, done, info = env.step(action_array.tolist())

                        if enable_openvla_sample_export and openvla_episode_dir is not None:
                            state_vec = element["observation/state"].astype(float).tolist()
                            gripper_qpos = gripper_qpos_current
                            state_fields = build_state_fields(gripper_qpos)
                            save_openvla_step_sample(
                                episode_dir=openvla_episode_dir,
                                step_index=t,
                                agentview_img=img,
                                wrist_img=wrist_img,
                                task_suite_name=args.task_suite_name,
                                safety_level=safety_level,
                                task_index=task_id,
                                episode_index=episode_idx,
                                instruction=task_description,
                                state=state_vec,
                                state_fields=state_fields,
                                action_nominal=action_nominal,
                                action_safe=action_safe,
                                action_delta=action_delta,
                                dual_variables=None,
                                active_set=None,
                                constraint_values=None,
                                constraint_gradients=None,
                                qp_status="no_safety_control",
                                extra_debug=dict(base_debug_flags),
                            )

                    if collide_flag is False:
                        then_obstacle_pos = obs[obstacle_name + "_pos"]
                        if np.sum(np.abs(then_obstacle_pos - initial_obstacle_pos)) > 0.001:
                            print("obstacle collided")
                            collide_flag = True
                            collide_time = t

                    eef_pos = obs["robot0_eef_pos"]
                    eef_quat = obs["robot0_eef_quat"]
                    R1 = R.from_quat(eef_quat).as_matrix()
                    offset_local = np.array([0, 0, -0.08])
                    offset_world = R1 @ offset_local
                    ball_pos = eef_pos + offset_world
                    env.sim.model.body_pos[eef_body_id] = ball_pos
                    env.sim.model.body_quat[eef_body_id] = eef_quat[[3, 0, 1, 2]]
                    p1 = ball_pos

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1

                except Exception as e:
                    logging.error("Caught exception: %s", e)
                    break

            task_episodes += 1
            total_episodes += 1

            time_steps.append(t)
            if collide_flag is True:
                collides += 1

            suffix = "success" if done else "failure"
            video_path = out_dir / "{}_{}.mp4".format(episode_idx, suffix)
            imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=30)

            if enable_kkt_label_export:
                output_path = make_episode_output_path(
                    kkt_label_output_dir,
                    args.task_suite_name,
                    safety_level,
                    task_id,
                    episode_idx,
                )
                export_episode(step_records, str(output_path))

            if enable_openvla_sample_export:
                build_openvla_sample_manifest(openvla_sample_output_dir)

            logging.info("Success: %s", done)
            logging.info("# episodes completed so far: %s", total_episodes)
            logging.info("Collision: %s", collide_flag)
            logging.info("# successes: %s (%.1f%%)", total_successes, total_successes / total_episodes * 100)
            logging.info("# collides: %s (%.1f%%)", collides, collides / total_episodes * 100)
            print("collide_flag:", collide_flag)
            print("collide_time:", collide_time)

        logging.info("Current task success rate: %s", float(task_successes) / float(task_episodes))
        logging.info("Current total success rate: %s", float(total_successes) / float(total_episodes))

    logging.info("Total success rate: %s", float(total_successes) / float(total_episodes))
    logging.info("Total episodes: %s", total_episodes)
    logging.info("Time steps: %s", time_steps)


def _get_libero_env(task, level, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    print(task_description)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": True,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    eval_libero(args)