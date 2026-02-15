from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pickle
import sys
from dataclasses import dataclass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DETR_PATH = os.path.join(PROJECT_ROOT, "detr")
if DETR_PATH not in sys.path:
    sys.path.insert(0, DETR_PATH)

import numpy as np
_IMPORT_ERROR = None

try:
    import cv2
    import torch
    from PIL import Image
    from torchvision import transforms

    from policy import ACTPolicy
    from piper_constants import DT, SIM_TASK_CONFIGS
    from piper_sim_env import MANYCUBES_COLORS, make_sim_env
    from train_state_classifier import (
        DualStateClassifier,
        STATE_COOP,
        STATE_HOLD,
        STATE_INDEP,
    )
    from utils import set_seed
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc
    cv2 = None
    torch = None
    Image = None
    transforms = None
    ACTPolicy = object
    DT = 0.02
    SIM_TASK_CONFIGS = {}
    MANYCUBES_COLORS = [None]
    make_sim_env = None
    DualStateClassifier = None
    STATE_HOLD = 0
    STATE_INDEP = 1
    STATE_COOP = 2
    def set_seed(_):
        return None

STATE_NAME = {
    STATE_HOLD: "HOLD",
    STATE_INDEP: "INDEP",
    STATE_COOP: "COOP",
}


@dataclass
class PolicyBundle:
    model: ACTPolicy
    stats: dict

    def pre(self, qpos: np.ndarray) -> np.ndarray:
        return (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]

    def post(self, action: np.ndarray) -> np.ndarray:
        return action * self.stats["action_std"] + self.stats["action_mean"]


class TemporalActionBank:
    def __init__(self, decay_k: float):
        self.decay_k = float(decay_k)
        self.entries = []

    def add(self, start_t: int, chunk_np: np.ndarray):
        self.entries.append((int(start_t), chunk_np.copy()))

    def prune(self, t: int):
        kept = []
        for st, chunk in self.entries:
            if st + len(chunk) > t:
                kept.append((st, chunk))
        self.entries = kept

    def get(self, t: int):
        self.prune(t)
        candidates = []
        for st, chunk in self.entries:
            off = t - st
            if 0 <= off < len(chunk):
                candidates.append(chunk[off])
        if not candidates:
            return None
        arr = np.stack(candidates, axis=0)
        n = arr.shape[0]
        weights = np.exp(-self.decay_k * (n - 1 - np.arange(n)))
        weights = weights / np.sum(weights)
        return np.sum(arr * weights[:, None], axis=0)


def make_policy(policy_class: str, policy_config: dict):
    if policy_class != "ACT":
        raise NotImplementedError("Only ACT policy is supported in hierarchical_policy.py")
    return ACTPolicy(policy_config)


def resolve_ckpt_path(ckpt_dir_or_file: str):
    if os.path.isfile(ckpt_dir_or_file):
        return ckpt_dir_or_file
    return os.path.join(ckpt_dir_or_file, "policy_best.ckpt")


def resolve_stats_path(ckpt_dir_or_file: str):
    if os.path.isfile(ckpt_dir_or_file):
        return os.path.join(os.path.dirname(ckpt_dir_or_file), "dataset_stats.pkl")
    return os.path.join(ckpt_dir_or_file, "dataset_stats.pkl")


def load_policy_bundle(
    ckpt_path: str,
    args,
    device: torch.device,
    state_dim: int,
    arm: str = None,
):
    config = {
        "lr": args.lr,
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": ["top"],
        "state_dim": state_dim,
    }
    if arm is not None:
        config["arm"] = arm

    model = make_policy(args.policy_class, config)

    with open(resolve_stats_path(ckpt_path), "rb") as f:
        stats = pickle.load(f)

    ckpt_file = resolve_ckpt_path(ckpt_path)
    loaded = torch.load(ckpt_file, map_location=device)
    if "model_state_dict" in loaded:
        loaded = loaded["model_state_dict"]

    if "model.pos_table" in loaded and "model.query_embed.weight" in loaded:
        curr_pos_len = model.model.pos_table.shape[1]
        load_pos_len = loaded["model.pos_table"].shape[1]
        if load_pos_len > curr_pos_len:
            loaded["model.pos_table"] = loaded["model.pos_table"][:, :curr_pos_len, :]

        curr_q_len = model.model.query_embed.weight.shape[0]
        load_q_len = loaded["model.query_embed.weight"].shape[0]
        if load_q_len > curr_q_len:
            loaded["model.query_embed.weight"] = loaded["model.query_embed.weight"][:curr_q_len, :]

    model.load_state_dict(loaded)
    model.to(device)
    model.eval()
    return PolicyBundle(model=model, stats=stats)


def load_state_classifier(ckpt_path: str, device: torch.device):
    model = DualStateClassifier(num_classes=3)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_dual_image(observation: dict, device: torch.device):
    image = np.ascontiguousarray(observation["images"]["top"].copy())
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(device) / 255.0
    return tensor.unsqueeze(0).unsqueeze(0)


def apply_torch_rgb_mask(images, strip_width=40, arm="right"):
    is_single = len(images.shape) == 3
    if is_single:
        images = images.unsqueeze(0)

    orig_shape = images.shape
    if len(orig_shape) == 5:
        b, n_cam, c, h, w = orig_shape
        images = images.view(b * n_cam, c, h, w)

    _, c, _, w = images.shape

    if arm == "left":
        strip_start = w - strip_width
        strip_end = w
    else:
        strip_start = 0
        strip_end = strip_width

    strip = images[:, :, :, strip_start:strip_end]

    t_100 = 100.0 / 255.0
    t_150 = 150.0 / 255.0
    r = strip[:, 0, :, :]
    g = strip[:, 1, :, :]
    b = strip[:, 2, :, :]

    mask_r = (r > t_100) & (g < t_100) & (b < t_100)
    mask_g = (g > t_100) & (r < t_100) & (b < t_100)
    mask_b = (b > t_150) & (r < t_100) & (g < t_100)

    combined_mask = (mask_r | mask_g | mask_b).unsqueeze(1).repeat(1, c, 1, 1).float()
    masked_strip = strip * combined_mask

    outputs = images.clone()
    outputs[:, :, :, strip_start:strip_end] = masked_strip

    if len(orig_shape) == 5:
        outputs = outputs.view(orig_shape)
    elif is_single:
        outputs = outputs.squeeze(0)

    return outputs


def get_indep_image(observation: dict, arm: str, device: torch.device):
    image = np.ascontiguousarray(observation["images"]["top"].copy())
    h, w, _ = image.shape
    image_t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    if arm == "left":
        image_t = image_t[:, :, :, : w // 2]
        image_t = apply_torch_rgb_mask(image_t, strip_width=40, arm="left")
    else:
        image_t = image_t[:, :, :, w // 2 :]
        image_t = apply_torch_rgb_mask(image_t, strip_width=40, arm="right")

    return image_t.unsqueeze(0)


def arm_is_home(qpos_slice: np.ndarray, home_slice: np.ndarray, threshold: float):
    return float(np.max(np.abs(qpos_slice - home_slice))) <= float(threshold)


def sync_envs(main_physics, shadow_physics):
    shadow_physics.data.qpos[:] = main_physics.data.qpos[:]
    shadow_physics.data.qvel[:] = main_physics.data.qvel[:]
    shadow_physics.model.geom_rgba[:] = main_physics.model.geom_rgba[:]
    shadow_physics.forward()


def set_visible_indices(physics, visible_indices):
    visible = set(visible_indices) if visible_indices else set()
    for i in range(10):
        try:
            gid = physics.model.name2id(f"cube_{i}", "geom")
            physics.model.geom_rgba[gid, 3] = 1.0 if i in visible else 0.0
        except Exception:
            continue
    physics.forward()


def remove_cubes(physics, indices):
    if not indices:
        return
    for i in indices:
        try:
            jid = physics.model.name2id(f"cube_{i}_joint", "joint")
            qpos_adr = physics.model.jnt_qposadr[jid]
            physics.data.qpos[qpos_adr + 0] = 10.0 + i
            physics.data.qpos[qpos_adr + 1] = 10.0 + i
            physics.data.qpos[qpos_adr + 2] = -10.0
            gid = physics.model.name2id(f"cube_{i}", "geom")
            physics.model.geom_rgba[gid, 3] = 0.0
        except Exception:
            continue
    physics.forward()


def get_touched_cubes_per_arm(physics):
    touched = {"left": set(), "right": set()}
    for i in range(physics.data.ncon):
        id1 = physics.data.contact[i].geom1
        id2 = physics.data.contact[i].geom2
        for g_id, o_id in [(id1, id2), (id2, id1)]:
            try:
                b_id = physics.model.geom_bodyid[g_id]
                b_name = physics.model.id2name(b_id, "body")
            except Exception:
                continue
            arm = None
            if b_name in ["l_link7", "l_link8"]:
                arm = "left"
            elif b_name in ["r_link7", "r_link8"]:
                arm = "right"
            if arm is None:
                continue
            o_name = physics.model.id2name(o_id, "geom")
            if o_name and o_name.startswith("cube_"):
                try:
                    touched[arm].add(int(o_name.split("_")[1]))
                except Exception:
                    pass
    return touched


def get_grasped_cubes(physics):
    grasped = {"left": set(), "right": set()}
    finger_hits = {"left": collections.defaultdict(set), "right": collections.defaultdict(set)}
    for i in range(physics.data.ncon):
        id1 = physics.data.contact[i].geom1
        id2 = physics.data.contact[i].geom2
        for g_id, o_id in [(id1, id2), (id2, id1)]:
            try:
                b_id = physics.model.geom_bodyid[g_id]
                b_name = physics.model.id2name(b_id, "body")
            except Exception:
                continue
            arm, side = None, None
            if b_name == "l_link7":
                arm, side = "left", "1"
            elif b_name == "l_link8":
                arm, side = "left", "2"
            elif b_name == "r_link7":
                arm, side = "right", "1"
            elif b_name == "r_link8":
                arm, side = "right", "2"
            if arm is None:
                continue
            o_name = physics.model.id2name(o_id, "geom")
            if o_name and o_name.startswith("cube_"):
                try:
                    c_idx = int(o_name.split("_")[1])
                    finger_hits[arm][c_idx].add(side)
                except Exception:
                    pass
    for arm in ["left", "right"]:
        for c_idx, sides in finger_hits[arm].items():
            if len(sides) >= 2:
                grasped[arm].add(c_idx)
    return grasped


def get_cubes_touching_targets(physics, target_geoms):
    contacts = []
    for i in range(physics.data.ncon):
        id1 = physics.data.contact[i].geom1
        id2 = physics.data.contact[i].geom2
        n1 = physics.model.id2name(id1, "geom")
        n2 = physics.model.id2name(id2, "geom")
        if n1 and n2:
            contacts.append((n1, n2))

    touching = set()
    for n1, n2 in contacts:
        for a, b in [(n1, n2), (n2, n1)]:
            if a in target_geoms and b.startswith("cube_"):
                try:
                    touching.add(int(b.split("_")[1]))
                except Exception:
                    pass

    changed = True
    while changed:
        changed = False
        for n1, n2 in contacts:
            c1 = int(n1.split("_")[1]) if n1.startswith("cube_") else -1
            c2 = int(n2.split("_")[1]) if n2.startswith("cube_") else -1
            if c1 >= 0 and c2 >= 0:
                if c1 in touching and c2 not in touching:
                    touching.add(c2)
                    changed = True
                elif c2 in touching and c1 not in touching:
                    touching.add(c1)
                    changed = True
    return touching


def get_cubes_in_goal(physics):
    return get_cubes_touching_targets(physics, {"goal_plate"})


def get_cubes_on_cushion(physics):
    return get_cubes_touching_targets(physics, {"cushion1"})


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quaternion_inverse(q):
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q)


def rotate_vector_by_quaternion(v, q):
    q_vec = np.array([0, v[0], v[1], v[2]])
    q_inv = quaternion_inverse(q)
    tmp = quaternion_multiply(q, q_vec)
    out = quaternion_multiply(tmp, q_inv)
    return out[1:]


def apply_magnet_logic(physics, magnetized_pairs, color_sequence=None):
    if color_sequence:
        greens = [i for i, c in enumerate(color_sequence) if c == "g"]
        blues = [i for i, c in enumerate(color_sequence) if c == "b"]
    else:
        greens, blues = [8], [9]

    threshold = 0.06
    for g_idx in greens:
        if g_idx in magnetized_pairs:
            continue
        try:
            g_bid = physics.model.name2id(f"cube_{g_idx}", "body")
            g_pos = physics.data.xpos[g_bid].copy()
            g_quat = physics.data.xquat[g_bid].copy()
        except Exception:
            continue

        for b_idx in blues:
            try:
                b_bid = physics.model.name2id(f"cube_{b_idx}", "body")
                b_pos = physics.data.xpos[b_bid].copy()
                b_quat = physics.data.xquat[b_bid].copy()
            except Exception:
                continue

            if np.linalg.norm(g_pos - b_pos) < threshold:
                b_inv = quaternion_inverse(b_quat)
                rel_pos = rotate_vector_by_quaternion(g_pos - b_pos, b_inv)
                rel_quat = quaternion_multiply(b_inv, g_quat)
                magnetized_pairs[g_idx] = {"blue_idx": b_idx, "rel_pos": rel_pos, "rel_quat": rel_quat}
                try:
                    g_gid = physics.model.name2id(f"cube_{g_idx}", "geom")
                    physics.model.geom_contype[g_gid] = 0
                    physics.model.geom_conaffinity[g_gid] = 0
                except Exception:
                    pass
                break

    for g_idx, data in magnetized_pairs.items():
        try:
            b_idx = data["blue_idx"]
            b_bid = physics.model.name2id(f"cube_{b_idx}", "body")
            b_pos = physics.data.xpos[b_bid].copy()
            b_quat = physics.data.xquat[b_bid].copy()
            target_pos = b_pos + rotate_vector_by_quaternion(data["rel_pos"], b_quat)
            target_quat = quaternion_multiply(b_quat, data["rel_quat"])
            j_id = physics.model.name2id(f"cube_{g_idx}_joint", "joint")
            qpos_adr = physics.model.jnt_qposadr[j_id]
            qvel_adr = physics.model.jnt_dofadr[j_id]
            physics.data.qpos[qpos_adr : qpos_adr + 3] = target_pos
            physics.data.qpos[qpos_adr + 3 : qpos_adr + 7] = target_quat
            physics.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        except Exception:
            continue


def reset_magnet_logic(physics):
    for i in range(10):
        try:
            gid = physics.model.name2id(f"cube_{i}", "geom")
            physics.model.geom_contype[gid] = 1
            physics.model.geom_conaffinity[gid] = 1
        except Exception:
            continue


def get_spatial_object_map(physics):
    cubes = []
    for i in range(10):
        try:
            bid = physics.model.name2id(f"cube_{i}", "body")
            x = float(physics.data.xpos[bid][0])
            cubes.append({"id": i, "x": x})
        except Exception:
            continue

    left_cubes = sorted([c for c in cubes if c["x"] < 0], key=lambda c: c["x"], reverse=True)
    right_cubes = sorted([c for c in cubes if c["x"] >= 0], key=lambda c: c["x"])

    obj_map = {}
    for i, c in enumerate(left_cubes):
        obj_map[f"L{i}"] = c["id"]
    for i, c in enumerate(right_cubes):
        obj_map[f"R{i}"] = c["id"]
    return obj_map


def add_highlight_border(img_bgr, active=False, title=""):
    out = img_bgr.copy()
    color = (0, 0, 255) if active else (130, 130, 130)
    thickness = 4 if active else 2
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, thickness)
    if title:
        cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def compose_dashboard(main_rgb, coop_rgb, indep_l_rgb, indep_r_rgb, active_keys, l_state, r_state, left_home_near=False, right_home_near=False):
    main = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2BGR)
    coop = cv2.cvtColor(coop_rgb, cv2.COLOR_RGB2BGR)
    indep_l = cv2.cvtColor(indep_l_rgb, cv2.COLOR_RGB2BGR)
    indep_r = cv2.cvtColor(indep_r_rgb, cv2.COLOR_RGB2BGR)

    main = add_highlight_border(main, active=("main" in active_keys), title="Main")
    coop = add_highlight_border(coop, active=("coop" in active_keys), title="Coop Shadow")
    indep_l = add_highlight_border(indep_l, active=("indep_left" in active_keys), title="Indep Left Shadow")
    indep_r = add_highlight_border(indep_r, active=("indep_right" in active_keys), title="Indep Right Shadow")

    top = np.hstack([main, coop])
    bottom = np.hstack([indep_l, indep_r])
    canvas = np.vstack([top, bottom])

    mixed_mode = (l_state != r_state)
    label = f"HL Mode  L:{l_state}  R:{r_state}"
    if mixed_mode:
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        x0 = 10
        y0 = canvas.shape[0] - 40
        x1 = x0 + text_size[0] + 12
        y1 = y0 + text_size[1] + 12
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 180), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(canvas, label, (x0 + 6, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(canvas, label, (12, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    marker_size = 24
    margin = 12
    if left_home_near:
        lx0 = margin
        ly1 = canvas.shape[0] - margin
        cv2.rectangle(canvas, (lx0, ly1 - marker_size), (lx0 + marker_size, ly1), (0, 255, 255), -1)
        cv2.rectangle(canvas, (lx0, ly1 - marker_size), (lx0 + marker_size, ly1), (0, 200, 200), 2)
    if right_home_near:
        rx1 = canvas.shape[1] - margin
        ry1 = canvas.shape[0] - margin
        cv2.rectangle(canvas, (rx1 - marker_size, ry1 - marker_size), (rx1, ry1), (0, 255, 255), -1)
        cv2.rectangle(canvas, (rx1 - marker_size, ry1 - marker_size), (rx1, ry1), (0, 200, 200), 2)

    return canvas


def max_possible_reward(color_sequence):
    if not color_sequence:
        return 0
    reds = color_sequence.count("r")
    greens = color_sequence.count("g")
    blues = color_sequence.count("b")
    return int(reds + 2 * min(greens, blues))


def read_sequences(sequence_file):
    rows = []
    if not sequence_file:
        return rows
    with open(sequence_file, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 1:
                rows.append(row)
    return rows


def filter_success_sequences(rows):
    success_rows = []
    for row in rows:
        if len(row) >= 3 and str(row[2]).strip().lower() == "success":
            success_rows.append(row)
    return success_rows


def pick_color_sequence(args, seq_rows, success_rows, episode_idx):
    source_rows = seq_rows
    if args.sequence_success_n is not None:
        if not success_rows:
            raise ValueError("No success rows found in sequence_file, but --sequence_success_n was specified.")
        start = int(args.sequence_success_n) - 1
        if start < 0 or start >= len(success_rows):
            raise ValueError(
                f"--sequence_success_n out of range: {args.sequence_success_n} (success rows={len(success_rows)})"
            )
        source_rows = success_rows
        row = source_rows[(start + episode_idx) % len(source_rows)]
    elif source_rows:
        row = source_rows[episode_idx % len(source_rows)]
    else:
        row = None

    if row is not None:
        seq = row[0].strip().lower()
        if len(seq) == 10 and all(c in ["r", "g", "b"] for c in seq):
            return list(seq)
    if args.color_sequence:
        seq = args.color_sequence.strip().lower()
        if len(seq) == 10 and all(c in ["r", "g", "b"] for c in seq):
            return list(seq)
    return None


def pick_color_sequence_with_row(args, seq_rows, success_rows, episode_idx):
    source_rows = seq_rows
    row_idx = None

    if args.sequence_success_n is not None:
        if not success_rows:
            raise ValueError("No success rows found in sequence_file, but --sequence_success_n was specified.")
        start = int(args.sequence_success_n) - 1
        if start < 0 or start >= len(success_rows):
            raise ValueError(
                f"--sequence_success_n out of range: {args.sequence_success_n} (success rows={len(success_rows)})"
            )
        row = success_rows[(start + episode_idx) % len(success_rows)]
        for idx, r in enumerate(seq_rows):
            if r == row:
                row_idx = idx
                break
    elif source_rows:
        row_idx = episode_idx % len(source_rows)
        row = source_rows[row_idx]
    else:
        row = None

    seq = None
    if row is not None:
        s = row[0].strip().lower()
        if len(s) == 10 and all(c in ["r", "g", "b"] for c in s):
            seq = list(s)
    if seq is None and args.color_sequence:
        s = args.color_sequence.strip().lower()
        if len(s) == 10 and all(c in ["r", "g", "b"] for c in s):
            seq = list(s)

    return seq, row_idx


def write_episode_stats(save_path, episode_idx, reward, success, env):
    task = env._task
    col_seq = task.color_sequence if getattr(task, "color_sequence", None) else ["?"] * 10

    indep_set = getattr(task, "completed_independent_cubes", set())
    coop_pairs = getattr(task, "completed_cooperative_pairs", set())
    coop_indices = set()
    for g_idx, b_idx in coop_pairs:
        coop_indices.add(g_idx)
        coop_indices.add(b_idx)

    row = {
        "Episode": episode_idx,
        "Total_Reward": reward,
        "Is_Success": success,
    }

    for i in range(10):
        status = "Fail"
        obj_reward = 0
        if i in indep_set:
            status = "Indep_Success"
            obj_reward = 1
        elif i in coop_indices:
            status = "Coop_Success"
            obj_reward = 2
        row[f"Obj{i}_Color"] = col_seq[i] if i < len(col_seq) else "?"
        row[f"Obj{i}_Status"] = status
        row[f"Obj{i}_Reward"] = obj_reward

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    exists = os.path.isfile(save_path)
    fieldnames = ["Episode", "Total_Reward", "Is_Success"]
    for i in range(10):
        fieldnames.extend([f"Obj{i}_Color", f"Obj{i}_Status", f"Obj{i}_Reward"])

    with open(save_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def compute_episode_object_reward(env):
    task = env._task
    indep_set = set(getattr(task, "completed_independent_cubes", set()))
    coop_pairs = set(getattr(task, "completed_cooperative_pairs", set()))
    coop_count = 2 * len(coop_pairs)
    return int(len(indep_set) + coop_count)


class HierarchicalRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(args.seed)

        self.dual = load_policy_bundle(args.ckpt_dual, args, self.device, state_dim=14)
        self.left = load_policy_bundle(args.ckpt_left, args, self.device, state_dim=7, arm="left")
        self.right = load_policy_bundle(args.ckpt_right, args, self.device, state_dim=7, arm="right")
        self.classifier = load_state_classifier(args.state_ckpt, self.device)

        self.cls_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        task_cfg = SIM_TASK_CONFIGS[args.task_name]
        self.max_steps = args.episode_len if args.episode_len else task_cfg["episode_len"]
        time_limit = (self.max_steps + 200) * DT
        self.env = make_sim_env(args.task_name, camera_names=task_cfg["camera_names"], time_limit=time_limit)
        self.env_shadow_coop = make_sim_env(args.task_name, camera_names=task_cfg["camera_names"], time_limit=time_limit)
        self.env_shadow_indep_left = make_sim_env(args.task_name, camera_names=task_cfg["camera_names"], time_limit=time_limit)
        self.env_shadow_indep_right = make_sim_env(args.task_name, camera_names=task_cfg["camera_names"], time_limit=time_limit)

        self.temporal = not args.no_temporal_agg
        self.bank_dual = TemporalActionBank(args.temporal_agg_k)
        self.bank_left = TemporalActionBank(args.temporal_agg_k)
        self.bank_right = TemporalActionBank(args.temporal_agg_k)
        self.curr_chunk_dual = None
        self.curr_chunk_left = None
        self.curr_chunk_right = None
        self.curr_chunk_dual_t0 = None
        self.curr_chunk_left_t0 = None
        self.curr_chunk_right_t0 = None
        self.active_coop_pair = None
        self.magnetized_pairs = {}
        self.removed_cubes = set()
        self.seq_rows = read_sequences(args.sequence_file)
        self.seq_success_rows = filter_success_sequences(self.seq_rows)
        self.hl_oracle_by_seqrow = collections.defaultdict(list)
        self.hl_oracle_by_episode = {}
        self.hl_oracle_cursor_by_seqrow = collections.defaultdict(int)
        self.current_hl_oracle_schedule = None
        self.shadow_vis_left = []
        self.shadow_vis_right = []
        self.shadow_vis_coop = []
        self.force_shadow_refresh = True
        self._load_hl_oracle_metadata(args.hl_oracle_metadata_csv)
        self._reset_hl_state()

    @staticmethod
    def _parse_int_safe(v, default=-1):
        try:
            return int(v)
        except Exception:
            return default

    @staticmethod
    def _is_valid_color_seq(s):
        s = (s or "").strip().lower()
        return len(s) == 10 and all(c in ["r", "g", "b"] for c in s)

    @staticmethod
    def _mode_char_to_state(ch):
        ch = (ch or "H").upper()
        if ch == "C":
            return STATE_COOP
        if ch == "I":
            return STATE_INDEP
        return STATE_HOLD

    @classmethod
    def _pair_to_states(cls, pair):
        p = (pair or "HH").strip().upper()
        if len(p) < 2:
            p = (p + "HH")[:2]
        return cls._mode_char_to_state(p[0]), cls._mode_char_to_state(p[1])

    def _oracle_states_at_t(self, schedule, step_t):
        pair = schedule["pair0"]
        for tr in schedule["transitions"]:
            if step_t >= tr["t"]:
                pair = tr["pair"]
            else:
                break
        return self._pair_to_states(pair), pair

    def _oracle_shifted_raw_states_at_t(self, schedule, step_t, shift_steps):
        shift_steps = int(max(0, shift_steps))
        src_t = step_t - shift_steps
        if src_t < 0:
            src_t = shift_steps
        (oracle_l, oracle_r), oracle_pair = self._oracle_states_at_t(schedule, src_t)
        return int(oracle_l), int(oracle_r), oracle_pair, int(src_t)

    def _select_oracle_schedule(self, ep_idx, sequence_row_idx):
        if sequence_row_idx is not None and sequence_row_idx in self.hl_oracle_by_seqrow:
            rows = self.hl_oracle_by_seqrow[sequence_row_idx]
            if rows:
                cursor = self.hl_oracle_cursor_by_seqrow[sequence_row_idx]
                picked = rows[cursor % len(rows)]
                self.hl_oracle_cursor_by_seqrow[sequence_row_idx] = cursor + 1
                return picked
        if ep_idx in self.hl_oracle_by_episode:
            return self.hl_oracle_by_episode[ep_idx]
        return None

    def _load_hl_oracle_metadata(self, metadata_csv_path):
        if not metadata_csv_path:
            return
        if not os.path.exists(metadata_csv_path):
            raise FileNotFoundError(f"hl_oracle_metadata_csv not found: {metadata_csv_path}")

        loaded_rows = 0
        print(f"Loading HL oracle metadata: {metadata_csv_path}")
        with open(metadata_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pair0 = (row.get("pair_at_t0") or "HH").strip().upper()
                trans_raw = row.get("transitions_json") or "[]"
                try:
                    transitions = json.loads(trans_raw)
                except Exception:
                    transitions = []

                parsed_transitions = []
                for tr in transitions:
                    try:
                        tr_t = int(tr.get("t", 0))
                        tr_pair = str(tr.get("pair", "HH")).strip().upper()
                        if len(tr_pair) >= 2:
                            parsed_transitions.append({"t": tr_t, "pair": tr_pair[:2]})
                    except Exception:
                        continue
                parsed_transitions.sort(key=lambda x: x["t"])

                entry = {
                    "pair0": pair0[:2] if len(pair0) >= 2 else "HH",
                    "transitions": parsed_transitions,
                    "color_sequence": (row.get("color_sequence") or "").strip().lower(),
                    "command_sequence": (row.get("command_sequence") or "").strip().upper(),
                }

                seq_row = self._parse_int_safe(row.get("sequence_row"), default=-1)
                ep = self._parse_int_safe(row.get("episode"), default=-1)

                if seq_row >= 0:
                    self.hl_oracle_by_seqrow[seq_row].append(entry)
                if ep >= 0:
                    self.hl_oracle_by_episode[ep] = entry
                loaded_rows += 1

        print(f"Loaded HL oracle rows: {loaded_rows}")
        if len(self.hl_oracle_by_episode) > 0 and self.args.num_rollouts > len(self.hl_oracle_by_episode):
            print(
                f"Warning: num_rollouts ({self.args.num_rollouts}) > oracle episodes ({len(self.hl_oracle_by_episode)}). "
                f"Clamping to {len(self.hl_oracle_by_episode)}."
            )
            self.args.num_rollouts = len(self.hl_oracle_by_episode)

    def _reset_hl_state(self):
        self.hl_mode_l = STATE_INDEP
        self.hl_mode_r = STATE_INDEP
        self.last_hl_update_step_l = -1
        self.last_hl_update_step_r = -1
        self.hl_pending_l = collections.deque()
        self.hl_pending_r = collections.deque()
        self.state_history_l = collections.deque(maxlen=self.args.hl_history_len)
        self.state_history_r = collections.deque(maxlen=self.args.hl_history_len)
        self.committed_plan_l_state = "INDEP"
        self.committed_plan_r_state = "INDEP"
        self.steps_since_switch_l = 0
        self.steps_since_switch_r = 0
        self.switch_candidate_l = None
        self.switch_candidate_r = None
        self.switch_candidate_count_l = 0
        self.switch_candidate_count_r = 0
        self.waiting_home_target_l = None
        self.waiting_home_target_r = None
        self.home_pause_count_l = 0
        self.home_pause_count_r = 0
        self.settle_after_switch_l = 0
        self.settle_after_switch_r = 0

    @staticmethod
    def _majority(queue):
        if not queue:
            return STATE_HOLD
        counts = collections.Counter(queue)
        return counts.most_common(1)[0][0]

    @staticmethod
    def _mode_to_plan(mode_int: int):
        if mode_int == STATE_COOP:
            return "COOP"
        if mode_int == STATE_INDEP:
            return "INDEP"
        return "HOLD"

    def _apply_pair_rules(self, t: int, qpos: np.ndarray, home_pose: np.ndarray, plan_l: str, plan_r: str, is_oracle_replay: bool):
        raw_pair = (plan_l, plan_r)
        committed_pair = (self.committed_plan_l_state, self.committed_plan_r_state)

        #if raw_pair in [("HOLD", "COOP"), ("COOP", "HOLD")]:
        #    plan_l, plan_r = "COOP", "COOP"

        mixed_pair = (plan_l, plan_r)
        if mixed_pair in [("INDEP", "COOP"), ("COOP", "INDEP")] and committed_pair != ("COOP", "COOP"):
            plan_l, plan_r = committed_pair

        current_pair = (plan_l, plan_r)
        allowed_pairs = None
        left_home = arm_is_home(qpos[:7], home_pose[:7], self.args.hl_home_threshold)
        right_home = arm_is_home(qpos[7:14], home_pose[7:14], self.args.hl_home_threshold)

        if left_home ^ right_home:
            allowed_pairs = {
                ("INDEP", "COOP"),
                ("COOP", "INDEP"),
                ("INDEP", "INDEP"),
            }
        elif left_home and right_home:
            allowed_pairs = {
                ("INDEP", "INDEP"),
                ("COOP", "COOP"),
                ("HOLD", "INDEP"),
                ("INDEP", "HOLD"),
            }

        if (not is_oracle_replay) and allowed_pairs is not None and current_pair not in allowed_pairs:
            ranked = sorted(
                list(allowed_pairs),
                key=lambda p: (
                    int(p[0] == committed_pair[0]) + int(p[1] == committed_pair[1]),
                    int(p[0] == current_pair[0]) + int(p[1] == current_pair[1]),
                ),
                reverse=True,
            )
            plan_l, plan_r = ranked[0]

        if t == 0:
            self.committed_plan_l_state = plan_l
            self.committed_plan_r_state = plan_r
            self.steps_since_switch_l = 0
            self.steps_since_switch_r = 0
            self.switch_candidate_l = None
            self.switch_candidate_r = None
            self.switch_candidate_count_l = 0
            self.switch_candidate_count_r = 0
            self.waiting_home_target_l = None
            self.waiting_home_target_r = None
            self.home_pause_count_l = 0
            self.home_pause_count_r = 0

        self.steps_since_switch_l += 1
        proposed_l_state = plan_l
        #if plan_l != self.committed_plan_l_state:
        #    if self.steps_since_switch_l > self.args.hl_min_state_duration_steps:
        #        self.committed_plan_l_state = plan_l
        #        self.steps_since_switch_l = 0
        #    else:
        #        plan_l = self.committed_plan_l_state
        plan_l = self.committed_plan_l_state
        self.steps_since_switch_r += 1
        proposed_r_state = plan_r
        #if plan_r != self.committed_plan_r_state:
        #    if self.steps_since_switch_r > self.args.hl_min_state_duration_steps:
        #        self.committed_plan_r_state = plan_r
        #        self.steps_since_switch_r = 0
        #    else:
        #        plan_r = self.committed_plan_r_state
        plan_r = self.committed_plan_r_state
        # state_switcher-style home-gated switching
        force_hold_l = False
        force_hold_r = False
        switch_activated = False

        if proposed_l_state in ["INDEP", "COOP"] and proposed_l_state != self.committed_plan_l_state:
            if self.switch_candidate_l == proposed_l_state:
                self.switch_candidate_count_l += 1
            else:
                self.switch_candidate_l = proposed_l_state
                self.switch_candidate_count_l = 1
        else:
            self.switch_candidate_l = None
            self.switch_candidate_count_l = 0

        if self.waiting_home_target_l is None and self.switch_candidate_l is not None and self.switch_candidate_count_l >= self.args.switch_guard_steps:
            self.waiting_home_target_l = self.switch_candidate_l
            self.home_pause_count_l = 0

        if proposed_r_state in ["INDEP", "COOP"] and proposed_r_state != self.committed_plan_r_state:
            if self.switch_candidate_r == proposed_r_state:
                self.switch_candidate_count_r += 1
            else:
                self.switch_candidate_r = proposed_r_state
                self.switch_candidate_count_r = 1
        else:
            self.switch_candidate_r = None
            self.switch_candidate_count_r = 0

        if self.waiting_home_target_r is None and self.switch_candidate_r is not None and self.switch_candidate_count_r >= self.args.switch_guard_steps:
            self.waiting_home_target_r = self.switch_candidate_r
            self.home_pause_count_r = 0

        if self.waiting_home_target_l is not None:
            if arm_is_home(qpos[:7], home_pose[:7], self.args.switch_home_threshold):
                if self.home_pause_count_l < self.args.switch_home_pause_steps:
                    force_hold_l = True
                    plan_l = "HOLD"
                    self.home_pause_count_l += 1
                else:
                    self.committed_plan_l_state = self.waiting_home_target_l
                    plan_l = self.committed_plan_l_state
                    self.steps_since_switch_l = 0
                    self.waiting_home_target_l = None
                    self.switch_candidate_l = None
                    self.switch_candidate_count_l = 0
                    self.home_pause_count_l = 0
                    self.settle_after_switch_l = max(0, self.args.switch_home_settle_steps)
                    switch_activated = True
            else:
                plan_l = self.committed_plan_l_state

        if self.waiting_home_target_r is not None:
            if arm_is_home(qpos[7:14], home_pose[7:14], self.args.switch_home_threshold):
                if self.home_pause_count_r < self.args.switch_home_pause_steps:
                    force_hold_r = True
                    plan_r = "HOLD"
                    self.home_pause_count_r += 1
                else:
                    self.committed_plan_r_state = self.waiting_home_target_r
                    plan_r = self.committed_plan_r_state
                    self.steps_since_switch_r = 0
                    self.waiting_home_target_r = None
                    self.switch_candidate_r = None
                    self.switch_candidate_count_r = 0
                    self.home_pause_count_r = 0
                    self.settle_after_switch_r = max(0, self.args.switch_home_settle_steps)
                    switch_activated = True
            else:
                plan_r = self.committed_plan_r_state

        if force_hold_l:
            plan_l = "HOLD"
        if force_hold_r:
            plan_r = "HOLD"

        # Mixed-mode COOP-exit rule:
        # If one arm is INDEP and the other is COOP, when the COOP arm reaches home,
        # transition COOP arm out of COOP (to HOLD, then INDEP commit).
        left_home_switch = arm_is_home(qpos[:7], home_pose[:7], self.args.switch_home_threshold)
        right_home_switch = arm_is_home(qpos[7:14], home_pose[7:14], self.args.switch_home_threshold)

        if plan_l == "COOP" and plan_r == "INDEP":
            if left_home_switch:
                plan_l = "HOLD"
                self.committed_plan_l_state = "INDEP"
                self.steps_since_switch_l = 0
                self.waiting_home_target_l = None
                self.switch_candidate_l = None
                self.switch_candidate_count_l = 0
                self.home_pause_count_l = 0
                switch_activated = True

        if plan_l == "INDEP" and plan_r == "COOP":
            if right_home_switch:
                plan_r = "HOLD"
                self.committed_plan_r_state = "INDEP"
                self.steps_since_switch_r = 0
                self.waiting_home_target_r = None
                self.switch_candidate_r = None
                self.switch_candidate_count_r = 0
                self.home_pause_count_r = 0
                switch_activated = True

        return plan_l, plan_r, switch_activated

    def _clear_inference_state(self):
        self.bank_dual = TemporalActionBank(self.args.temporal_agg_k)
        self.bank_left = TemporalActionBank(self.args.temporal_agg_k)
        self.bank_right = TemporalActionBank(self.args.temporal_agg_k)
        self.curr_chunk_dual = None
        self.curr_chunk_left = None
        self.curr_chunk_right = None
        self.curr_chunk_dual_t0 = None
        self.curr_chunk_left_t0 = None
        self.curr_chunk_right_t0 = None

    def _bootstrap_initial_hl_mode(self, ts):
        if self.current_hl_oracle_schedule is not None:
            (oracle_l, oracle_r), _ = self._oracle_states_at_t(self.current_hl_oracle_schedule, 0)
            self.hl_mode_l = int(oracle_l)
            self.hl_mode_r = int(oracle_r)
        else:
            img_np = ts.observation["images"]["top"]
            img = Image.fromarray(img_np.astype("uint8"))
            inp = self.cls_transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out_l, out_r = self.classifier(inp)
                self.hl_mode_l = int(torch.argmax(out_l, dim=1).item())
                self.hl_mode_r = int(torch.argmax(out_r, dim=1).item())

        self.last_hl_update_step_l = 0
        self.last_hl_update_step_r = 0
        self.committed_plan_l_state = self._mode_to_plan(self.hl_mode_l)
        self.committed_plan_r_state = self._mode_to_plan(self.hl_mode_r)

        self.state_history_l.clear()
        self.state_history_r.clear()
        self.state_history_l.append(self.hl_mode_l)
        self.state_history_r.append(self.hl_mode_r)

    @staticmethod
    def _interp_toward_home(current_qpos: np.ndarray, home_qpos: np.ndarray, max_delta: float):
        delta = home_qpos - current_qpos
        step = np.clip(delta, -float(max_delta), float(max_delta))
        return current_qpos + step

    def _select_shadow_targets(self):
        physics = self.env.physics
        spatial_map = get_spatial_object_map(physics)
        color_seq = getattr(self.env._task, "color_sequence", [])
        side_thresh = -0.05

        def color_of(idx):
            if idx is None:
                return None
            if idx < len(color_seq):
                return color_seq[idx]
            return None

        def is_removed_cube_local(idx):
            try:
                geom_id = physics.model.name2id(f"cube_{idx}", "geom")
                alpha = float(physics.model.geom_rgba[geom_id, 3])
            except Exception:
                alpha = 1.0
            try:
                body_id = physics.model.name2id(f"cube_{idx}", "body")
                z_pos = float(physics.data.xpos[body_id][2])
            except Exception:
                z_pos = 0.0
            return (alpha <= 0.01) or (z_pos < -1.0)

        def safe_cube_x_local(idx):
            try:
                bid = physics.model.name2id(f"cube_{idx}", "body")
                return float(physics.data.xpos[bid][0])
            except Exception:
                return None

        final_target_indices_i = [idx for idx in range(10) if not is_removed_cube_local(idx)]

        def pick_single_indep_target(arm, candidates, exclude=None):
            if exclude is None:
                exclude = set()
            side_candidates = []
            for idx in candidates:
                if idx in exclude:
                    continue
                if color_of(idx) != "r":
                    continue
                x_val = safe_cube_x_local(idx)
                if x_val is None:
                    continue
                if arm == "left" and x_val < 0 and x_val > -0.4:
                    side_candidates.append((x_val, idx))
                elif arm == "right" and x_val >= 0 and x_val < 0.4:
                    side_candidates.append((x_val, idx))
            if side_candidates:
                return [max(side_candidates, key=lambda t_: t_[0])[1]]
            return []

        final_target_indices_i_left = pick_single_indep_target("left", final_target_indices_i)
        final_target_indices_i_right = pick_single_indep_target(
            "right",
            final_target_indices_i,
            exclude=set(final_target_indices_i_left),
        )

        grasped_dict = get_grasped_cubes(physics)
        held_indices_left = set(grasped_dict["left"])
        held_indices_right = set(grasped_dict["right"])
        held_indices = held_indices_left.union(held_indices_right)

        for idx in held_indices_left:
            if is_removed_cube_local(idx):
                continue
            if color_of(idx) == "r" and idx not in final_target_indices_i_left:
                final_target_indices_i_left.append(idx)

        for idx in held_indices_right:
            if is_removed_cube_local(idx):
                continue
            if color_of(idx) == "r" and idx not in final_target_indices_i_right:
                final_target_indices_i_right.append(idx)

        coop_candidates = []
        for _, c_idx in spatial_map.items():
            try:
                if is_removed_cube_local(c_idx):
                    continue
                bid = physics.model.name2id(f"cube_{c_idx}", "body")
                x_pos = float(physics.data.xpos[bid][0])
                color = color_of(c_idx)
                if x_pos < 0.3 and color in ["g", "b"]:
                    coop_candidates.append((x_pos, c_idx, color))
            except Exception:
                continue

        latest_target_indices_c = []
        if coop_candidates:
            g_items = [c for c in coop_candidates if c[2] == "g"]
            b_items = [c for c in coop_candidates if c[2] == "b"]
            if g_items and b_items:
                latest_target_indices_c = [max(g_items, key=lambda x: x[0])[1], max(b_items, key=lambda x: x[0])[1]]

        coop_pair = []
        if self.active_coop_pair is not None:
            g_idx, b_idx = self.active_coop_pair
            if (not is_removed_cube_local(g_idx)) and (not is_removed_cube_local(b_idx)):
                coop_pair = [g_idx, b_idx]
            else:
                self.active_coop_pair = None
        if not coop_pair:
            coop_pool = set(latest_target_indices_c)
            for h_idx in held_indices:
                if color_of(h_idx) in ["g", "b"] and (not is_removed_cube_local(h_idx)):
                    coop_pool.add(h_idx)

            g_candidates = []
            b_candidates = []
            for c_idx in coop_pool:
                color = color_of(c_idx)
                if color not in ["g", "b"]:
                    continue
                x_pos = safe_cube_x_local(c_idx)
                if x_pos is None:
                    x_pos = -1e9
                if color == "g":
                    g_candidates.append((x_pos, c_idx))
                else:
                    b_candidates.append((x_pos, c_idx))

            if g_candidates and b_candidates:
                g_best = max(g_candidates, key=lambda x: x[0])[1]
                b_best = max(b_candidates, key=lambda x: x[0])[1]
                self.active_coop_pair = (g_best, b_best)
                coop_pair = [g_best, b_best]

        return final_target_indices_i_left, final_target_indices_i_right, coop_pair

    def _build_observation_map(self, ts, refresh_targets=True):
        sync_envs(self.env.physics, self.env_shadow_coop.physics)
        sync_envs(self.env.physics, self.env_shadow_indep_left.physics)
        sync_envs(self.env.physics, self.env_shadow_indep_right.physics)

        # Match state_switcher behavior: independent targets update every step.
        vis_left, vis_right, vis_coop = self._select_shadow_targets()
        

        # Cooperative target refresh remains gated by stage/force flag.
        if refresh_targets or self.force_shadow_refresh:
            self.shadow_vis_left = list(vis_left)
            self.shadow_vis_right = list(vis_right)
            self.shadow_vis_coop = list(vis_coop)
            self.force_shadow_refresh = False

        set_visible_indices(self.env_shadow_coop.physics, self.shadow_vis_coop)
        set_visible_indices(self.env_shadow_indep_left.physics, self.shadow_vis_left)
        set_visible_indices(self.env_shadow_indep_right.physics, self.shadow_vis_right)

        obs_main = ts.observation
        obs_coop = self.env_shadow_coop.task.get_observation(self.env_shadow_coop.physics)
        obs_indep_l = self.env_shadow_indep_left.task.get_observation(self.env_shadow_indep_left.physics)
        obs_indep_r = self.env_shadow_indep_right.task.get_observation(self.env_shadow_indep_right.physics)
        return {
            "main": obs_main,
            "coop": obs_coop,
            "indep_left": obs_indep_l,
            "indep_right": obs_indep_r,
        }

    def _predict_mode(self, ts, t, home_pose):
        qpos = np.array(ts.observation["qpos"])
        debug_info = {
            "raw_source": "none",
            "raw_pred_l": "",
            "raw_pred_r": "",
            "maj_count_l": 0,
            "maj_count_r": 0,
            "maj_mode_l": STATE_NAME.get(self.hl_mode_l, str(self.hl_mode_l)),
            "maj_mode_r": STATE_NAME.get(self.hl_mode_r, str(self.hl_mode_r)),
        }
        hl_update_tick_l = (t == 0)
        hl_update_tick_r = (t == 0)
        if self.args.hl_update_at_home_only and not (hl_update_tick_l or hl_update_tick_r):
            left_home = arm_is_home(qpos[:7], home_pose[:7], self.args.hl_home_threshold)
            right_home = arm_is_home(qpos[7:14], home_pose[7:14], self.args.hl_home_threshold)
            left_open = qpos[6] > self.args.hl_home_gripper_threshold
            right_open = qpos[13] > self.args.hl_home_gripper_threshold
            interval_ready = (t % self.args.hl_update_interval == 0)
            hl_update_tick_l = left_home and left_open and interval_ready
            hl_update_tick_r = right_home and right_open and interval_ready

        if self.args.hl_dense_inference:
            should_update_hl = True
        else:
            should_update_hl = (hl_update_tick_l or hl_update_tick_r) if self.args.hl_update_at_home_only else True

        if should_update_hl:
            if self.current_hl_oracle_schedule is not None:
                pred_l, pred_r, oracle_pair, src_t = self._oracle_shifted_raw_states_at_t(
                    self.current_hl_oracle_schedule,
                    t,
                    self.args.hl_lookahead_steps,
                )
                debug_info["raw_source"] = "oracle_shifted"
                debug_info["raw_pred_l"] = STATE_NAME.get(int(pred_l), str(pred_l))
                debug_info["raw_pred_r"] = STATE_NAME.get(int(pred_r), str(pred_r))
                if t % 50 == 0:
                    print(f"[Step {t}] HL oracle raw(pair={oracle_pair}, src_t={src_t})")
            else:
                img_np = ts.observation["images"]["top"]
                img = Image.fromarray(img_np.astype("uint8"))
                inp = self.cls_transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out_l, out_r = self.classifier(inp)
                    pred_l = int(torch.argmax(out_l, dim=1).item())
                    pred_r = int(torch.argmax(out_r, dim=1).item())
                debug_info["raw_source"] = "classifier"
                debug_info["raw_pred_l"] = STATE_NAME.get(pred_l, str(pred_l))
                debug_info["raw_pred_r"] = STATE_NAME.get(pred_r, str(pred_r))

            if t == 0:
                for i in range(int(self.args.hl_lookahead_steps) + 1):
                    fill_t = t + i
                    if self.args.hl_dense_inference or (not self.args.hl_update_at_home_only) or hl_update_tick_l:
                        self.hl_pending_l.append((fill_t, int(pred_l)))
                    if self.args.hl_dense_inference or (not self.args.hl_update_at_home_only) or hl_update_tick_r:
                        self.hl_pending_r.append((fill_t, int(pred_r)))
            else:
                target_t = t + int(self.args.hl_lookahead_steps)
                if self.args.hl_dense_inference or (not self.args.hl_update_at_home_only) or hl_update_tick_l:
                    self.hl_pending_l.append((target_t, int(pred_l)))
                if self.args.hl_dense_inference or (not self.args.hl_update_at_home_only) or hl_update_tick_r:
                    self.hl_pending_r.append((target_t, int(pred_r)))
                        
        future_window = int(self.args.hl_future_window_steps)
        window_end_t = t + future_window
        prune_before_t = t - int(self.args.hl_lookahead_steps) - future_window

        while self.hl_pending_l and self.hl_pending_l[0][0] < prune_before_t:
            self.hl_pending_l.popleft()
        while self.hl_pending_r and self.hl_pending_r[0][0] < prune_before_t:
            self.hl_pending_r.popleft()

        cand_l = [mode for target_t, mode in self.hl_pending_l if t <= target_t <= window_end_t]
        cand_r = [mode for target_t, mode in self.hl_pending_r if t <= target_t <= window_end_t]
        debug_info["maj_count_l"] = len(cand_l)
        debug_info["maj_count_r"] = len(cand_r)

        if cand_l:
            self.hl_mode_l = int(self._majority(cand_l))
            self.last_hl_update_step_l = t
            debug_info["maj_mode_l"] = STATE_NAME.get(self.hl_mode_l, str(self.hl_mode_l))
        if cand_r:
            self.hl_mode_r = int(self._majority(cand_r))
            self.last_hl_update_step_r = t
            debug_info["maj_mode_r"] = STATE_NAME.get(self.hl_mode_r, str(self.hl_mode_r))

        #s_l = self.hl_mode_l
        #s_r = self.hl_mode_r
        #self.state_history_l.append(s_l)
        #self.state_history_r.append(s_r)
        #s_l_smooth = self._majority(self.state_history_l)
        #s_r_smooth = self._majority(self.state_history_r)
        s_l_smooth = self.hl_mode_l
        s_r_smooth = self.hl_mode_r
        if s_l_smooth == STATE_COOP and s_r_smooth == STATE_COOP:
            plan_l_state, plan_r_state = "COOP", "COOP"
        else:
            plan_l_state = self._mode_to_plan(s_l_smooth)
            plan_r_state = self._mode_to_plan(s_r_smooth)

        plan_l_state, plan_r_state, switch_activated = self._apply_pair_rules(
            t,
            qpos,
            home_pose,
            plan_l_state,
            plan_r_state,
            is_oracle_replay=(self.current_hl_oracle_schedule is not None),
        )
        if switch_activated:
            self._clear_inference_state()
            self.force_shadow_refresh = True
        return plan_l_state, plan_r_state, debug_info

    def _query_chunks(self, ts, obs_map, t, l_state, r_state):
        qpos = np.array(ts.observation["qpos"])

        need_dual = (l_state == "COOP") or (r_state == "COOP")
        need_left = l_state == "INDEP"
        need_right = r_state == "INDEP"

        with torch.no_grad():
            dual_expired = (
                self.curr_chunk_dual is None
                or self.curr_chunk_dual_t0 is None
                or (t - self.curr_chunk_dual_t0) >= self.args.chunk_size
            )
            if need_dual and (self.temporal or dual_expired):
                q = torch.from_numpy(self.dual.pre(qpos)).float().to(self.device).unsqueeze(0)
                img = get_dual_image(obs_map["coop"], self.device)
                chunk = self.dual.model(q, img).squeeze(0).detach().cpu().numpy()
                if self.temporal:
                    self.bank_dual.add(t, chunk)
                else:
                    self.curr_chunk_dual = chunk
                    self.curr_chunk_dual_t0 = t

            left_expired = (
                self.curr_chunk_left is None
                or self.curr_chunk_left_t0 is None
                or (t - self.curr_chunk_left_t0) >= self.args.chunk_size
            )
            if need_left and (self.temporal or left_expired):
                q_l = torch.from_numpy(self.left.pre(qpos[:7])).float().to(self.device).unsqueeze(0)
                img_l = get_indep_image(obs_map["indep_left"], "left", self.device)
                chunk_l = self.left.model(q_l, img_l).squeeze(0).detach().cpu().numpy()
                if self.temporal:
                    self.bank_left.add(t, chunk_l)
                else:
                    self.curr_chunk_left = chunk_l
                    self.curr_chunk_left_t0 = t

            right_expired = (
                self.curr_chunk_right is None
                or self.curr_chunk_right_t0 is None
                or (t - self.curr_chunk_right_t0) >= self.args.chunk_size
            )
            if need_right and (self.temporal or right_expired):
                q_r = torch.from_numpy(self.right.pre(qpos[7:14])).float().to(self.device).unsqueeze(0)
                img_r = get_indep_image(obs_map["indep_right"], "right", self.device)
                chunk_r = self.right.model(q_r, img_r).squeeze(0).detach().cpu().numpy()
                if self.temporal:
                    self.bank_right.add(t, chunk_r)
                else:
                    self.curr_chunk_right = chunk_r
                    self.curr_chunk_right_t0 = t

        return qpos

    def _decode_action(self, t, qpos, l_state, r_state):
        if self.temporal:
            dual_now = self.bank_dual.get(t)
            left_now = self.bank_left.get(t)
            right_now = self.bank_right.get(t)
        else:
            dual_now = None
            left_now = None
            right_now = None

            if self.curr_chunk_dual is not None and self.curr_chunk_dual_t0 is not None:
                idx = t - self.curr_chunk_dual_t0
                if 0 <= idx < len(self.curr_chunk_dual):
                    dual_now = self.curr_chunk_dual[idx]
            if self.curr_chunk_left is not None and self.curr_chunk_left_t0 is not None:
                idx = t - self.curr_chunk_left_t0
                if 0 <= idx < len(self.curr_chunk_left):
                    left_now = self.curr_chunk_left[idx]
            if self.curr_chunk_right is not None and self.curr_chunk_right_t0 is not None:
                idx = t - self.curr_chunk_right_t0
                if 0 <= idx < len(self.curr_chunk_right):
                    right_now = self.curr_chunk_right[idx]

        if l_state == "COOP" and dual_now is not None:
            action_l = self.dual.post(dual_now)[:7]
        elif l_state == "INDEP" and left_now is not None:
            action_l = self.left.post(left_now)
        else:
            action_l = qpos[:7]

        if r_state == "COOP" and dual_now is not None:
            action_r = self.dual.post(dual_now)[7:14]
        elif r_state == "INDEP" and right_now is not None:
            action_r = self.right.post(right_now)
        else:
            action_r = qpos[7:14]

        return np.concatenate([action_l, action_r])

    def run(self):
        total_rewards = []
        success_count = 0
        rehome_stage = "none"
        post_rehome_lock_steps = 0  # ★追加: ロック用カウンタ
        hl_debug_writer = None
        if self.args.save_hl_debug_csv:
            hl_debug_dir = os.path.dirname(self.args.save_hl_debug_csv)
            if hl_debug_dir:
                os.makedirs(hl_debug_dir, exist_ok=True)
            file_exists = os.path.isfile(self.args.save_hl_debug_csv)
            hl_debug_f = open(self.args.save_hl_debug_csv, "a", newline="")
            hl_debug_fields = [
                "Episode", "Step", "Raw_Source", "Raw_L", "Raw_R",
                "Maj_Count_L", "Maj_Mode_L", "Maj_Count_R", "Maj_Mode_R",
                "Plan_L", "Plan_R", "Mode_Changed", "Adopted_L", "Adopted_R", "Rehome_Stage",
            ]
            hl_debug_writer = csv.DictWriter(hl_debug_f, fieldnames=hl_debug_fields)
            if not file_exists:
                hl_debug_writer.writeheader()

        for ep in range(self.args.num_rollouts):
            seq, sequence_row_idx = pick_color_sequence_with_row(self.args, self.seq_rows, self.seq_success_rows, ep)
            self.current_hl_oracle_schedule = self._select_oracle_schedule(ep, sequence_row_idx)
            if self.current_hl_oracle_schedule is not None:
                oracle_seq = self.current_hl_oracle_schedule.get("color_sequence", "")
                if self._is_valid_color_seq(oracle_seq):
                    seq = list(oracle_seq)
            if seq is not None:
                MANYCUBES_COLORS[0] = seq

            ts = self.env.reset()
            self.env_shadow_coop.reset()
            self.env_shadow_indep_left.reset()
            self.env_shadow_indep_right.reset()
            home_pose = np.array(ts.observation["qpos"]).copy()
            ep_reward = 0
            video_frames = []

            self._clear_inference_state()
            self.active_coop_pair = None
            self.magnetized_pairs = {}
            self.removed_cubes = set()
            self.shadow_vis_left = []
            self.shadow_vis_right = []
            self.shadow_vis_coop = []
            self.force_shadow_refresh = True
            self._reset_hl_state()
            self._bootstrap_initial_hl_mode(ts)
            reset_magnet_logic(self.env.physics)

            # Post-success reset flow:
            # 1) wait arms near home
            # 2) smoothly return to exact home pose
            # 3) clear chunk state and resume fresh inference
            rehome_stage = "none"  # none | wait_near_home | smooth_to_home
            rehome_settle_left = 0
            rehome_trigger_pending = False
            rehome_detect_threshold = 0.1
            rehome_interp_delta = 0.01
            rehome_settle_steps = 10
            prev_mode_pair = None
            prev_l_state = None
            prev_r_state = None
            coop_pair_contact_streak = 0
            coop_fail_recover_active = False
            coop_fail_pair = None
            coop_fail_recover_total_steps = 50
            coop_fail_recover_step = 0
            coop_fail_recover_start_qpos = None
            coop_contact_pair_key = None
            coop_contact_ready = False
            coop_pair_release_streak = 0
            last_hl_debug_info = {
                "raw_source": "none",
                "raw_pred_l": "",
                "raw_pred_r": "",
                "maj_count_l": 0,
                "maj_count_r": 0,
                "maj_mode_l": STATE_NAME.get(self.hl_mode_l, str(self.hl_mode_l)),
                "maj_mode_r": STATE_NAME.get(self.hl_mode_r, str(self.hl_mode_r)),
            }

            for t in range(self.max_steps):
                qpos = np.array(ts.observation["qpos"])
                l_state = "HOLD"
                r_state = "HOLD"
                saved_frame_bgr = None
                mode_changed = False

                if rehome_trigger_pending and rehome_stage == "none":
                    rehome_stage = "wait_near_home"
                    rehome_trigger_pending = False

                refresh_targets = rehome_stage == "none"
                obs_map = self._build_observation_map(ts, refresh_targets=refresh_targets)
                pred_l, pred_r, debug_info_step = self._predict_mode(ts, t, home_pose)
                last_hl_debug_info = debug_info_step

                if coop_fail_recover_active:
                    l_state = "HOLD"
                    r_state = "HOLD"
                    coop_fail_recover_step += 1
                    alpha = min(1.0, float(coop_fail_recover_step) / float(coop_fail_recover_total_steps))
                    action = coop_fail_recover_start_qpos + alpha * (home_pose - coop_fail_recover_start_qpos)
                    if coop_fail_recover_step >= coop_fail_recover_total_steps:
                        coop_fail_recover_active = False
                        if coop_fail_pair is not None:
                            remove_cubes(self.env.physics, set(coop_fail_pair))
                            self.removed_cubes.update(set(coop_fail_pair))
                            g_idx, _ = coop_fail_pair
                            self.magnetized_pairs.pop(g_idx, None)
                            self.active_coop_pair = None
                            self.force_shadow_refresh = True
                        coop_fail_pair = None

                elif rehome_stage == "wait_near_home":
                    #l_state, r_state = self._predict_mode(ts, t, home_pose)
                    l_state = self.committed_plan_l_state
                    r_state = self.committed_plan_r_state

                    qpos = self._query_chunks(ts, obs_map, t, l_state, r_state)
                    action = self._decode_action(t, qpos, l_state, r_state)

                    left_near = arm_is_home(qpos[:7], home_pose[:7], rehome_detect_threshold)
                    right_near = arm_is_home(qpos[7:14], home_pose[7:14], rehome_detect_threshold)
                    if left_near and right_near:
                        rehome_stage = "smooth_to_home"
                        rehome_settle_left = rehome_settle_steps
                elif rehome_stage == "smooth_to_home":
                    # Smoothly converge to exact home and hold for a few clean steps.
                    action = self._interp_toward_home(qpos, home_pose, max_delta=rehome_interp_delta * 0.8)
                    rehome_settle_left -= 1
                    if rehome_settle_left <= 0:
                        rehome_stage = "none"
                        self._clear_inference_state()
                        self.active_coop_pair = None
                        self.force_shadow_refresh = True
                        #self._reset_hl_state()
                        # Prevent immediate HOLD re-entry right after rehome completion.
                        # Keep freshly reset INDEP state for a short cooldown window.
                        self.steps_since_switch_l = 0
                        self.steps_since_switch_r = 0
                        # 2. ★追加: ロック期間を設定 (30ステップ = 約0.6秒)
                        #    この期間中は、画像認識もルール判定も一切行わず、
                        #    ひたすら現在のモードを維持して静止させます。
                        post_rehome_lock_steps = 30
                        # 3. 判定中のスイッチ候補もクリアしておく
                        self.waiting_home_target_l = None
                        self.waiting_home_target_r = None
                        self.switch_candidate_l = None
                        self.switch_candidate_r = None
                        self.switch_candidate_count_l = 0
                        self.switch_candidate_count_r = 0
                        self.home_pause_count_l = 0
                        self.home_pause_count_r = 0
                        #cooldown_steps = int(self.args.post_rehome_hl_cooldown_steps)
                        #self.last_hl_update_step_l = t - self.args.hl_update_interval + cooldown_steps
                        #self.last_hl_update_step_r = t - self.args.hl_update_interval + cooldown_steps
                else:
                    #l_state, r_state = self._predict_mode(ts, t, home_pose)
                    if post_rehome_lock_steps > 0:
                        post_rehome_lock_steps -= 1
                        l_state = self.committed_plan_l_state
                        r_state = self.committed_plan_r_state
                    else:
                        # ロックが明けたら通常通り予測を開始
                        l_state = pred_l
                        r_state = pred_r
                    qpos = self._query_chunks(ts, obs_map, t, l_state, r_state)
                    action = self._decode_action(t, qpos, l_state, r_state)

                # HOLD handling:
                # - During HOLD, smoothly move that arm toward home.
                # - When HOLD is released, clear inference state for fresh chunks.
                hold_released_l = (prev_l_state == "HOLD") and (l_state != "HOLD")
                hold_released_r = (prev_r_state == "HOLD") and (r_state != "HOLD")
                if hold_released_l or hold_released_r:
                    self._clear_inference_state()
                    self.force_shadow_refresh = True

                if rehome_stage != "smooth_to_home":
                    action = action.copy()
                    if l_state == "HOLD":
                        action[:7] = self._interp_toward_home(
                            qpos[:7], home_pose[:7], max_delta=self.args.hold_home_interp_delta
                        )
                    if r_state == "HOLD":
                        action[7:14] = self._interp_toward_home(
                            qpos[7:14], home_pose[7:14], max_delta=self.args.hold_home_interp_delta
                        )

                curr_mode_pair = (l_state, r_state)
                if curr_mode_pair != prev_mode_pair:
                    mode_changed = True
                    prev_text = "None" if prev_mode_pair is None else f"{prev_mode_pair[0]}/{prev_mode_pair[1]}"
                    print(
                        f"[Episode {ep+1} Step {t}] Mode changed: {prev_text} -> "
                        f"{l_state}/{r_state} (rehome_stage={rehome_stage})"
                    )
                    prev_mode_pair = curr_mode_pair

                if hl_debug_writer is not None and (t % int(self.args.hl_debug_interval_steps) == 0):
                    hl_debug_writer.writerow({
                        "Episode": ep + 1,
                        "Step": t,
                        "Raw_Source": last_hl_debug_info.get("raw_source", "none"),
                        "Raw_L": last_hl_debug_info.get("raw_pred_l", ""),
                        "Raw_R": last_hl_debug_info.get("raw_pred_r", ""),
                        "Maj_Count_L": int(last_hl_debug_info.get("maj_count_l", 0)),
                        "Maj_Mode_L": last_hl_debug_info.get("maj_mode_l", ""),
                        "Maj_Count_R": int(last_hl_debug_info.get("maj_count_r", 0)),
                        "Maj_Mode_R": last_hl_debug_info.get("maj_mode_r", ""),
                        "Plan_L": l_state,
                        "Plan_R": r_state,
                        "Mode_Changed": int(mode_changed),
                        "Adopted_L": l_state if mode_changed else "",
                        "Adopted_R": r_state if mode_changed else "",
                        "Rehome_Stage": rehome_stage,
                    })

                ts = self.env.step(action)

                # Magnet logic (visual cooperative assembly glue)
                curr_colors = getattr(self.env._task, "color_sequence", None)
                apply_magnet_logic(self.env.physics, self.magnetized_pairs, curr_colors)

                # COOP failure detection for active BG pair:
                # after >20 consecutive gripper-contact steps (same pair), if magnet not activated and
                # object is low (z<threshold) and non-contact persists, trigger fail recovery.
                in_coop_mode = (l_state == "COOP") or (r_state == "COOP")
                if (not coop_fail_recover_active) and in_coop_mode and (self.active_coop_pair is not None):
                    try:
                        g_idx, b_idx = self.active_coop_pair
                        pair_key = (int(g_idx), int(b_idx))
                        if pair_key != coop_contact_pair_key:
                            coop_contact_pair_key = pair_key
                            coop_pair_contact_streak = 0
                            coop_contact_ready = False
                            coop_pair_release_streak = 0

                        touched = get_touched_cubes_per_arm(self.env.physics)
                        grasped = get_grasped_cubes(self.env.physics)
                        touching_now = (
                            (g_idx in touched["left"]) or (g_idx in touched["right"]) or
                            (b_idx in touched["left"]) or (b_idx in touched["right"])
                        )
                        grasping_now = (
                            (g_idx in grasped["left"]) or (g_idx in grasped["right"]) or
                            (b_idx in grasped["left"]) or (b_idx in grasped["right"])
                        )

                        # Count prerequisite contact only when the pair is actually grasped,
                        # not for incidental brush contacts.
                        if touching_now and grasping_now:
                            coop_pair_contact_streak += 1
                            coop_pair_release_streak = 0
                            if coop_pair_contact_streak > 20:
                                coop_contact_ready = True
                        else:
                            if coop_contact_ready:
                                coop_pair_release_streak += 1
                            else:
                                coop_pair_contact_streak = 0
                                coop_pair_release_streak = 0

                        magnet_active = (
                            (g_idx in self.magnetized_pairs) and
                            (self.magnetized_pairs[g_idx].get("blue_idx", None) == b_idx)
                        )
                        if magnet_active:
                            coop_contact_ready = False
                            coop_pair_contact_streak = 0
                            coop_pair_release_streak = 0

                        g_bid = self.env.physics.model.name2id(f"cube_{g_idx}", "body")
                        b_bid = self.env.physics.model.name2id(f"cube_{b_idx}", "body")
                        z_g = float(self.env.physics.data.xpos[g_bid][2])
                        z_b = float(self.env.physics.data.xpos[b_bid][2])
                        low_z = (z_g < 0.05) or (z_b < 0.05)

                        if coop_contact_ready and (coop_pair_release_streak >= 3) and (not magnet_active) and low_z and (not touching_now) and (not grasping_now):
                            coop_fail_recover_active = True
                            coop_fail_pair = (int(g_idx), int(b_idx))
                            coop_fail_recover_step = 0
                            coop_fail_recover_start_qpos = np.array(ts.observation["qpos"]).copy()
                            rehome_stage = "none"
                            rehome_trigger_pending = False
                            self._clear_inference_state()
                            print(
                                f"[Episode {ep+1} Step {t}] COOP fail detected for pair G{g_idx}+B{b_idx} "
                                f"(contact_streak={coop_pair_contact_streak}, release_streak={coop_pair_release_streak}, "
                                f"magnet={magnet_active}, z=({z_g:.3f},{z_b:.3f}))"
                            )
                            coop_pair_contact_streak = 0
                            coop_pair_release_streak = 0
                            coop_contact_ready = False
                    except Exception:
                        pass
                elif self.active_coop_pair is None:
                    coop_pair_contact_streak = 0
                    coop_pair_release_streak = 0
                    coop_contact_ready = False
                    coop_contact_pair_key = None
                elif not in_coop_mode:
                    coop_pair_contact_streak = 0
                    coop_pair_release_streak = 0
                    coop_contact_ready = False

                # Removal logic (goal / cushion / completed-task based)
                in_goal = get_cubes_in_goal(self.env.physics)
                on_cushion = get_cubes_on_cushion(self.env.physics)
                completed_indep = set(getattr(self.env._task, "completed_independent_cubes", set()))
                completed_pairs = set(getattr(self.env._task, "completed_cooperative_pairs", set()))
                completed_coop = set()
                for g_idx, b_idx in completed_pairs:
                    completed_coop.add(g_idx)
                    completed_coop.add(b_idx)

                to_remove = (set(in_goal) | set(on_cushion) | completed_indep | completed_coop) - self.removed_cubes
                if to_remove:
                    remove_cubes(self.env.physics, to_remove)
                    self.removed_cubes.update(to_remove)
                    # Ensure shadow target selection is recomputed right away,
                    # even during wait/smooth rehome stages.
                    #self.force_shadow_refresh = True
                    # Trigger post-success clean reset flow on the next control step.
                    rehome_trigger_pending = True

                if self.args.onscreen_render:
                    qpos_now = np.array(ts.observation["qpos"])
                    left_home_near = arm_is_home(qpos_now[:7], home_pose[:7], self.args.hl_home_threshold)
                    right_home_near = arm_is_home(qpos_now[7:14], home_pose[7:14], self.args.hl_home_threshold)

                    active_views = set()
                    if l_state == "COOP" or r_state == "COOP":
                        active_views.add("coop")
                    if l_state == "INDEP":
                        active_views.add("indep_left")
                    if r_state == "INDEP":
                        active_views.add("indep_right")

                    dash_bgr = compose_dashboard(
                        obs_map["main"]["images"]["top"],
                        obs_map["coop"]["images"]["top"],
                        obs_map["indep_left"]["images"]["top"],
                        obs_map["indep_right"]["images"]["top"],
                        active_views,
                        l_state,
                        r_state,
                        left_home_near=left_home_near,
                        right_home_near=right_home_near,
                    )
                    saved_frame_bgr = dash_bgr
                    cv2.imshow("hierarchical_policy", dash_bgr)
                    cv2.waitKey(1)

                if self.args.save_video:
                    if saved_frame_bgr is not None:
                        video_frames.append(saved_frame_bgr.copy())
                    else:
                        video_frames.append(ts.observation["images"]["top"].copy())

                prev_l_state = l_state
                prev_r_state = r_state

            color_seq = getattr(self.env._task, "color_sequence", [])
            ep_reward = compute_episode_object_reward(self.env)
            max_reward = max_possible_reward(color_seq)
            success = ep_reward >= max_reward and max_reward > 0
            if success:
                success_count += 1
            total_rewards.append(ep_reward)

            print(
                f"Episode {ep+1}/{self.args.num_rollouts} | "
                f"Reward={ep_reward}/{max_reward} | Success={success}"
            )

            if self.args.save_stats_path:
                write_episode_stats(self.args.save_stats_path, ep + 1, ep_reward, success, self.env)

            if self.args.save_video and video_frames:
                out_dir = self.args.save_video if isinstance(self.args.save_video, str) else "videos"
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"hier_ep{ep+1}_r{int(ep_reward)}.mp4")
                h, w, _ = video_frames[0].shape
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                for frame in video_frames:
                    if self.args.onscreen_render:
                        writer.write(frame)
                    else:
                        writer.write(frame[:, :, [2, 1, 0]])
                writer.release()

        if hl_debug_writer is not None:
            hl_debug_f.close()

        if total_rewards:
            print("=" * 40)
            print(f"Average Reward: {np.mean(total_rewards):.2f}")
            print(f"Success Rate: {100.0 * success_count / len(total_rewards):.1f}%")
            print("=" * 40)


def parse_args():
    parser = argparse.ArgumentParser(description="Simplified hierarchical policy runner")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--ckpt_dual", type=str, required=True)
    parser.add_argument("--ckpt_left", type=str, required=True)
    parser.add_argument("--ckpt_right", type=str, required=True)
    parser.add_argument("--state_ckpt", type=str, required=True)

    parser.add_argument("--policy_class", type=str, default="ACT")
    parser.add_argument("--chunk_size", type=int, default=75)
    parser.add_argument("--kl_weight", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument("--num_rollouts", type=int, default=10)
    parser.add_argument("--episode_len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2)

    parser.add_argument("--hl_update_interval", type=int, default=1)
    parser.add_argument(
        "--hl_lookahead_steps",
        type=int,
        default=50,
        help="Classifier lookahead steps used during training. Predictions are delayed by this amount to align with true mode at time t.",
    )
    parser.add_argument(
        "--hl_future_window_steps",
        type=int,
        default=30,
        help="Use majority vote of predictions whose target_t is in [t, t+window] as the current high-level mode.",
    )
    parser.add_argument(
        "--hl_update_at_home_only",
        dest="hl_update_at_home_only",
        action="store_true",
        default=True,
        help="Update high-level classifier only when arm(s) are near home. Default: enabled.",
    )
    parser.add_argument(
        "--hl_update_anytime",
        dest="hl_update_at_home_only",
        action="store_false",
        help="Disable home-only gate and allow high-level updates every step.",
    )
    parser.add_argument(
        "--hl_dense_inference",
        dest="hl_dense_inference",
        action="store_true",
        default=True,
        help="Run high-level classifier every step and enqueue predictions densely. Default: enabled.",
    )
    parser.add_argument(
        "--no_hl_dense_inference",
        dest="hl_dense_inference",
        action="store_false",
        help="Disable dense high-level inference and use legacy gated inference timing.",
    )
    parser.add_argument("--hl_home_threshold", type=float, default=0.1)
    parser.add_argument(
        "--hl_home_gripper_threshold",
        type=float,
        default=0.8,
        help="Gripper-open threshold for home-gated high-level updates.",
    )
    parser.add_argument("--hold_home_interp_delta", type=float, default=0.01)
    parser.add_argument("--hl_history_len", type=int, default=10)
    parser.add_argument("--hl_min_state_duration_steps", type=int, default=100)
    parser.add_argument(
        "--switch_guard_steps",
        type=int,
        default=8,
        help="Stable high-level prediction steps required before requesting mode switch.",
    )
    parser.add_argument(
        "--switch_home_pause_steps",
        type=int,
        default=15,
        help="Pause steps at home before activating next policy.",
    )
    parser.add_argument(
        "--switch_home_settle_steps",
        type=int,
        default=6,
        help="Additional settle steps after switch activation.",
    )
    parser.add_argument(
        "--switch_home_threshold",
        type=float,
        default=0.12,
        help="Home detection threshold for gated switching.",
    )
    parser.add_argument(
        "--hl_switch_guard_steps",
        dest="switch_guard_steps",
        type=int,
        help="Alias of --switch_guard_steps.",
    )
    parser.add_argument(
        "--post_rehome_hl_cooldown_steps",
        type=int,
        default=0,
        help="Cooldown steps after smooth_to_home before allowing new HL updates.",
    )

    parser.add_argument(
        "--no_temporal_agg",
        dest="no_temporal_agg",
        action="store_true",
        default=True,
        help="Disable temporal aggregation. Default: disabled temporal aggregation.",
    )
    parser.add_argument(
        "--temporal_agg",
        dest="no_temporal_agg",
        action="store_false",
        help="Enable temporal aggregation.",
    )
    parser.add_argument("--temporal_agg_k", type=float, default=0.03)

    parser.add_argument("--sequence_file", type=str, default=None)
    parser.add_argument(
        "--sequence_success_n",
        type=int,
        default=None,
        help="Use success-only rows and start from the N-th success sequence (1-based).",
    )
    parser.add_argument("--color_sequence", type=str, default=None)
    parser.add_argument("--hl_oracle_metadata_csv", type=str, default=None)

    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--save_video", nargs="?", const="videos", type=str)
    parser.add_argument("--save_stats_path", type=str, default=None)
    parser.add_argument("--save_hl_debug_csv", type=str, default=None, help="Save high-level debug rows to CSV.")
    parser.add_argument("--hl_debug_interval_steps", type=int, default=10, help="Write HL debug row every N steps.")
    return parser.parse_args()


def main():
    args = parse_args()
    if _IMPORT_ERROR is not None:
        raise SystemExit(
            f"Missing dependency: {_IMPORT_ERROR}. Activate the project environment (e.g. conda env) and retry."
        )
    runner = HierarchicalRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
