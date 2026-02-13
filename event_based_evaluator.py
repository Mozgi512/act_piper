import os
import sys
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'detr'))

from piper_constants import DT, START_ARM_POSE, SIM_TASK_CONFIGS
from piper_sim_env import make_sim_env, MANYCUBES_COLORS
from utils import set_seed

from state_switcher import (
    load_policy_and_stats,
    load_state_classifier,
    get_image_dual,
    get_image_independent,
    sync_envs,
    hide_objects,
    get_grasped_cubes,
    apply_magnet_logic,
    reset_magnet_logic,
    remove_cubes,
)
from train_state_classifier import STATE_HOLD, STATE_INDEP, STATE_COOP


CLS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def is_arm_at_home(current_qpos, home_qpos, threshold=0.12):
    return np.max(np.abs(current_qpos - home_qpos)) < threshold


def is_removed_cube(physics, idx):
    try:
        geom_id = physics.model.name2id(f'cube_{idx}', 'geom')
        alpha = float(physics.model.geom_rgba[geom_id, 3])
    except Exception:
        alpha = 1.0
    try:
        body_id = physics.model.name2id(f'cube_{idx}', 'body')
        z_pos = float(physics.data.xpos[body_id][2])
    except Exception:
        z_pos = 0.0
    return (alpha <= 0.01) or (z_pos < -1.0)


def get_goal_object_indices(physics):
    indices = []
    all_contacts = set()
    for i in range(physics.data.ncon):
        id1, id2 = physics.data.contact[i].geom1, physics.data.contact[i].geom2
        n1 = physics.model.id2name(id1, 'geom')
        n2 = physics.model.id2name(id2, 'geom')
        if n1 and n2:
            all_contacts.add((n1, n2))
            all_contacts.add((n2, n1))

    for i in range(10):
        cube_name = f'cube_{i}'
        if ('goal_plate', cube_name) in all_contacts:
            indices.append(i)
            continue
        try:
            bid = physics.model.name2id(cube_name, 'body')
            pos = physics.data.xpos[bid]
            if abs(pos[0]) < 0.1 and 0.04 < pos[1] < 0.14 and pos[2] < 0.15:
                indices.append(i)
        except Exception:
            pass
    return list(set(indices))


def run_hl_query(cls_model, ts):
    full_img = ts.observation['images']['top']
    img_t = CLS_TRANSFORM(Image.fromarray(full_img.astype('uint8'))).unsqueeze(0).cuda()
    with torch.no_grad():
        out_lm, out_rm = cls_model(img_t)
        l_state = int(torch.argmax(out_lm, 1).item())
        r_state = int(torch.argmax(out_rm, 1).item())

    # Event evaluator currently uses binary low-level routing: INDEP vs COOP.
    # HOLD is mapped to INDEP by default.
    l_mode = 1 if l_state == STATE_COOP else 0
    r_mode = 1 if r_state == STATE_COOP else 0
    return {
        'l_mode': l_mode,
        'r_mode': r_mode,
        'l_state': l_state,
        'r_state': r_state,
    }


def choose_next_task(hl_out, physics, color_sequence, removed_objects):
    def safe_color(i):
        try:
            return color_sequence[i]
        except Exception:
            return None

    def safe_x(i):
        try:
            bid = physics.model.name2id(f'cube_{i}', 'body')
            return float(physics.data.xpos[bid][0])
        except Exception:
            return None

    def pick_indep_target(arm):
        candidates = []
        for i in range(10):
            if i in removed_objects or is_removed_cube(physics, i):
                continue
            if safe_color(i) != 'r':
                continue
            x_val = safe_x(i)
            if x_val is None:
                continue
            if arm == 'left' and x_val < 0:
                candidates.append((x_val, i))
            if arm == 'right' and x_val >= 0:
                candidates.append((x_val, i))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p[0])[1]

    def pick_coop_pair():
        g_candidates = []
        b_candidates = []
        for i in range(10):
            if i in removed_objects or is_removed_cube(physics, i):
                continue
            c = safe_color(i)
            if c not in ['g', 'b']:
                continue
            x_val = safe_x(i)
            if x_val is None or x_val >= 0.3:
                continue
            if c == 'g':
                g_candidates.append((x_val, i))
            else:
                b_candidates.append((x_val, i))
        if not g_candidates or not b_candidates:
            return None, None
        return max(g_candidates, key=lambda p: p[0])[1], max(b_candidates, key=lambda p: p[0])[1]

    l_mode = hl_out['l_mode']
    r_mode = hl_out['r_mode']

    if l_mode == 0:
        l_obj = pick_indep_target('left')
        if l_obj is not None:
            return {'mode': 'INDEP', 'arm': 'left', 'target': l_obj}
    if r_mode == 0:
        r_obj = pick_indep_target('right')
        if r_obj is not None:
            return {'mode': 'INDEP', 'arm': 'right', 'target': r_obj}

    if l_mode == 1 and r_mode == 1:
        l_obj, r_obj = pick_coop_pair()
        if l_obj is None or r_obj is None:
            return None
        return {
            'mode': 'COOP',
            'left_target': l_obj,
            'right_target': r_obj,
            'assembled': False,
            'top_arm': None,
        }

    return None


def main(args):
    set_seed(args.seed)

    if args.color_sequence:
        seq = args.color_sequence.lower().strip()
        if len(seq) == 10:
            MANYCUBES_COLORS[0] = list(seq)

    print('Loading state classifier...')
    cls_model = load_state_classifier(args.state_ckpt)

    print('Loading low-level policies...')
    policy_dual, stats_dual = load_policy_and_stats(args.ckpt_dual, 'ACT', args, override_state_dim=14)
    policy_left, stats_left = load_policy_and_stats(args.ckpt_left, 'ACT', args, override_state_dim=7, override_arm='left')
    policy_right, stats_right = load_policy_and_stats(args.ckpt_right, 'ACT', args, override_state_dim=7, override_arm='right')

    pre_process_dual = lambda s: (s - stats_dual['qpos_mean']) / stats_dual['qpos_std']
    post_process_dual = lambda a: a * stats_dual['action_std'] + stats_dual['action_mean']
    pre_process_left = lambda s: (s - stats_left['qpos_mean']) / stats_left['qpos_std']
    post_process_left = lambda a: a * stats_left['action_std'] + stats_left['action_mean']
    pre_process_right = lambda s: (s - stats_right['qpos_mean']) / stats_right['qpos_std']
    post_process_right = lambda a: a * stats_right['action_std'] + stats_right['action_mean']

    task_config = SIM_TASK_CONFIGS[args.task_name]
    max_steps = args.episode_len if args.episode_len is not None else task_config['episode_len']
    time_limit = (max_steps + 200) * DT

    env_main = make_sim_env(args.task_name, camera_names=['top'], time_limit=time_limit)
    env_shadow_c = make_sim_env(args.task_name, camera_names=['top'], time_limit=time_limit)
    env_shadow_i_left = make_sim_env(args.task_name, camera_names=['top'], time_limit=time_limit)
    env_shadow_i_right = make_sim_env(args.task_name, camera_names=['top'], time_limit=time_limit)

    home_pose = np.zeros(14)
    home_pose[:6] = START_ARM_POSE[:6]
    home_pose[6] = START_ARM_POSE[6]
    home_pose[7:13] = START_ARM_POSE[8:14]
    home_pose[13] = START_ARM_POSE[14]

    if args.onscreen_render:
        plt.ion()
        fig, (ax_main, ax_c, ax_il, ax_ir) = plt.subplots(1, 4, figsize=(20, 5))
        plt_main = ax_main.imshow(np.zeros((240, 320, 3), dtype=np.uint8))
        plt_c = ax_c.imshow(np.zeros((240, 320, 3), dtype=np.uint8))
        plt_il = ax_il.imshow(np.zeros((240, 320, 3), dtype=np.uint8))
        plt_ir = ax_ir.imshow(np.zeros((240, 320, 3), dtype=np.uint8))
        ax_main.set_title('Main')
        ax_c.set_title('Coop Shadow')
        ax_il.set_title('Indep Left')
        ax_ir.set_title('Indep Right')

    for ep in range(args.num_rollouts):
        print(f'\n=== Episode {ep} ===')
        ts = env_main.reset()
        env_shadow_c.reset()
        env_shadow_i_left.reset()
        env_shadow_i_right.reset()

        reset_magnet_logic(env_main.physics, args.color_sequence)
        magnetized_pairs = {}
        removed_objects = set()

        active_task = None
        current_chunk_dual = None
        current_chunk_left = None
        current_chunk_right = None
        step_in_chunk_dual = 0
        step_in_chunk_left = 0
        step_in_chunk_right = 0

        for t in range(max_steps):
            qpos = np.array(ts.observation['qpos'])
            left_home = is_arm_at_home(qpos[:7], home_pose[:7], threshold=args.switch_home_threshold)
            right_home = is_arm_at_home(qpos[7:14], home_pose[7:14], threshold=args.switch_home_threshold)

            for idx in get_goal_object_indices(env_main.physics):
                if idx not in removed_objects:
                    removed_objects.add(idx)
            if removed_objects:
                remove_cubes(env_main.physics, list(removed_objects))

            sync_envs(env_main.physics, env_shadow_c.physics)
            sync_envs(env_main.physics, env_shadow_i_left.physics)
            sync_envs(env_main.physics, env_shadow_i_right.physics)

            # Schedule when idle
            if active_task is None:
                hl_out = run_hl_query(cls_model, ts)
                active_task = choose_next_task(hl_out, env_main.physics, args.color_sequence, removed_objects)
                if active_task is not None:
                    print(f"[Step {t}] Scheduled: {active_task}")
                    current_chunk_dual = None
                    current_chunk_left = None
                    current_chunk_right = None
                    step_in_chunk_dual = 0
                    step_in_chunk_left = 0
                    step_in_chunk_right = 0

            # COOP exception: assembly completed + top arm home => allow independent next schedule
            if active_task is not None and active_task['mode'] == 'COOP' and active_task.get('assembled', False):
                top_arm = active_task.get('top_arm', None)
                top_home = (left_home if top_arm == 'left' else right_home) if top_arm in ['left', 'right'] else False
                if top_home:
                    hl_out = run_hl_query(cls_model, ts)
                    if hl_out['l_mode'] == 0 or hl_out['r_mode'] == 0:
                        nxt = choose_next_task(hl_out, env_main.physics, args.color_sequence, removed_objects)
                        if nxt is not None and nxt['mode'] == 'INDEP':
                            print(f"[Step {t}] COOP exception scheduling: {nxt}")
                            active_task = nxt
                            current_chunk_dual = None
                            current_chunk_left = None
                            current_chunk_right = None
                            step_in_chunk_dual = 0
                            step_in_chunk_left = 0
                            step_in_chunk_right = 0

            indep_left_targets = []
            indep_right_targets = []
            coop_targets = []

            if active_task is not None:
                if active_task['mode'] == 'INDEP':
                    target = int(active_task['target'])
                    if active_task['arm'] == 'left':
                        indep_left_targets = [target]
                    else:
                        indep_right_targets = [target]
                else:
                    coop_targets = [int(active_task['left_target']), int(active_task['right_target'])]

            hide_objects(env_shadow_i_left.physics, indep_left_targets, 'IndepLeft')
            hide_objects(env_shadow_i_right.physics, indep_right_targets, 'IndepRight')
            hide_objects(env_shadow_c.physics, coop_targets, 'Coop')

            action = qpos.copy()

            with torch.inference_mode():
                if active_task is not None:
                    if active_task['mode'] == 'INDEP':
                        if active_task['arm'] == 'left':
                            if current_chunk_left is None or step_in_chunk_left >= args.chunk_size:
                                obs_il = env_shadow_i_left.task.get_observation(env_shadow_i_left.physics)
                                ts_il = type('TS', (object,), {'observation': obs_il})()
                                q_l = torch.from_numpy(pre_process_left(qpos[:7])).float().cuda().unsqueeze(0)
                                img_l = get_image_independent(ts_il, ['top'], 'left', mask=False)
                                out_l = policy_left(q_l, img_l)[0].cpu().numpy()
                                current_chunk_left = post_process_left(out_l)
                                step_in_chunk_left = 0
                            action[:7] = current_chunk_left[step_in_chunk_left]
                            step_in_chunk_left += 1
                        else:
                            if current_chunk_right is None or step_in_chunk_right >= args.chunk_size:
                                obs_ir = env_shadow_i_right.task.get_observation(env_shadow_i_right.physics)
                                ts_ir = type('TS', (object,), {'observation': obs_ir})()
                                q_r = torch.from_numpy(pre_process_right(qpos[7:14])).float().cuda().unsqueeze(0)
                                img_r = get_image_independent(ts_ir, ['top'], 'right', mask=False)
                                out_r = policy_right(q_r, img_r)[0].cpu().numpy()
                                current_chunk_right = post_process_right(out_r)
                                step_in_chunk_right = 0
                            action[7:14] = current_chunk_right[step_in_chunk_right]
                            step_in_chunk_right += 1
                    else:
                        if current_chunk_dual is None or step_in_chunk_dual >= args.chunk_size:
                            obs_c = env_shadow_c.task.get_observation(env_shadow_c.physics)
                            ts_c = type('TS', (object,), {'observation': obs_c})()
                            q_c = torch.from_numpy(pre_process_dual(qpos)).float().cuda().unsqueeze(0)
                            img_c = get_image_dual(ts_c, ['top'], mask=False)
                            out_c = policy_dual(q_c, img_c)[0].cpu().numpy()
                            current_chunk_dual = post_process_dual(out_c)
                            step_in_chunk_dual = 0
                        action[:] = current_chunk_dual[step_in_chunk_dual]
                        step_in_chunk_dual += 1

            ts = env_main.step(action)

            apply_magnet_logic(env_main.physics, magnetized_pairs, args.color_sequence)

            # Detect COOP assembly completion event
            if active_task is not None and active_task['mode'] == 'COOP' and not active_task.get('assembled', False):
                if len(magnetized_pairs) > 0:
                    active_task['assembled'] = True
                    grasped = get_grasped_cubes(env_main.physics)
                    lt = active_task['left_target']
                    rt = active_task['right_target']
                    if lt in grasped['left'] or rt in grasped['left']:
                        active_task['top_arm'] = 'left'
                    elif lt in grasped['right'] or rt in grasped['right']:
                        active_task['top_arm'] = 'right'
                    else:
                        active_task['top_arm'] = 'left'
                    print(f"[Step {t}] COOP assembled (magnet). top_arm={active_task['top_arm']}")

            # Completion event handling
            if active_task is not None:
                if active_task['mode'] == 'INDEP':
                    tgt = int(active_task['target'])
                    arm = active_task['arm']
                    arm_home = left_home if arm == 'left' else right_home
                    done_removed = is_removed_cube(env_main.physics, tgt) or (tgt in removed_objects)
                    if done_removed and arm_home:
                        print(f"[Step {t}] Task done: INDEP {arm} target={tgt}")
                        active_task = None
                else:
                    lt = int(active_task['left_target'])
                    rt = int(active_task['right_target'])
                    removed_pair = (is_removed_cube(env_main.physics, lt) or lt in removed_objects) and \
                                   (is_removed_cube(env_main.physics, rt) or rt in removed_objects)
                    if removed_pair and left_home and right_home:
                        print(f"[Step {t}] Task done: COOP pair=({lt}, {rt})")
                        active_task = None

            if args.onscreen_render and t % 5 == 0:
                plt_main.set_data(ts.observation['images']['top'])
                plt_c.set_data(env_shadow_c.physics.render(height=240, width=320, camera_id='top'))
                plt_il.set_data(env_shadow_i_left.physics.render(height=240, width=320, camera_id='top'))
                plt_ir.set_data(env_shadow_i_right.physics.render(height=240, width=320, camera_id='top'))
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

        print(f'Episode {ep} finished.')

    print('All done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--state_ckpt', type=str, required=True)
    parser.add_argument('--ckpt_dual', type=str, required=True)
    parser.add_argument('--ckpt_left', type=str, required=True)
    parser.add_argument('--ckpt_right', type=str, required=True)

    parser.add_argument('--chunk_size', type=int, default=75)
    parser.add_argument('--kl_weight', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=3200)
    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument('--num_rollouts', type=int, default=1)
    parser.add_argument('--episode_len', type=int, default=2700)
    parser.add_argument('--color_sequence', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--switch_home_threshold', type=float, default=0.12)

    args = parser.parse_args()
    main(args)
