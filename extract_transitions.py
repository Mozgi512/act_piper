import os
import h5py
import numpy as np
import cv2
import argparse
from tqdm import tqdm

def detect_objects(image, min_area=40): # Lowered area for semi-obscured targets
    """Detects objects based on HSV color masking (from label_dataset.py logic)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ranges = {
        'Red': [(np.array([0, 70, 30]), np.array([12, 255, 255])), # Slightly relaxed for target frame
                (np.array([168, 70, 30]), np.array([180, 255, 255]))],
        'Green': [(np.array([35, 70, 30]), np.array([85, 255, 255]))],
        'Blue': [(np.array([115, 70, 30]), np.array([128, 255, 255]))]
    }
    detected = []
    for color_name, color_ranges in ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in color_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                if 0.3 < (w/float(h)) < 3.0:
                    detected.append({'color': color_name, 'bbox': (x, y, w, h), 'center': (x+w//2, y+h//2), 'area': area})
    return detected

def extract_transition_images(input_path, output_dir, cam_name='top', step_offset=0, target_offset=120):

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with h5py.File(input_path, 'r') as root:
        img_dataset = root[f'observations/images/{cam_name}']
        num_frames = img_dataset.shape[0]
        
        left_segments = root['metadata/left_segments'][()]
        right_segments = root['metadata/right_segments'][()]
        
        l_segs = []
        for s in left_segments:
            l_segs.append((int(s['start']), int(s['end']), s['type'], int(s['coop_split']), s['top_arm']))
        r_segs = []
        for s in right_segments:
            r_segs.append((int(s['start']), int(s['end']), s['type'], int(s['coop_split']), s['top_arm']))
        
        starts = sorted(list(set([s[0] for s in l_segs] + [s[0] for s in r_segs])))
        
        episode_name = os.path.splitext(os.path.basename(input_path))[0]
        episode_out_dir = os.path.join(output_dir, episode_name)
        if not os.path.exists(episode_out_dir):
            os.makedirs(episode_out_dir)

        print(f"Processing {input_path}, found {len(starts)} transition points.")

        # Prepare crop positions (H, W determined from first frame)
        sample_img = img_dataset[0]
        H, W, _ = sample_img.shape
        crop_y1, crop_y2 = H//4, H//2

        for idx in starts:
            # 1. Start image extraction (at transition moment)
            start_img_full = img_dataset[idx]
            start_crop = start_img_full[crop_y1:crop_y2, :]
            start_crop_bgr = cv2.cvtColor(start_crop, cv2.COLOR_RGB2BGR)

            # 2. Target identification at target_idx
            target_idx = min(idx + target_offset, num_frames - 1)
            target_img_full = img_dataset[target_idx]
            target_crop = target_img_full[crop_y1:crop_y2, :]
            target_crop_bgr = cv2.cvtColor(target_crop, cv2.COLOR_RGB2BGR)

            hsv = cv2.cvtColor(target_crop, cv2.COLOR_RGB2HSV)
            v_channel = hsv[:, :, 2]
            s_channel = hsv[:, :, 1]
            # --- Temporal Pixel Change Analysis ---
            # Calculate absolute difference between start and target crops
            diff_img = cv2.absdiff(start_crop, target_crop)
            diff_gray = cv2.cvtColor(diff_img, cv2.COLOR_RGB2GRAY)
            
            # --- Gripper Localization via diff_mask ---
            # Create the arm mask (Color filter applied to diff)
            lower = np.array([0, 0, 0])
            upper = np.array([180, 10, 45])
            mask = cv2.inRange(hsv, lower, upper)
            diff_mask = cv2.bitwise_and(diff_img, diff_img, mask=mask)

            gripper_centers = {} # side: (x, y)
            
            # Binary mask for scanning
            arm_gray = cv2.cvtColor(diff_mask, cv2.COLOR_RGB2GRAY)
            _, arm_binary = cv2.threshold(arm_gray, 5, 255, cv2.THRESH_BINARY)
            
            mid_x = W // 2
            for side in ['L', 'R']:
                if side == 'L':
                    side_mask = arm_binary[:, :mid_x]
                    offset_x = 0
                else:
                    side_mask = arm_binary[:, mid_x:]
                    offset_x = mid_x
                    
                cnts, _ = cv2.findContours(side_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts: continue
                
                # Arm is the largest contour in the half-image
                arm_cnt = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(arm_cnt) < 50: continue
                
                # Find extremities
                if side == 'L':
                    # Rightmost point
                    x_ext = np.max(arm_cnt[:, 0, 0])
                    x_scan = x_ext - 5
                else:
                    # Leftmost point
                    x_ext = np.min(arm_cnt[:, 0, 0])
                    x_scan = x_ext + 5
                
                # Vertical scan for fingers
                if 0 <= x_scan < side_mask.shape[1]:
                    column = side_mask[:, x_scan]
                    y_indices = np.where(column > 0)[0]
                    
                    if len(y_indices) >= 2:
                        # Cluster into fingers
                        clusters = []
                        current_cluster = [y_indices[0]]
                        for i in range(1, len(y_indices)):
                            if y_indices[i] - y_indices[i-1] <= 4: # Gap tolerance
                                current_cluster.append(y_indices[i])
                            else:
                                clusters.append(current_cluster)
                                current_cluster = [y_indices[i]]
                        clusters.append(current_cluster)
                        
                        if len(clusters) >= 2:
                            # Use top-most and bottom-most clusters
                            y1 = np.mean(clusters[0])
                            y2 = np.mean(clusters[-1])
                            
                            # Max 30px gap constraint
                            if abs(y1 - y2) <= 30:
                                grip_y = int((y1 + y2) / 2)
                                grip_x = int(x_scan + offset_x)
                                gripper_centers[side] = (grip_x, grip_y)
                                
                                # Draw 1x3 white cross on diff_mask
                                # Thickness 1px, length 3px
                                cv2.line(diff_mask, (grip_x - 1, grip_y), (grip_x + 1, grip_y), (255, 255, 255), 1)
                                cv2.line(diff_mask, (grip_x, grip_y - 1), (grip_x, grip_y + 1), (255, 255, 255), 1)

            # Target object detection
            target_objs = detect_objects(target_crop, min_area=30)
            held_targets = []
            
            for obj in target_objs:
                ox, oy, ow, oh = obj['bbox']
                oc = obj['center']
                
                # Determine holding based on proximity to detected gripper centers
                for side, (gx, gy) in gripper_centers.items():
                    dist = np.sqrt((oc[0] - gx)**2 + (oc[1] - gy)**2)
                    if dist < 20: # Match threshold
                        obj['held_by'] = side
                        held_targets.append(obj)
                        print(f"    - Obj at {oc} matched to {side} gripper (dist={dist:.1f})")
                        break
                
                # Keep mean_diff for logging
                roi_diff = diff_gray[oy:oy+oh, ox:ox+ow]
                obj['mean_diff'] = np.mean(roi_diff)
                print(f"    - Obj at {oc}: Pixel Change={obj['mean_diff']:.2f}")

            # 2. Display image extraction at viz_idx
            viz_idx = idx + step_offset
            if viz_idx < 0 or viz_idx >= num_frames:
                continue
                
            viz_img_full = img_dataset[viz_idx] # INDEXED READ
            viz_crop = viz_img_full[crop_y1:crop_y2, :].copy()
            viz_objs = detect_objects(viz_crop, min_area=60)
            
            # Sort Symmetrically
            mid_x = W // 2
            left_objs = sorted([o for o in viz_objs if o['center'][0] < mid_x], key=lambda o: o['center'][0], reverse=True)
            right_objs = sorted([o for o in viz_objs if o['center'][0] >= mid_x], key=lambda o: o['center'][0])
            
            # 3. Conveyor shift calculation
            shift_x = 0
            if target_objs and viz_objs:
                shifts = []
                for t_obj in target_objs:
                    # Skip if held
                    if any(np.array_equal(t_obj['center'], h['center']) for h in held_targets):
                        continue
                    for v_obj in viz_objs:
                        if t_obj['color'] == v_obj['color'] and abs(t_obj['center'][1] - v_obj['center'][1]) < 15:
                            shifts.append(t_obj['center'][0] - v_obj['center'][0])
                if shifts:
                    shift_x = np.median(shifts)
                    print(f"  Detected Conveyor Shift X: {shift_x:.2f}")
            
            # Identify Targets in viz_objs
            print(f"  Held Targets: {len(held_targets)}")
            target_labels = []
            for h in held_targets:
                expected_x = h['center'][0] - shift_x
                print(f"    - Held Obj Color: {h['color']} at {h['center']}, Expected X in Viz: {expected_x:.2f}")
                best_match = None
                min_dx = 35
                for i, v_obj in enumerate(left_objs):
                    dx = abs(v_obj['center'][0] - expected_x)
                    print(f"      - Checking L{i} ({v_obj['color']} at {v_obj['center'][0]}): dx={dx:.2f}")
                    if v_obj['color'] == h['color'] and dx < min_dx:
                        best_match = f"L{i}"
                        min_dx = dx
                for i, v_obj in enumerate(right_objs):
                    dx = abs(v_obj['center'][0] - expected_x)
                    print(f"      - Checking R{i} ({v_obj['color']} at {v_obj['center'][0]}): dx={dx:.2f}")
                    if v_obj['color'] == h['color'] and dx < min_dx:
                        best_match = f"R{i}"
                        min_dx = dx
                if best_match:
                    print(f"      -> Matched to {best_match}")
                    if best_match not in target_labels:
                        target_labels.append(best_match)
                else:
                    print("      -> No match found")

            # Visualization
            viz_bgr = cv2.cvtColor(viz_crop, cv2.COLOR_RGB2BGR)
            cv2.line(viz_bgr, (mid_x, 0), (mid_x, crop_y2-crop_y1), (255, 255, 255), 1)
            color_map = {'Red': (0, 0, 255), 'Green': (0, 255, 0), 'Blue': (255, 0, 0)}
            
            # Label arms state
            l_state, r_state = "None", "None"
            for s, e, state, _, _ in l_segs:
                if s <= idx < e:
                    l_state = state.decode('utf-8') if isinstance(state, bytes) else state
                    break
            for s, e, state, _, _ in r_segs:
                if s <= idx < e:
                    r_state = state.decode('utf-8') if isinstance(state, bytes) else state
                    break

            # Draw Grip Points (Crosses) - NO LONGER USED FOR DETECTION LOGIC but useful for viz if we have a "best cnt"
            # Actually, let's visualize the Proximity check result
            
            # Draw boxes
            for i, obj in enumerate(left_objs):
                x, y, w, h = obj['bbox']
                lbl = f"L{i}"
                is_target = lbl in target_labels
                cv2.rectangle(viz_bgr, (x, y), (x+w, y+h), color_map.get(obj['color']), 2 if is_target else 1)
                cv2.putText(viz_bgr, f"{lbl}:{obj['color'][0]}{'*' if is_target else ''}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            for i, obj in enumerate(right_objs):
                x, y, w, h = obj['bbox']
                lbl = f"R{i}"
                is_target = lbl in target_labels
                cv2.rectangle(viz_bgr, (x, y), (x+w, y+h), color_map.get(obj['color']), 2 if is_target else 1)
                cv2.putText(viz_bgr, f"{lbl}:{obj['color'][0]}{'*' if is_target else ''}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # --- 4-Dim Labeling and Anomaly Filtering ---
            # Map Object IDs: L3:0, L2:1, L1:2, L0:3, R0:4, R1:5, NONE:6
            OBJ_MAP = {'L3': 0, 'L2': 1, 'L1': 2, 'L0': 3, 'R0': 4, 'R1': 5, 'NONE': 6}
            
            # Collect all targets with their X positions in viz_img
            targets_with_x = []
            for t in target_labels:
                side = 'L' if t.startswith('L') else 'R'
                idx_in_side = int(t[1:])
                objs = left_objs if side == 'L' else right_objs
                if idx_in_side < len(objs):
                    x_pos = objs[idx_in_side]['center'][0]
                    targets_with_x.append((t, x_pos))
            
            # Sort targets by X (left to right)
            targets_with_x.sort(key=lambda x: x[1])
            sorted_targets = [t[0] for t in targets_with_x]
            
            # Anomaly checks
            is_anomaly = False
            anomaly_reason = ""
            if not target_labels: 
                is_anomaly = True
                anomaly_reason = "no_target"
            elif len(target_labels) > 2: 
                is_anomaly = True 
                anomaly_reason = "too_many_total_objs"
            
            # Determine Modes (0: INDEP, 1: COOP/HOLD)
            l_mode = 0 if l_state == 'independent' else 1
            r_mode = 0 if r_state == 'independent' else 1
            
            # Constraint: "1 object -> 1 hand independent, 1 hand coop/hold"
            if not is_anomaly and len(sorted_targets) == 1:
                if l_mode == 1 and r_mode == 1:
                    is_anomaly = True
                    anomaly_reason = "mode_mismatch_single_obj_no_indep"
            
            if is_anomaly:
                if anomaly_reason != "no_target": 
                    print(f"  [Anomaly] {anomaly_reason} for frame {idx}: targets={target_labels}")
                    anomaly_dir = os.path.join(output_dir, "anomalies")
                    if not os.path.exists(anomaly_dir): os.makedirs(anomaly_dir)
                    img_name = f"ep{episode_name}_{idx:04d}_{anomaly_reason}.png"
                    cv2.imwrite(os.path.join(anomaly_dir, img_name), viz_bgr)
                continue
                
            # Final indices: "the left hand takes the one further left"
            if len(sorted_targets) == 2:
                lo_id, ro_id = sorted_targets[0], sorted_targets[1]
            elif len(sorted_targets) == 1:
                if l_mode == 0: # Left is approaching
                    lo_id, ro_id = sorted_targets[0], 'NONE'
                else: # Right is approaching
                    lo_id, ro_id = 'NONE', sorted_targets[0]
            else:
                lo_id, ro_id = 'NONE', 'NONE'
                
            lo_idx = OBJ_MAP[lo_id]
            ro_idx = OBJ_MAP[ro_id]
            
            # Folder name format: LM{m0}_RM{m1}_LO{o0}_RO{o1}
            t_label_str = f"LM{l_mode}_RM{r_mode}_LO{lo_idx}_RO{ro_idx}"
            print(f"[Summary] Frame {idx}: 4-Dim Label = {t_label_str}")
            
            class_out_dir = os.path.join(episode_out_dir, t_label_str)
            if not os.path.exists(class_out_dir):
                os.makedirs(class_out_dir)

            cv2.putText(viz_bgr, f"L: {l_state} | R: {r_state}", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            cv2.putText(viz_bgr, f"Target: {t_label_str}", (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            cv2.imwrite(os.path.join(class_out_dir, f"start_frame_{idx:04d}.png"), start_crop_bgr)
            cv2.imwrite(os.path.join(class_out_dir, f"debug_frame_{idx:04d}.png"), viz_bgr)
            
            # --- DEBUG: Save Target Frame with Gripper Overlay ---
            debug_target = target_crop_bgr.copy()
            
            # Draw Gripper Centers on Debug Target
            for side, (gx, gy) in gripper_centers.items():
                color = (255, 0, 0) if side == 'L' else (0, 0, 255) # Blue for L, Red for R
                cv2.line(debug_target, (gx - 3, gy), (gx + 3, gy), color, 1)
                cv2.line(debug_target, (gx, gy - 3), (gx, gy + 3), color, 1)
                # cv2.putText(debug_target, side, (gx + 5, gy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            for obj in target_objs:
                # Mark detected objects on target frame too
                x, y, w, h = obj['bbox']
                color = color_map.get(obj['color'])
                cv2.rectangle(debug_target, (x, y), (x+w, y+h), color, 1)
                
                if 'held_by' in obj:
                     cv2.putText(debug_target, f"HELD:{obj['held_by']}", (x, y+h+20 if y+h+20 < H else y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw Change Info
                if 'mean_diff' in obj:
                     cv2.putText(debug_target, f"Chg:{obj['mean_diff']:.1f}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # --- DEBUG: Save Start Frame and Difference Image ---
            cv2.imwrite(os.path.join(class_out_dir, f"start_frame_{idx}.png"), start_crop_bgr)
            cv2.imwrite(os.path.join(class_out_dir, f"diff_frame_{idx}.png"), diff_mask)
            
            cv2.imwrite(os.path.join(class_out_dir, f"debug_target_frame_{idx:04d}.png"), debug_target)
            
    print(f"Done. Images saved to {episode_out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='transition_images')
    parser.add_argument('--cam_name', type=str, default='top')
    parser.add_argument('--step_offset', type=int, default=0)
    parser.add_argument('--target_offset', type=int, default=120)
    args = parser.parse_args()
    
    if os.path.isdir(args.input_path):
        files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.endswith('.hdf5')]
        for f in sorted(files):
            extract_transition_images(f, args.output_dir, args.cam_name, args.step_offset, args.target_offset)
    else:
        extract_transition_images(args.input_path, args.output_dir, args.cam_name, args.step_offset, args.target_offset)

if __name__ == '__main__':
    main()
